"""Runs an ordered list of preprocessing steps over images, and builds the on-disk cache."""
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from image_processing.core import repository as repo
from image_processing.core.config_schema import PipelineStep, PreprocessConfig
from image_processing.core.hashing import stable_hash
from image_processing.core.project import Project
from image_processing.core.store import read_json, write_json
from image_processing.preprocessing.registry import get_tool


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def apply(image: np.ndarray, steps: list[PipelineStep]) -> np.ndarray:
    for step in steps:
        image = get_tool(step.type).apply(image, step.params)
    return image


def apply_up_to(image: np.ndarray, steps: list[PipelineStep], index: int) -> np.ndarray:
    return apply(image, steps[:index])


def missing_cached_images(project: Project, preproc_hash: str, label_to_ids: dict[str, list[str]]) -> list[str]:
    """image_ids referenced by a run that have no processed file in this preproc_hash's cache.

    Catches the common case of importing/labeling images after the cache was last built,
    so the GUI can surface a clear warning instead of a deep Anomalib FileNotFoundError.
    """
    cache_dir = project.preprocessing_cache_dir / preproc_hash
    missing = []
    for label, ids in label_to_ids.items():
        for image_id in ids:
            if not (cache_dir / label / f"{image_id}.png").exists():
                missing.append(image_id)
    return missing


def build_cache(
    project: Project, preprocess_config: PreprocessConfig, image_ids: list[str] | None = None
) -> dict:
    """Process every labeled image through the step chain, caching results by content hash.

    Returns a report dict with the resolved preproc_hash, any warnings (low-confidence
    alignment matches, inconsistent output sizes), and the set of output sizes observed.
    """
    steps = preprocess_config.steps
    config_dict = preprocess_config.model_dump()
    preproc_hash = stable_hash(config_dict)
    write_json(project.preprocessing_configs_dir / f"{preproc_hash}.json", config_dict)

    cache_dir = project.preprocessing_cache_dir / preproc_hash
    (cache_dir / "good").mkdir(parents=True, exist_ok=True)
    (cache_dir / "anomaly").mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest = read_json(manifest_path, default={})

    catalog_images = repo.list_images(project)
    if image_ids is not None:
        catalog_images = [img for img in catalog_images if img["id"] in image_ids]

    warnings: list[str] = []
    output_sizes: set[tuple[int, int]] = set()

    for img in catalog_images:
        image_id = img["id"]
        label = repo.get_label(project, image_id)["label"]
        if label not in ("good", "anomaly"):
            continue

        src_path = project.images_raw_dir / img["relpath"]
        src_stat = src_path.stat()
        source_sig = f"{src_stat.st_mtime_ns}:{src_stat.st_size}"

        target_rel = f"{label}/{image_id}.png"
        target_path = cache_dir / target_rel

        other_label = "anomaly" if label == "good" else "good"
        stale_path = cache_dir / other_label / f"{image_id}.png"
        if stale_path.exists():
            stale_path.unlink()

        cached_entry = manifest.get(image_id)
        needs_processing = (
            cached_entry is None
            or cached_entry.get("source_sig") != source_sig
            or cached_entry.get("processed_path") != target_rel
            or not target_path.exists()
        )

        if needs_processing:
            raw_image = load_image(src_path)

            for step_index, step in enumerate(steps):
                if step.type == "align":
                    align_tool = get_tool("align")
                    image_before = apply_up_to(raw_image, steps, step_index)
                    low_conf = align_tool.low_confidence_landmarks(image_before, step.params)
                    if low_conf:
                        warnings.append(
                            f"Image {image_id}: low-confidence alignment "
                            f"(step {step_index}) on landmark(s) {low_conf}"
                        )

            processed = apply(raw_image, steps)
            save_image(target_path, processed)
            manifest[image_id] = {"processed_path": target_rel, "source_sig": source_sig}
        else:
            processed = load_image(target_path)

        output_sizes.add((processed.shape[0], processed.shape[1]))

    write_json(manifest_path, manifest)

    if len(output_sizes) > 1:
        warnings.append(
            f"Inconsistent output sizes across processed images: {sorted(output_sizes)}. "
            "Add or adjust a resize step so every image ends up the same size before training."
        )

    return {
        "preproc_hash": preproc_hash,
        "cache_dir": str(cache_dir),
        "warnings": warnings,
        "output_sizes": sorted(output_sizes),
    }


def delete_config(project: Project, preproc_hash: str) -> None:
    """Removes a preprocessing config's definition, its cached processed images, and its
    nickname. Does not check whether existing runs reference this hash — the caller (the
    sidebar UI) is responsible for warning the user about that before calling this."""
    config_path = project.preprocessing_configs_dir / f"{preproc_hash}.json"
    if config_path.exists():
        config_path.unlink()
    cache_dir = project.preprocessing_cache_dir / preproc_hash
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    repo.delete_config_name(project, preproc_hash)
