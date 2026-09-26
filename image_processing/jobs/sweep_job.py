"""Expands a sweep grid into individual RunConfigs and runs each as its own subprocess,
sequentially (single-GPU assumption). Writes progress to sweeps/<sweep_id>/status.json
and appends a row to results.csv as each grid point finishes, so a sweep killed partway
through still leaves usable partial results.
"""
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

from image_processing.core import repository as repo
from image_processing.core.config_schema import ModelConfig, PreprocessConfig, RunConfig
from image_processing.core.project import Project
from image_processing.core.store import write_json
from image_processing.ml import metrics as ml_metrics
from image_processing.preprocessing.pipeline import load_image

RESULTS_FIELDS = [
    "run_id",
    "preproc_hash",
    "backbone",
    "layers",
    "coreset_sampling_ratio",
    "num_neighbors",
    "patch_size",
    "auroc",
    "f1_anomaly",
    "recall_anomaly",
    "n_predictions",
    "status",
]


def _set_status(sweep_dir: Path, state: str, current_index: int, total: int, message: str = "") -> None:
    write_json(
        sweep_dir / "status.json",
        {"state": state, "current_index": current_index, "total": total, "message": message},
    )


def _append_result_row(sweep_dir: Path, row: dict) -> None:
    results_path = sweep_dir / "results.csv"
    write_header = not results_path.exists()
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _image_size_for_hash(project: Project, preproc_hash: str) -> tuple[int, int]:
    # Different preprocessing variants can legitimately produce different output
    # resolutions, so this must be resolved per-hash, not once globally.
    cache_dir = project.preprocessing_cache_dir / preproc_hash
    sample_files = list((cache_dir / "good").glob("*.png")) or list((cache_dir / "anomaly").glob("*.png"))
    if not sample_files:
        return (256, 256)
    height, width = load_image(sample_files[0]).shape[:2]
    return (int(height), int(width))


def _dataset_for_config(project: Project, preprocess_config: PreprocessConfig) -> tuple[list[str], list[str], list[str]]:
    # Different preprocessing configs in the same sweep can carry different category
    # scopes (a fork config and a spoon config), so the train/val image sets must be
    # resolved per-config, not once globally — mirrors _image_size_for_hash above for the
    # same underlying reason.
    categories = preprocess_config.categories
    train_ids = [
        iid
        for iid in repo.list_by_label_and_split(project, "good", "train")
        if repo.image_matches_categories(project, iid, categories)
    ]
    val_good_ids = [
        iid
        for iid in repo.list_by_label_and_split(project, "good", "val")
        if repo.image_matches_categories(project, iid, categories)
    ]
    val_anomaly_ids = [
        iid
        for iid in repo.list_by_label_and_split(project, "anomaly", "val")
        if repo.image_matches_categories(project, iid, categories)
    ]
    return train_ids, val_good_ids, val_anomaly_ids


def run_sweep(project: Project, sweep_id: str, grid: dict) -> dict:
    sweep_dir = project.sweeps_dir / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    write_json(sweep_dir / "grid.json", grid)

    combos = list(
        itertools.product(
            grid["preproc_hashes"],
            grid["backbones"],
            grid["layer_sets"],
            grid["coreset_ratios"],
            grid["num_neighbors_list"],
            grid["patch_sizes"],
        )
    )
    total = len(combos)
    _set_status(sweep_dir, "running", 0, total, "starting")

    image_size_cache: dict[str, tuple[int, int]] = {}
    preprocess_config_cache: dict[str, PreprocessConfig] = {}
    dataset_cache: dict[str, tuple[list[str], list[str], list[str]]] = {}

    for i, (preproc_hash, backbone, layers, coreset_ratio, num_neighbors, patch_size) in enumerate(combos):
        run_id = project.new_id()
        _set_status(sweep_dir, "running", i, total, f"run {i + 1}/{total}: {run_id}")

        if preproc_hash not in image_size_cache:
            image_size_cache[preproc_hash] = _image_size_for_hash(project, preproc_hash)
        if preproc_hash not in preprocess_config_cache:
            config_path = project.preprocessing_configs_dir / f"{preproc_hash}.json"
            preprocess_config_cache[preproc_hash] = PreprocessConfig.model_validate(
                json.loads(config_path.read_text())
            )
        if preproc_hash not in dataset_cache:
            dataset_cache[preproc_hash] = _dataset_for_config(project, preprocess_config_cache[preproc_hash])
        train_ids, val_good_ids, val_anomaly_ids = dataset_cache[preproc_hash]

        run_config = RunConfig(
            preproc_hash=preproc_hash,
            preprocess=preprocess_config_cache[preproc_hash],
            model=ModelConfig(
                backbone=backbone,
                layers=layers,
                coreset_sampling_ratio=coreset_ratio,
                num_neighbors=num_neighbors,
                patch_size=patch_size,
            ),
            train_image_ids=train_ids,
            val_good_image_ids=val_good_ids,
            val_anomaly_image_ids=val_anomaly_ids,
            image_size=image_size_cache[preproc_hash],
        )

        run_dir = project.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "config.json", run_config.model_dump())

        result_row = {
            "run_id": run_id,
            "preproc_hash": preproc_hash,
            "backbone": backbone,
            "layers": "+".join(layers),
            "coreset_sampling_ratio": coreset_ratio,
            "num_neighbors": num_neighbors,
            "patch_size": patch_size,
            "auroc": "",
            "f1_anomaly": "",
            "recall_anomaly": "",
            "n_predictions": "",
            "status": "failed",
        }
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "image_processing.jobs.train_job",
                    "--project-root",
                    str(project.root),
                    "--run-id",
                    run_id,
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"train_job exited {proc.returncode}: {proc.stderr[-2000:]}")

            summary = ml_metrics.summarize(run_dir / "predictions.csv")
            result_row.update(
                {
                    "auroc": summary["auroc"],
                    "f1_anomaly": summary["at_threshold"]["f1_anomaly"],
                    "recall_anomaly": summary["at_threshold"]["recall_anomaly"],
                    "n_predictions": summary["n_good"] + summary["n_anomaly"],
                    "status": "done",
                }
            )
        except Exception as exc:  # noqa: BLE001 - one bad grid point shouldn't kill the sweep
            result_row["status"] = f"failed: {exc}"

        _append_result_row(sweep_dir, result_row)

    _set_status(sweep_dir, "done", total, total, f"completed {total} run(s)")
    return {"sweep_id": sweep_id, "total": total}
