"""Data-access functions over a project's catalog.json (images + labels) and its runs/sweeps directories."""
from pathlib import Path
from datetime import datetime, timezone

from image_processing.core.project import Project
from image_processing.core.store import read_json, write_json

LabelValue = str  # "good" | "anomaly" | "unlabeled"
SplitValue = str  # "train" | "val" | "unassigned"


def load_catalog(project: Project) -> dict:
    # setdefault (not just the read_json default=) so a catalog.json written before the
    # "categories" key existed still normalizes cleanly — the default= only covers a
    # missing *file*, not a missing *key* within an existing one.
    catalog = read_json(project.catalog_path, default={"images": [], "labels": {}, "categories": []})
    catalog.setdefault("images", [])
    catalog.setdefault("labels", {})
    catalog.setdefault("categories", [])
    return catalog


def save_catalog(project: Project, catalog: dict) -> None:
    write_json(project.catalog_path, catalog)


def add_image(project: Project, image_id: str, relpath: str, width: int, height: int) -> None:
    catalog = load_catalog(project)
    catalog["images"].append(
        {
            "id": image_id,
            "relpath": relpath,
            "width": width,
            "height": height,
            "imported_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    catalog["labels"].setdefault(
        image_id,
        {"label": "unlabeled", "split": "unassigned", "labeled_at": None, "notes": "", "categories": []},
    )
    save_catalog(project, catalog)


def list_images(project: Project) -> list[dict]:
    return load_catalog(project)["images"]


def get_image(project: Project, image_id: str) -> dict | None:
    for img in list_images(project):
        if img["id"] == image_id:
            return img
    return None


def set_label(
    project: Project,
    image_id: str,
    label: LabelValue | None = None,
    split: SplitValue | None = None,
    notes: str | None = None,
) -> None:
    catalog = load_catalog(project)
    entry = catalog["labels"].setdefault(
        image_id,
        {"label": "unlabeled", "split": "unassigned", "labeled_at": None, "notes": "", "categories": []},
    )
    if label is not None:
        entry["label"] = label
        entry["labeled_at"] = datetime.now(timezone.utc).isoformat()
    if split is not None:
        entry["split"] = split
    if notes is not None:
        entry["notes"] = notes
    save_catalog(project, catalog)


def _normalize_label_entry(entry: dict) -> dict:
    # Built field-by-field with individual .get(key, default) calls, not a single
    # .get(image_id, {whole default dict}) — this is what lets a new field (like
    # "categories", added after many catalogs already existed) default cleanly for every
    # pre-existing image with no migration step, instead of only for brand-new entries.
    return {
        "label": entry.get("label", "unlabeled"),
        "split": entry.get("split", "unassigned"),
        "labeled_at": entry.get("labeled_at"),
        "notes": entry.get("notes", ""),
        "categories": entry.get("categories", []),
    }


def get_label(project: Project, image_id: str) -> dict:
    entry = load_catalog(project)["labels"].get(image_id, {})
    return _normalize_label_entry(entry)


def get_all_labels(project: Project) -> dict[str, dict]:
    """Normalized label entries for every image, from a single catalog read — use this
    instead of calling get_label() once per image in a loop, which reloads and re-parses
    the whole catalog.json file on every single call and scales badly as a project grows."""
    labels = load_catalog(project)["labels"]
    return {image_id: _normalize_label_entry(entry) for image_id, entry in labels.items()}


def apply_label_updates(project: Project, updates: dict[str, dict]) -> None:
    """Applies label/split/categories changes for multiple images in one read-modify-write,
    instead of one separate catalog read-modify-write per image. `updates` is
    {image_id: {"label": ..., "split": ..., "categories": [...]}} — any field left out of
    an image's dict is untouched. Used by the Label Images gallery to save an entire batch
    of pending widget changes atomically in a single write, rather than one write per
    changed image with a script-restart in between each — the latter meant only the first
    change found in a pass ever got persisted before the script stopped to rerun, so a
    burst of rapid edits needed several uninterrupted reruns in a row to fully catch up,
    and a new incoming interaction could reset that catch-up before it finished, silently
    dropping whichever edits hadn't been reached yet.
    """
    catalog = load_catalog(project)
    for image_id, fields in updates.items():
        entry = catalog["labels"].setdefault(
            image_id,
            {"label": "unlabeled", "split": "unassigned", "labeled_at": None, "notes": "", "categories": []},
        )
        if "label" in fields:
            entry["label"] = fields["label"]
            entry["labeled_at"] = datetime.now(timezone.utc).isoformat()
        if "split" in fields:
            entry["split"] = fields["split"]
        if "categories" in fields:
            entry["categories"] = list(fields["categories"])
    save_catalog(project, catalog)


def list_by_label(project: Project, label: LabelValue) -> list[str]:
    labels = load_catalog(project)["labels"]
    return [iid for iid, entry in labels.items() if entry["label"] == label]


def list_by_label_and_split(project: Project, label: LabelValue, split: SplitValue) -> list[str]:
    labels = load_catalog(project)["labels"]
    return [
        iid for iid, entry in labels.items() if entry["label"] == label and entry["split"] == split
    ]


def image_path(project: Project, image_id: str) -> Path:
    img = get_image(project, image_id)
    if img is None:
        raise KeyError(f"Unknown image_id {image_id}")
    return project.images_raw_dir / img["relpath"]


# -- runs / sweeps (directory-scanned, small dataset scale so no index needed) --


def list_runs(project: Project) -> list[dict]:
    runs = []
    if not project.runs_dir.exists():
        return runs
    for run_dir in sorted(project.runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        config = read_json(run_dir / "config.json", default={})
        status = read_json(run_dir / "status.json", default={"state": "unknown"})
        metrics = read_json(run_dir / "metrics.json", default={})
        runs.append({"id": run_dir.name, "config": config, "status": status, "metrics": metrics})
    return runs


def list_sweeps(project: Project) -> list[dict]:
    sweeps = []
    if not project.sweeps_dir.exists():
        return sweeps
    for sweep_dir in sorted(project.sweeps_dir.iterdir()):
        if not sweep_dir.is_dir():
            continue
        grid = read_json(sweep_dir / "grid.json", default={})
        status = read_json(sweep_dir / "status.json", default={"state": "unknown"})
        sweeps.append({"id": sweep_dir.name, "grid": grid, "status": status})
    return sweeps


# -- preprocessing config nicknames --------------------------------------------------


def load_config_names(project: Project) -> dict[str, str]:
    return read_json(project.preprocessing_config_names_path, default={})


def save_config_names(project: Project, names: dict[str, str]) -> None:
    write_json(project.preprocessing_config_names_path, names)


def get_config_name(project: Project, preproc_hash: str) -> str | None:
    return load_config_names(project).get(preproc_hash)


def set_config_name(project: Project, preproc_hash: str, name: str) -> None:
    names = load_config_names(project)
    if name.strip():
        names[preproc_hash] = name.strip()
    else:
        names.pop(preproc_hash, None)
    save_config_names(project, names)


def delete_config_name(project: Project, preproc_hash: str) -> None:
    names = load_config_names(project)
    if preproc_hash in names:
        del names[preproc_hash]
        save_config_names(project, names)


# -- image categories: orthogonal, multi-valued tags independent of label/split ------


def list_categories(project: Project) -> list[str]:
    return load_catalog(project)["categories"]


def create_category(project: Project, name: str) -> None:
    name = name.strip()
    if not name:
        return
    catalog = load_catalog(project)
    if name not in catalog["categories"]:
        catalog["categories"].append(name)
        save_catalog(project, catalog)


def delete_category(project: Project, name: str) -> None:
    catalog = load_catalog(project)
    if name in catalog["categories"]:
        catalog["categories"].remove(name)
    for entry in catalog["labels"].values():
        if name in entry.get("categories", []):
            entry["categories"].remove(name)
    save_catalog(project, catalog)


def count_images_with_category(project: Project, name: str) -> int:
    labels = load_catalog(project)["labels"]
    return sum(1 for entry in labels.values() if name in entry.get("categories", []))


def set_image_categories(project: Project, image_id: str, categories: list[str]) -> None:
    """Replaces (not merges) the given image's category list."""
    catalog = load_catalog(project)
    entry = catalog["labels"].setdefault(
        image_id,
        {"label": "unlabeled", "split": "unassigned", "labeled_at": None, "notes": "", "categories": []},
    )
    entry["categories"] = list(categories)
    save_catalog(project, catalog)


def add_category_to_images(project: Project, category: str, image_ids: list[str]) -> None:
    """Bulk-assign: adds `category` to every image in image_ids that doesn't already have it."""
    catalog = load_catalog(project)
    for image_id in image_ids:
        entry = catalog["labels"].setdefault(
            image_id,
            {"label": "unlabeled", "split": "unassigned", "labeled_at": None, "notes": "", "categories": []},
        )
        cats = entry.setdefault("categories", [])
        if category not in cats:
            cats.append(category)
    save_catalog(project, catalog)


def image_matches_categories(project: Project, image_id: str, categories: list[str]) -> bool:
    """True if `categories` is empty (no scope = matches everything) or the image has at
    least one of the given categories (OR match, not AND — requiring all of a config's
    categories wouldn't make sense for e.g. a shared fork+spoon config)."""
    if not categories:
        return True
    image_categories = set(get_label(project, image_id).get("categories", []))
    return bool(image_categories & set(categories))
