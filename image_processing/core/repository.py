"""Data-access functions over a project's catalog.json (images + labels) and its runs/sweeps directories."""
from pathlib import Path
from datetime import datetime, timezone

from image_processing.core.project import Project
from image_processing.core.store import read_json, write_json

LabelValue = str  # "good" | "anomaly" | "unlabeled"
SplitValue = str  # "train" | "val" | "unassigned"


def load_catalog(project: Project) -> dict:
    return read_json(project.catalog_path, default={"images": [], "labels": {}})


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
        image_id, {"label": "unlabeled", "split": "unassigned", "labeled_at": None, "notes": ""}
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
        image_id, {"label": "unlabeled", "split": "unassigned", "labeled_at": None, "notes": ""}
    )
    if label is not None:
        entry["label"] = label
        entry["labeled_at"] = datetime.now(timezone.utc).isoformat()
    if split is not None:
        entry["split"] = split
    if notes is not None:
        entry["notes"] = notes
    save_catalog(project, catalog)


def get_label(project: Project, image_id: str) -> dict:
    return load_catalog(project)["labels"].get(
        image_id, {"label": "unlabeled", "split": "unassigned", "labeled_at": None, "notes": ""}
    )


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
