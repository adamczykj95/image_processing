"""Project: the on-disk unit of persistence for one defect-detection workspace."""
import uuid
from datetime import datetime, timezone
from pathlib import Path

from image_processing.core.store import read_json, write_json

SCHEMA_VERSION = 1


class Project:
    def __init__(self, root: Path):
        # Always store an absolute path: relative roots would make every derived path
        # (including symlink targets materialized for training) relative too, and a
        # relative symlink resolves relative to the symlink's own directory, not the
        # original cwd — silently producing broken links deep under runs/<id>/data/.
        self.root = Path(root).expanduser().resolve()

    # -- paths -----------------------------------------------------------
    @property
    def project_json_path(self) -> Path:
        return self.root / "project.json"

    @property
    def catalog_path(self) -> Path:
        return self.root / "catalog.json"

    @property
    def images_raw_dir(self) -> Path:
        return self.root / "images" / "raw"

    @property
    def preprocessing_configs_dir(self) -> Path:
        return self.root / "preprocessing" / "configs"

    @property
    def preprocessing_cache_dir(self) -> Path:
        return self.root / "preprocessing" / "cache"

    @property
    def draft_pipeline_path(self) -> Path:
        """The in-progress (not-yet-applied) preprocessing tool chain, persisted so it
        survives a browser refresh instead of living only in Streamlit session_state."""
        return self.root / "preprocessing" / "draft_pipeline.json"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def sweeps_dir(self) -> Path:
        return self.root / "sweeps"

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def create(cls, root: Path, name: str) -> "Project":
        root = Path(root)
        project = cls(root)
        for d in (
            project.images_raw_dir,
            project.preprocessing_configs_dir,
            project.preprocessing_cache_dir,
            project.runs_dir,
            project.sweeps_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        write_json(
            project.project_json_path,
            {
                "name": name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": SCHEMA_VERSION,
                "reference_image_id": None,
            },
        )
        write_json(project.catalog_path, {"images": [], "labels": {}})
        return project

    @classmethod
    def open(cls, root: Path) -> "Project":
        project = cls(Path(root))
        if not project.project_json_path.exists():
            raise FileNotFoundError(f"No project.json found at {root}")
        return project

    @classmethod
    def exists(cls, root: Path) -> bool:
        return (Path(root) / "project.json").exists()

    # -- metadata ------------------------------------------------------------
    def meta(self) -> dict:
        return read_json(self.project_json_path, default={})

    def set_reference_image(self, image_id: str) -> None:
        meta = self.meta()
        meta["reference_image_id"] = image_id
        write_json(self.project_json_path, meta)

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]
