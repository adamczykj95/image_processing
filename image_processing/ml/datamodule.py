"""Builds an Anomalib Folder datamodule for one run, from the preprocessing cache + a train/val split.

Anomalib's Folder datamodule needs three real directories (normal_dir / normal_test_dir /
abnormal_dir); the preprocessing cache stores processed images flatly under good/anomaly.
This module materializes a run-specific directory of symlinks into those three shapes.

Split-mode note (verified empirically against anomalib 2.6.2): Folder's default
val_split_mode is FROM_TEST, which siphons half of the test set into an internal
validation split we don't need and would silently shrink our validation set. Passing
val_split_mode=SAME_AS_TEST instead makes val_data a full copy of test_data, so both
predict()/test() see the complete val/good + val/anomaly set with nothing held out.
"""
import os
from pathlib import Path

from anomalib.data import Folder
from anomalib.data.utils.split import TestSplitMode, ValSplitMode


def _symlink_into(cache_dir: Path, label: str, image_ids: list[str], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for image_id in image_ids:
        src = cache_dir / label / f"{image_id}.png"
        if not src.exists():
            missing.append(image_id)
            continue
        dst = dest_dir / f"{image_id}.png"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(src.resolve(), dst)  # absolute target: a relative one would resolve
        # relative to dst's own directory, not the original cwd, and silently break.
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} '{label}' image(s) not found in preprocessing cache "
            f"'{cache_dir.name}': {missing}. This usually means images were imported or "
            "(re)labeled after the cache was last built. Go to the Preprocessing page and "
            "click 'Apply to project' again to (re)process them before training."
        )


def materialize_run_data(
    run_dir: Path,
    cache_dir: Path,
    train_image_ids: list[str],
    val_good_image_ids: list[str],
    val_anomaly_image_ids: list[str],
) -> Path:
    data_root = run_dir / "data"
    _symlink_into(cache_dir, "good", train_image_ids, data_root / "train" / "good")
    _symlink_into(cache_dir, "good", val_good_image_ids, data_root / "val" / "good")
    _symlink_into(cache_dir, "anomaly", val_anomaly_image_ids, data_root / "val" / "anomaly")
    return data_root


def build_datamodule(
    data_root: Path,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    num_workers: int = 4,
) -> Folder:
    return Folder(
        name="project",
        root=data_root,
        normal_dir="train/good",
        normal_test_dir="val/good",
        abnormal_dir="val/anomaly",
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )
