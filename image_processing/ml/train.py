"""Trains a PatchCore model for one run: fit on the training set, predict over validation,
persist predictions + per-image anomaly maps + cross-check metrics."""
import csv
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from anomalib.engine import Engine
from anomalib.models import Patchcore

from image_processing.core.config_schema import RunConfig
from image_processing.core.project import Project
from image_processing.core.store import write_json
from image_processing.ml.datamodule import build_datamodule, materialize_run_data
from image_processing.ml.metrics import compute_run_range


def _set_status(run_dir: Path, state: str, message: str = "") -> None:
    write_json(run_dir / "status.json", {"state": state, "message": message})


def run_training(project: Project, run_id: str, run_config: RunConfig) -> dict:
    run_dir = project.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", run_config.model_dump())

    try:
        _set_status(run_dir, "running", "materializing data")
        cache_dir = project.preprocessing_cache_dir / run_config.preproc_hash
        data_root = materialize_run_data(
            run_dir,
            cache_dir,
            run_config.train_image_ids,
            run_config.val_good_image_ids,
            run_config.val_anomaly_image_ids,
        )
        datamodule = build_datamodule(data_root)

        _set_status(run_dir, "running", "building model")
        pre_processor = Patchcore.configure_pre_processor(
            image_size=run_config.image_size, center_crop_size=None
        )
        model = Patchcore(
            backbone=run_config.model.backbone,
            layers=run_config.model.layers,
            coreset_sampling_ratio=run_config.model.coreset_sampling_ratio,
            num_neighbors=run_config.model.num_neighbors,
            pre_processor=pre_processor,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        engine = Engine(
            default_root_dir=str(run_dir),
            max_epochs=1,
            accelerator=device,
            devices=1,
            enable_progress_bar=False,
        )

        _set_status(run_dir, "running", "training (building coreset memory bank)")
        engine.fit(model=model, datamodule=datamodule)

        ckpt_candidates = list(run_dir.rglob("*.ckpt"))
        ckpt_path = str(ckpt_candidates[0]) if ckpt_candidates else None

        _set_status(run_dir, "running", "cross-check metrics (engine.test)")
        test_results = engine.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

        _set_status(run_dir, "running", "predicting over validation set")
        predictions = engine.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

        heatmaps_dir = run_dir / "heatmaps"
        heatmaps_dir.mkdir(exist_ok=True)
        rows = []
        for batch in predictions:
            n = len(batch.image_path)
            for i in range(n):
                image_id = Path(batch.image_path[i]).stem
                gt_label = int(batch.gt_label[i])
                pred_score = float(batch.pred_score[i])
                pred_label = int(batch.pred_label[i])
                rows.append(
                    {
                        "image_id": image_id,
                        "path": batch.image_path[i],
                        "gt_label": gt_label,
                        "pred_score": pred_score,
                        "pred_label": pred_label,
                    }
                )
                if batch.anomaly_map is not None:
                    amap = batch.anomaly_map[i]
                    if isinstance(amap, torch.Tensor):
                        amap = amap.detach().cpu().numpy()
                    np.save(heatmaps_dir / f"{image_id}.npy", amap.astype(np.float32))

        predictions_path = run_dir / "predictions.csv"
        with open(predictions_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "path", "gt_label", "pred_score", "pred_label"])
            writer.writeheader()
            writer.writerows(rows)

        heatmap_vmin, heatmap_vmax = compute_run_range(run_dir)

        write_json(
            run_dir / "metrics.json",
            {
                "anomalib_test_results": test_results,
                "ckpt_path": ckpt_path,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "n_predictions": len(rows),
                "heatmap_range": {"vmin": heatmap_vmin, "vmax": heatmap_vmax},
            },
        )

        _set_status(run_dir, "done", f"completed with {len(rows)} validation predictions")
        return {"run_id": run_id, "predictions_path": str(predictions_path), "n_predictions": len(rows)}

    except Exception as exc:  # noqa: BLE001 - surface any failure into status.json for the GUI
        _set_status(run_dir, "failed", f"{exc}\n{traceback.format_exc()}")
        raise
