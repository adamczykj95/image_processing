import json

import numpy as np
import pandas as pd
import streamlit as st

from image_processing.app.components.heatmap import overlay_heatmap
from image_processing.app.state import require_project
from image_processing.core import repository as repo
from image_processing.core.config_schema import ModelConfig, PreprocessConfig, RunConfig
from image_processing.ml import metrics as ml_metrics
from image_processing.ml.train import run_training
from image_processing.preprocessing.pipeline import load_image, missing_cached_images

st.set_page_config(page_title="Train / Validate", page_icon="🧠", layout="wide")
project = require_project()
st.title("Train / Validate")

config_files = sorted(project.preprocessing_configs_dir.glob("*.json"))
if not config_files:
    st.warning("No preprocessing config found yet — build and apply one on the Preprocessing page first.")
    st.stop()

hash_options = [f.stem for f in config_files]
default_hash = st.session_state.get("last_preproc_hash", hash_options[-1])
preproc_hash = st.selectbox(
    "Preprocessing config", hash_options, index=hash_options.index(default_hash) if default_hash in hash_options else 0
)
cache_dir = project.preprocessing_cache_dir / preproc_hash

train_ids = repo.list_by_label_and_split(project, "good", "train")
val_good_ids = repo.list_by_label_and_split(project, "good", "val")
val_anomaly_ids = repo.list_by_label_and_split(project, "anomaly", "val")

st.write(
    f"Train (good): **{len(train_ids)}** &nbsp;·&nbsp; "
    f"Val good: **{len(val_good_ids)}** &nbsp;·&nbsp; Val anomaly: **{len(val_anomaly_ids)}**"
)

sample_files = list((cache_dir / "good").glob("*.png")) or list((cache_dir / "anomaly").glob("*.png"))
image_size = load_image(sample_files[0]).shape[:2] if sample_files else (256, 256)
st.caption(f"Cached image size: {image_size[1]}x{image_size[0]} (width x height)")

st.subheader("Model hyperparameters")
c1, c2 = st.columns(2)
with c1:
    backbone = st.selectbox("Backbone", ["wide_resnet50_2", "resnet18", "wide_resnet101_2"], index=0)
    layers = st.multiselect("Feature layers", ["layer1", "layer2", "layer3", "layer4"], default=["layer2", "layer3"])
with c2:
    coreset_ratio = st.slider("Coreset sampling ratio", 0.01, 1.0, 0.1, step=0.01)
    num_neighbors = st.slider("Num neighbors", 1, 20, 9)

missing = missing_cached_images(
    project,
    preproc_hash,
    {"good": train_ids + val_good_ids, "anomaly": val_anomaly_ids},
)
can_train = bool(train_ids and val_good_ids and val_anomaly_ids and layers and not missing)
if not (train_ids and val_good_ids and val_anomaly_ids and layers):
    st.warning(
        "Need at least one train image, one val-good image, one val-anomaly image, "
        "and one selected feature layer before training."
    )
if missing:
    st.error(
        f"{len(missing)} labeled image(s) haven't been processed by this preprocessing "
        f"config yet: {missing}. Go to the Preprocessing page and click 'Apply to project' "
        "again — this happens when images are imported or labeled after the cache was built."
    )

if st.button("Start Training", type="primary", disabled=not can_train):
    run_id = project.new_id()
    preprocess_config = PreprocessConfig.model_validate(
        json.loads(config_files[hash_options.index(preproc_hash)].read_text())
    )
    run_config = RunConfig(
        preproc_hash=preproc_hash,
        preprocess=preprocess_config,
        model=ModelConfig(
            backbone=backbone, layers=layers, coreset_sampling_ratio=coreset_ratio, num_neighbors=num_neighbors
        ),
        train_image_ids=train_ids,
        val_good_image_ids=val_good_ids,
        val_anomaly_image_ids=val_anomaly_ids,
        image_size=(int(image_size[0]), int(image_size[1])),
    )
    with st.spinner("Training PatchCore (building coreset memory bank)..."):
        run_training(project, run_id, run_config)
    st.session_state["last_run_id"] = run_id
    st.success(f"Training complete: run `{run_id}`")
    st.rerun()

st.divider()
st.subheader("Run results")

runs = repo.list_runs(project)
if not runs:
    st.info("No runs yet.")
    st.stop()

run_ids = [r["id"] for r in runs]
default_run = st.session_state.get("last_run_id", run_ids[-1])
selected_run_id = st.selectbox("Run", run_ids, index=run_ids.index(default_run) if default_run in run_ids else len(run_ids) - 1)
run = next(r for r in runs if r["id"] == selected_run_id)
run_dir = project.runs_dir / selected_run_id

st.write(f"Status: **{run['status'].get('state')}** — {run['status'].get('message', '')}")

predictions_path = run_dir / "predictions.csv"
if run["status"].get("state") == "done" and predictions_path.exists():
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, step=0.01, key=f"thresh_{selected_run_id}")
    summary = ml_metrics.summarize(predictions_path, threshold=threshold)

    m1, m2, m3 = st.columns(3)
    m1.metric("AUROC", f"{summary['auroc']:.3f}" if summary["auroc"] == summary["auroc"] else "n/a")
    m2.metric("F1 (anomaly)", f"{summary['at_threshold']['f1_anomaly']:.3f}")
    m3.metric("Recall (anomaly)", f"{summary['at_threshold']['recall_anomaly']:.3f}")

    cm = summary["at_threshold"]["confusion_matrix"]
    st.write("Confusion matrix (rows = actual, cols = predicted)")
    st.dataframe(
        pd.DataFrame(cm, index=["Actual: good", "Actual: anomaly"], columns=["Pred: good", "Pred: anomaly"])
    )

    roc = summary["roc_curve"]
    if roc["fpr"]:
        st.write("ROC curve")
        st.line_chart(pd.DataFrame({"tpr": roc["tpr"]}, index=roc["fpr"]))

    st.divider()
    show_heatmap = st.toggle("Show anomaly heatmap overlay", value=True)
    heatmap_alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.5, step=0.05, disabled=not show_heatmap)

    df = ml_metrics.load_predictions(predictions_path)
    cache_for_run = project.preprocessing_cache_dir / run["config"]["preproc_hash"]
    cols = st.columns(4)
    for i, row in df.iterrows():
        label_dir = "anomaly" if row["gt_label"] == 1 else "good"
        image_path = cache_for_run / label_dir / f"{row['image_id']}.png"
        with cols[i % 4]:
            base_image = load_image(image_path)
            if show_heatmap:
                heatmap_path = run_dir / "heatmaps" / f"{row['image_id']}.npy"
                amap = np.load(heatmap_path)
                display_image = overlay_heatmap(base_image, amap, alpha=heatmap_alpha)
            else:
                display_image = base_image
            st.image(display_image, width="stretch")
            pred = "anomaly" if row["pred_score"] >= threshold else "good"
            st.caption(f"gt={label_dir} · score={row['pred_score']:.3f} · pred={pred}")
elif run["status"].get("state") == "failed":
    st.error(run["status"].get("message", "Run failed."))
else:
    st.info("Run still in progress or has no predictions yet.")
