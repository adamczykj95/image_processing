import pandas as pd
import streamlit as st

from image_processing.app.components.sweep_builder import render_grid_form
from image_processing.app.state import require_project
from image_processing.app.style import inject_global_css
from image_processing.core import repository as repo
from image_processing.jobs.sweep_job import run_sweep

st.set_page_config(page_title="Sweep", page_icon="🧪", layout="wide")
inject_global_css()
project = require_project()
st.title("Hyperparameter Sweep")
st.caption(
    "Define a grid of preprocessing configs and PatchCore hyperparameters, then train "
    "and validate every combination. Each result is an ordinary run you can inspect in "
    "full detail (metrics, heatmap) on the Train / Validate page."
)

grid = render_grid_form(project)

if st.button("Start Sweep", type="primary", disabled=grid is None):
    sweep_id = project.new_id()
    with st.spinner(f"Running sweep ({grid['total']} combination(s))... this can take a while."):
        run_sweep(project, sweep_id, grid)
    st.session_state["last_sweep_id"] = sweep_id
    st.success(f"Sweep complete: `{sweep_id}`")
    st.rerun()

st.divider()
st.subheader("Sweep results")

sweeps = repo.list_sweeps(project)
if not sweeps:
    st.info("No sweeps yet.")
    st.stop()

sweep_ids = [s["id"] for s in sweeps]
default_sweep = st.session_state.get("last_sweep_id", sweep_ids[-1])
selected_sweep_id = st.selectbox(
    "Sweep",
    sweep_ids,
    index=sweep_ids.index(default_sweep) if default_sweep in sweep_ids else len(sweep_ids) - 1,
)
sweep = next(s for s in sweeps if s["id"] == selected_sweep_id)
sweep_dir = project.sweeps_dir / selected_sweep_id

status = sweep["status"]
st.write(f"Status: **{status.get('state', 'unknown')}** — {status.get('message', '')}")
if status.get("total"):
    st.progress(min(1.0, status.get("current_index", 0) / status["total"]))

results_path = sweep_dir / "results.csv"
if results_path.exists():
    df = pd.read_csv(results_path)
    config_names = repo.load_config_names(project)
    df.insert(0, "config", df["preproc_hash"].map(lambda h: config_names.get(h, h)))

    # st.dataframe can't embed interactive buttons in a cell, so the table is built row by
    # row from st.columns instead — this is what lets "View" live inside the table itself
    # rather than needing a separate list of buttons below it.
    col_widths = [0.6, 1.3, 1.1, 1.3, 1.1, 0.9, 0.9, 0.7, 0.7, 0.7, 0.7, 0.6, 1.0]
    headers = [
        "",
        "Config",
        "Run ID",
        "Backbone",
        "Layers",
        "Coreset",
        "Neighbors",
        "Patch",
        "AUROC",
        "F1",
        "Recall",
        "N",
        "Status",
    ]
    header_cols = st.columns(col_widths)
    for col, header in zip(header_cols, headers):
        col.markdown(f"**{header}**")

    for _, row in df.iterrows():
        cols = st.columns(col_widths)
        if cols[0].button("View", key=f"view_{row['run_id']}"):
            st.session_state["last_run_id"] = row["run_id"]
            st.switch_page("pages/4_Train_Validate.py")
        cols[1].write(row["config"])
        cols[2].write(f"`{row['run_id']}`")
        cols[3].write(row["backbone"])
        cols[4].write(row["layers"])
        cols[5].write(row["coreset_sampling_ratio"])
        cols[6].write(row["num_neighbors"])
        cols[7].write(row["patch_size"])
        cols[8].write(row["auroc"])
        cols[9].write(row["f1_anomaly"])
        cols[10].write(row["recall_anomaly"])
        cols[11].write(row["n_predictions"])
        cols[12].write(row["status"])
else:
    st.info("No results yet for this sweep.")
