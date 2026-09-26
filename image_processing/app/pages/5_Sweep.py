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
    st.dataframe(df, width="stretch")

    st.write("View a specific run in detail:")
    for _, row in df.iterrows():
        view_col, label_col = st.columns([1, 6])
        with view_col:
            if st.button("View", key=f"view_{row['run_id']}"):
                st.session_state["last_run_id"] = row["run_id"]
                st.switch_page("pages/4_Train_Validate.py")
        with label_col:
            st.write(
                f"`{row['run_id']}` · {row['backbone']} · {row['layers']} · "
                f"coreset={row['coreset_sampling_ratio']} · neighbors={row['num_neighbors']} · "
                f"patch={row['patch_size']} · AUROC={row['auroc']} · status={row['status']}"
            )
else:
    st.info("No results yet for this sweep.")
