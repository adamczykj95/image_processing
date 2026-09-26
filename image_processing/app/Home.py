from pathlib import Path

import streamlit as st

from image_processing.app.state import get_current_project, set_current_project
from image_processing.core.project import Project

st.set_page_config(page_title="Defect Detection", page_icon="🔍", layout="wide")
st.title("PatchCore Defect Detection")

project = get_current_project()
if project is not None:
    meta = project.meta()
    st.success(f"Open project: **{meta.get('name', project.root.name)}** ({project.root})")

st.divider()

tab_open, tab_create = st.tabs(["Open Existing Project", "Create New Project"])

with tab_open:
    path_str = st.text_input("Project directory path", value=str(project.root) if project else "")
    if st.button("Open Project", type="primary"):
        if not path_str.strip():
            st.error("Enter a path.")
        else:
            candidate = Path(path_str.strip())
            if not Project.exists(candidate):
                st.error(f"No project.json found at {candidate}")
            else:
                set_current_project(Project.open(candidate))
                st.success(f"Opened project at {candidate}")
                st.rerun()

with tab_create:
    new_name = st.text_input("Project name", key="new_name")
    new_root = st.text_input("New project directory (will be created)", key="new_root")
    if st.button("Create Project"):
        if not new_name.strip() or not new_root.strip():
            st.error("Both name and directory are required.")
        else:
            candidate = Path(new_root.strip())
            if Project.exists(candidate):
                st.error(f"A project already exists at {candidate}")
            else:
                new_project = Project.create(candidate, new_name.strip())
                set_current_project(new_project)
                st.success(f"Created project '{new_name}' at {candidate}")
                st.rerun()

if project is not None:
    st.divider()
    st.subheader("Next steps")
    st.markdown(
        "1. **Import Images** — add your normal/defect photos to the project.\n"
        "2. **Label Images** — mark each as good or anomaly, assign train/val split.\n"
        "3. **Preprocessing** — build your alignment/crop/resize/color tool chain.\n"
        "4. **Train / Validate** — train PatchCore and review metrics + heatmaps.\n"
        "5. **Sweep** — search hyperparameters across configs.\n"
        "6. **Inference** — run a trained model on new images."
    )
