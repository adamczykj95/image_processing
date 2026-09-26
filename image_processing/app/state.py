"""Shared helpers for locating/opening the currently-active project across Streamlit pages.

st.session_state persists across page navigation within one browser session but not
across a server restart or a different tab, so the *last opened* project path is also
mirrored to a small file under the user's home directory as a durable fallback.
"""
from pathlib import Path

import streamlit as st

from image_processing.core.project import Project

LAST_PROJECT_FILE = Path.home() / ".image_processing" / "last_project.txt"


def get_last_project_path() -> Path | None:
    if LAST_PROJECT_FILE.exists():
        text = LAST_PROJECT_FILE.read_text().strip()
        if text:
            return Path(text)
    return None


def set_last_project_path(path: Path) -> None:
    LAST_PROJECT_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_PROJECT_FILE.write_text(str(path))


def get_current_project() -> Project | None:
    if "project_root" not in st.session_state:
        last = get_last_project_path()
        if last and Project.exists(last):
            st.session_state["project_root"] = str(last)
        else:
            return None
    root = Path(st.session_state["project_root"])
    if not Project.exists(root):
        return None
    return Project.open(root)


def set_current_project(project: Project) -> None:
    st.session_state["project_root"] = str(project.root)
    set_last_project_path(project.root)


def require_project() -> Project:
    project = get_current_project()
    if project is None:
        st.warning("No project open. Go to the Home page to create or open one.")
        st.stop()
    return project
