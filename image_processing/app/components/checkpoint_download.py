"""Name a run and download its trained model checkpoint as a .ckpt file.

Shared by Train / Validate and Sweep. The name is saved as the run's name
(repository.set_run_name) as well as used for the file name, so it's remembered next time
and shown wherever runs are listed.
"""
import re

import streamlit as st

from image_processing.core import repository as repo


def _file_name(name: str, run_id: str) -> str:
    # Keep names filesystem-safe on every OS the browser might save to.
    stem = re.sub(r'[\\/:*?"<>|\s]+', "_", name.strip()).strip("._") or run_id
    return stem if stem.lower().endswith(".ckpt") else f"{stem}.ckpt"


def render(project, run_id: str, key_prefix: str) -> None:
    ckpt_path = repo.find_checkpoint(project, run_id)
    if ckpt_path is None:
        st.caption("No model checkpoint for this run (it may have failed or still be in progress).")
        return

    saved_name = repo.load_run_names(project).get(run_id, "")
    name_key = f"{key_prefix}_run_name_{run_id}"
    name_col, download_col = st.columns([3, 1])
    with name_col:
        name = st.text_input(
            "Checkpoint name",
            value=saved_name,
            key=name_key,
            placeholder=run_id,
            on_change=lambda: repo.set_run_name(project, run_id, st.session_state[name_key]),
            help="Press Enter to apply. Also saved as this run's name.",
        )
    file_name = _file_name(name, run_id)
    with download_col:
        st.write("")  # aligns the button with the input box, not its label
        st.download_button(
            "Download .ckpt",
            # A callable, not the bytes: checkpoints run to hundreds of MB, and bytes passed
            # directly would be read and re-sent on every rerun of the page, not just on click.
            data=lambda: ckpt_path.read_bytes(),
            file_name=file_name,
            mime="application/octet-stream",
            icon=":material/download:",
            on_click="ignore",
            width="stretch",
            key=f"{key_prefix}_download_{run_id}",
        )
    size_mb = ckpt_path.stat().st_size / 1e6
    st.caption(f"Downloads as `{file_name}` ({size_mb:,.0f} MB).")
