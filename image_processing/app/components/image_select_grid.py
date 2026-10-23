"""Thumbnail grid with click and Shift+click range multi-select (a custom Streamlit component).

Streamlit's own checkboxes can't see modifier keys, so range selection needs a frontend
of its own — see image_select_grid_frontend/index.html. Python stays the source of truth
for the selection (a set in session_state), so plain Streamlit buttons elsewhere on the
page (select all / none / invert, delete) can change it with set_selection() just like any
other state; the grid sends back the full new selection after each click.

Call get_selection() near the top of the script, before anything that displays the
selection count: the grid's latest click is already in session_state at that point (its
component value), even though the grid itself renders further down the page.
"""
import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_component = components.declare_component(
    "image_select_grid", path=str(Path(__file__).parent / "image_select_grid_frontend")
)


def _selected_key(key: str) -> str:
    return f"{key}__selected"


def _event_key(key: str) -> str:
    return f"{key}__applied_event"


def get_selection(key: str) -> set[str]:
    """The current selection, after applying any click the grid sent since the last run."""
    selection = st.session_state.setdefault(_selected_key(key), set())
    event = st.session_state.get(key)
    if event and event.get("event") != st.session_state.get(_event_key(key)):
        selection = set(event["selected"])
        st.session_state[_selected_key(key)] = selection
        st.session_state[_event_key(key)] = event["event"]
    return selection


def set_selection(key: str, image_ids) -> None:
    st.session_state[_selected_key(key)] = set(image_ids)


@lru_cache(maxsize=4096)
def _data_uri(path: str, mtime_ns: int) -> str:
    # mtime_ns is only part of the cache key, so a rewritten thumbnail isn't served stale.
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def render(key: str, items: list[tuple[str, Path, str]], columns: int = 6) -> None:
    """Renders the grid. `items` is [(image_id, thumbnail_path, caption)], in display order —
    Shift+click ranges follow this order."""
    _component(
        key=key,
        default=None,
        columns=columns,
        items=[
            {"id": image_id, "src": _data_uri(str(path), path.stat().st_mtime_ns), "caption": caption}
            for image_id, path, caption in items
        ],
        selected=sorted(get_selection(key)),
        applied_event=st.session_state.get(_event_key(key)),
    )
