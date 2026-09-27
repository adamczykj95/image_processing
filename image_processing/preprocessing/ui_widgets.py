"""Shared preprocessing-tool UI helper: a slider paired with a linked number input, so
users can either drag or type an exact value. The two widgets are kept in sync through
session_state + on_change callbacks (Streamlit's documented pattern for linked widgets) —
each widget is bound to session_state via its own `key` and takes no `value=` argument, so
there's a single source of truth instead of two independently-tracked values that could
drift apart.
"""
from typing import Any

import streamlit as st


def slider_with_input(
    label: str,
    min_value: Any,
    max_value: Any,
    value: Any,
    step: Any,
    key_prefix: str,
    format: str | None = None,
) -> Any:
    slider_key = f"{key_prefix}_slider"
    number_key = f"{key_prefix}_number"

    if slider_key not in st.session_state:
        st.session_state[slider_key] = value
    if number_key not in st.session_state:
        st.session_state[number_key] = value

    def _sync_to_number() -> None:
        st.session_state[number_key] = st.session_state[slider_key]

    def _sync_to_slider() -> None:
        st.session_state[slider_key] = st.session_state[number_key]

    slider_col, number_col = st.columns([3, 1])
    with slider_col:
        st.slider(label, min_value, max_value, step=step, key=slider_key, on_change=_sync_to_number)
    with number_col:
        st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=number_key,
            on_change=_sync_to_slider,
            label_visibility="collapsed",
            format=format,
        )
    return st.session_state[slider_key]
