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


def dynamic_bounded_slider(
    label: str,
    min_value: Any,
    max_value: Any,
    current_value: Any,
    step: Any,
    key_prefix: str,
    format: str | None = None,
) -> Any:
    """Like slider_with_input, but for a slider whose min/max bounds are recomputed from
    another widget's current value on every rerun — e.g. a position slider bounded by a
    size slider, so the box it controls can never be positioned off-image. Two things
    plain slider_with_input doesn't handle, both needed for that use case:

    1. Streamlit folds a slider's min/max into its identity even when it has an explicit
       key, so a bounds change makes Streamlit treat it as a brand-new widget and reset it
       to `min_value` — silently snapping the box to an edge — unless given an explicit,
       correct value to fall back to. `current_value` (which the caller must track itself,
       e.g. from a params dict or its own session_state entry — never read back from this
       widget's own slider/number keys, or the reset value would already be baked in) is
       passed as that fallback on every render. Confirmed empirically this does not fight
       a live drag/type on this same widget when its identity is *unchanged*: the
       frontend-reported value still wins over a freshly supplied value= in that case, so
       the explicit value only ever matters right when the identity actually changes.
    2. A slider's min and max must not be equal (Streamlit raises otherwise) — this
       happens whenever the box dimension that bounds this one exactly fills the image on
       that axis, at which point this axis's position has exactly one valid value anyway,
       so a slider would be both invalid and pointless. Falls back to a fixed caption.
    """
    if max_value <= min_value:
        st.caption(f"{label}: fixed at {min_value} (limited by image size)")
        return min_value

    slider_key = f"{key_prefix}_slider"
    number_key = f"{key_prefix}_number"
    current_value = min(max(current_value, min_value), max_value)

    def _sync_to_number() -> None:
        st.session_state[number_key] = st.session_state[slider_key]

    def _sync_to_slider() -> None:
        st.session_state[slider_key] = st.session_state[number_key]

    slider_col, number_col = st.columns([3, 1])
    with slider_col:
        st.slider(
            label, min_value, max_value, value=current_value, step=step, key=slider_key, on_change=_sync_to_number
        )
    with number_col:
        st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            value=current_value,
            step=step,
            key=number_key,
            on_change=_sync_to_slider,
            label_visibility="collapsed",
            format=format,
        )
    return st.session_state[slider_key]
