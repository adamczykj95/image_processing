"""Resize tool: target resolution + interpolation."""
import cv2
import numpy as np

_INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
}


MIN_SIZE, MAX_SIZE = 32, 1024


def _fit_aspect(driver: int, ratio: float) -> tuple[int, int]:
    """(driver, follower) with follower = driver / ratio, both kept within the slider
    bounds. If the follower would leave them, it's clamped and the driver is recomputed
    from it, so the ratio holds as closely as integer pixels allow."""
    follower = round(driver / ratio)
    if not MIN_SIZE <= follower <= MAX_SIZE:
        follower = min(max(follower, MIN_SIZE), MAX_SIZE)
        driver = min(max(round(follower * ratio), MIN_SIZE), MAX_SIZE)
    return driver, follower


def default_params() -> dict:
    return {"width": 256, "height": 256, "interpolation": "area"}


def apply(image: np.ndarray, params: dict) -> np.ndarray:
    interp = _INTERPOLATIONS.get(params.get("interpolation", "area"), cv2.INTER_AREA)
    return cv2.resize(image, (params["width"], params["height"]), interpolation=interp)


def render_controls(params: dict, context: dict | None = None) -> dict:
    import streamlit as st

    from image_processing.preprocessing.ui_widgets import slider_with_input

    context = context or {}
    step_id = context.get("step_id", "")
    # Step-scoped key prefixes — see crop.py's render_controls for why these are needed.
    w_prefix, h_prefix = f"resize_w_{step_id}", f"resize_h_{step_id}"
    # Which slider the user moved last; the other one follows it while the lock is on.
    driver_key = f"resize_driver_{step_id}"

    sample_image = context.get("sample_image")
    lock = False
    if sample_image is not None:
        in_h, in_w = sample_image.shape[:2]
        # Session-only (not a saved param), so toggling it never changes the config hash.
        lock = st.checkbox(
            f"Keep input aspect ratio ({in_w}×{in_h})",
            value=True,
            key=f"resize_lock_{step_id}",
            help="Ratio comes from the image entering this step (the current preview sample "
            "after any earlier steps). Uncheck to set width and height freely.",
        )

    def _set(prefix: str, value: int) -> None:
        st.session_state[f"{prefix}_slider"] = value
        st.session_state[f"{prefix}_number"] = value

    def _on_change(driver: str) -> None:
        st.session_state[driver_key] = driver

    if lock:
        # Enforced on every render (not just inside the slider callbacks), so re-checking
        # the box or paging to a differently-shaped sample image also snaps the follower
        # back to the input ratio. Writing the keys here is safe: the sliders below haven't
        # been instantiated yet in this script pass.
        width = st.session_state.get(f"{w_prefix}_slider", params["width"])
        height = st.session_state.get(f"{h_prefix}_slider", params["height"])
        if st.session_state.get(driver_key, "width") == "width":
            width, height = _fit_aspect(width, in_w / in_h)
        else:
            height, width = _fit_aspect(height, in_h / in_w)
        _set(w_prefix, width)
        _set(h_prefix, height)

    width = slider_with_input(
        "Resize Width", MIN_SIZE, MAX_SIZE, params["width"], 16, w_prefix, on_change=lambda _: _on_change("width")
    )
    height = slider_with_input(
        "Resize Height", MIN_SIZE, MAX_SIZE, params["height"], 16, h_prefix, on_change=lambda _: _on_change("height")
    )
    interpolation = st.selectbox(
        "Interpolation",
        list(_INTERPOLATIONS.keys()),
        index=list(_INTERPOLATIONS.keys()).index(params.get("interpolation", "area")),
        key=f"resize_interp_{step_id}",
    )
    return {"width": width, "height": height, "interpolation": interpolation}
