"""Resize tool: target resolution + interpolation."""
import cv2
import numpy as np

_INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
}


def default_params() -> dict:
    return {"width": 256, "height": 256, "interpolation": "area"}


def apply(image: np.ndarray, params: dict) -> np.ndarray:
    interp = _INTERPOLATIONS.get(params.get("interpolation", "area"), cv2.INTER_AREA)
    return cv2.resize(image, (params["width"], params["height"]), interpolation=interp)


def render_controls(params: dict, context: dict | None = None) -> dict:
    import streamlit as st

    from image_processing.preprocessing.ui_widgets import slider_with_input

    step_id = (context or {}).get("step_id", "")
    # Step-scoped key prefixes — see crop.py's render_controls for why these are needed.
    width = slider_with_input("Resize Width", 32, 1024, params["width"], 16, f"resize_w_{step_id}")
    height = slider_with_input("Resize Height", 32, 1024, params["height"], 16, f"resize_h_{step_id}")
    interpolation = st.selectbox(
        "Interpolation",
        list(_INTERPOLATIONS.keys()),
        index=list(_INTERPOLATIONS.keys()).index(params.get("interpolation", "area")),
        key=f"resize_interp_{step_id}",
    )
    return {"width": width, "height": height, "interpolation": interpolation}
