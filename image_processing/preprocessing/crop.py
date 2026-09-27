"""Crop tool: fixed pixel ROI box."""
import numpy as np


def default_params() -> dict:
    return {"x": 0, "y": 0, "w": 256, "h": 256}


def apply(image: np.ndarray, params: dict) -> np.ndarray:
    h, w = image.shape[:2]
    x = max(0, min(params["x"], w - 1))
    y = max(0, min(params["y"], h - 1))
    cw = max(1, min(params["w"], w - x))
    ch = max(1, min(params["h"], h - y))
    return image[y : y + ch, x : x + cw]


def render_controls(params: dict, context: dict | None = None) -> dict:
    from image_processing.preprocessing.ui_widgets import slider_with_input

    context = context or {}
    sample = context.get("sample_image")
    step_id = context.get("step_id", "")
    h, w = (sample.shape[:2] if sample is not None else (1080, 1920))
    # slider_with_input pairs each slider with a linked number box so exact values can be
    # typed in, and (like the plain st.slider it replaces) needs a step-scoped key prefix —
    # without one, Streamlit derives a widget's identity from all its arguments including
    # its value, which changes on every edit, so a drag/entry would only "stick" every
    # other time. See pipeline_builder.py's call site for step_id.
    x = slider_with_input("Crop X", 0, max(1, w - 1), min(params["x"], w - 1), 1, f"crop_x_{step_id}")
    y = slider_with_input("Crop Y", 0, max(1, h - 1), min(params["y"], h - 1), 1, f"crop_y_{step_id}")
    cw = slider_with_input("Crop Width", 1, max(1, w - x), min(params["w"], w - x), 1, f"crop_w_{step_id}")
    ch = slider_with_input("Crop Height", 1, max(1, h - y), min(params["h"], h - y), 1, f"crop_h_{step_id}")
    return {"x": x, "y": y, "w": cw, "h": ch}
