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
    import streamlit as st

    sample = (context or {}).get("sample_image")
    h, w = (sample.shape[:2] if sample is not None else (1080, 1920))
    x = st.slider("Crop X", 0, max(1, w - 1), min(params["x"], w - 1))
    y = st.slider("Crop Y", 0, max(1, h - 1), min(params["y"], h - 1))
    cw = st.slider("Crop Width", 1, max(1, w - x), min(params["w"], w - x))
    ch = st.slider("Crop Height", 1, max(1, h - y), min(params["h"], h - y))
    return {"x": x, "y": y, "w": cw, "h": ch}
