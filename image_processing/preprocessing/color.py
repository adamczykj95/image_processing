"""Color tool: RGB<->gray. Gray output is replicated to 3 channels for backbone compatibility."""
import cv2
import numpy as np


def default_params() -> dict:
    return {"mode": "rgb"}


def apply(image: np.ndarray, params: dict) -> np.ndarray:
    if params.get("mode", "rgb") == "gray":
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return image


def render_controls(params: dict, context: dict | None = None) -> dict:
    import streamlit as st

    mode = st.radio("Color mode", ["rgb", "gray"], index=0 if params.get("mode", "rgb") == "rgb" else 1)
    return {"mode": mode}
