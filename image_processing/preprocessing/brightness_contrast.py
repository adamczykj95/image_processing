"""Brightness/contrast tool: linear gain/offset (output = alpha*input + beta)."""
import cv2
import numpy as np


def default_params() -> dict:
    return {"alpha": 1.0, "beta": 0.0}


def apply(image: np.ndarray, params: dict) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=params["alpha"], beta=params["beta"])


def render_controls(params: dict, context: dict | None = None) -> dict:
    import streamlit as st

    alpha = st.slider("Contrast (gain)", 0.1, 3.0, float(params["alpha"]), step=0.05)
    beta = st.slider("Brightness (offset)", -100.0, 100.0, float(params["beta"]), step=1.0)
    return {"alpha": alpha, "beta": beta}
