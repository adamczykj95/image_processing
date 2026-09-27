"""Brightness/contrast tool: linear gain/offset (output = alpha*input + beta)."""
import cv2
import numpy as np


def default_params() -> dict:
    return {"alpha": 1.0, "beta": 0.0}


def apply(image: np.ndarray, params: dict) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=params["alpha"], beta=params["beta"])


def render_controls(params: dict, context: dict | None = None) -> dict:
    from image_processing.preprocessing.ui_widgets import slider_with_input

    step_id = (context or {}).get("step_id", "")
    # Step-scoped key prefixes — see crop.py's render_controls for why these are needed.
    alpha = slider_with_input(
        "Contrast (gain)", 0.1, 3.0, float(params["alpha"]), 0.05, f"bc_alpha_{step_id}", format="%.2f"
    )
    beta = slider_with_input(
        "Brightness (offset)", -100.0, 100.0, float(params["beta"]), 1.0, f"bc_beta_{step_id}", format="%.1f"
    )
    return {"alpha": alpha, "beta": beta}
