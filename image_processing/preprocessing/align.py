"""Align tool: multi-ROI landmark matching.

Each landmark stores its own reference template (as a base64 PNG) plus the anchor
position it was cut from in the reference image. At apply-time, each landmark's
template is located in the incoming image via normalized cross-correlation template
matching; one landmark yields a pure translation, two or more yield a rigid
(rotation + translation, no scale) transform fit by least squares.
"""
import base64

import cv2
import numpy as np


def default_params() -> dict:
    return {"landmarks": [], "search_margin": 40, "confidence_threshold": 0.5}


def _encode_template(template: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(template, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Failed to encode landmark template")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_template(template_b64: str) -> np.ndarray:
    buf = base64.b64decode(template_b64)
    arr = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def make_landmark(reference_image: np.ndarray, x: int, y: int, w: int, h: int) -> dict:
    """Cut a landmark template out of the reference image at (x, y, w, h)."""
    template = reference_image[y : y + h, x : x + w]
    return {"template_b64": _encode_template(template), "x": x, "y": y, "w": w, "h": h}


def match_landmarks(image: np.ndarray, params: dict) -> list[dict]:
    """Return per-landmark match results: matched top-left point + confidence."""
    margin = params.get("search_margin", 40)
    ih, iw = image.shape[:2]
    results = []
    for lm in params.get("landmarks", []):
        template = _decode_template(lm["template_b64"])
        th, tw = template.shape[:2]
        sx0 = max(0, lm["x"] - margin)
        sy0 = max(0, lm["y"] - margin)
        sx1 = min(iw, lm["x"] + tw + margin)
        sy1 = min(ih, lm["y"] + th + margin)
        search_region = image[sy0:sy1, sx0:sx1]
        if search_region.shape[0] < th or search_region.shape[1] < tw:
            results.append({"matched_x": lm["x"], "matched_y": lm["y"], "confidence": 0.0})
            continue
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        matched_x = sx0 + max_loc[0]
        matched_y = sy0 + max_loc[1]
        results.append({"matched_x": matched_x, "matched_y": matched_y, "confidence": float(max_val)})
    return results


def apply(image: np.ndarray, params: dict) -> np.ndarray:
    landmarks = params.get("landmarks", [])
    if not landmarks:
        return image
    matches = match_landmarks(image, params)
    h, w = image.shape[:2]

    if len(landmarks) == 1:
        lm, match = landmarks[0], matches[0]
        dx = match["matched_x"] - lm["x"]
        dy = match["matched_y"] - lm["y"]
        m = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
        return cv2.warpAffine(image, m, (w, h))

    ref_pts = np.array([[lm["x"], lm["y"]] for lm in landmarks], dtype=np.float32)
    matched_pts = np.array([[m["matched_x"], m["matched_y"]] for m in matches], dtype=np.float32)
    transform, _ = cv2.estimateAffinePartial2D(matched_pts, ref_pts, method=cv2.LMEDS)
    if transform is None:
        return image
    return cv2.warpAffine(image, transform, (w, h))


def low_confidence_landmarks(image: np.ndarray, params: dict) -> list[int]:
    """Indices of landmarks whose match confidence fell below the configured threshold."""
    threshold = params.get("confidence_threshold", 0.5)
    matches = match_landmarks(image, params)
    return [i for i, m in enumerate(matches) if m["confidence"] < threshold]


def render_controls(params: dict, context: dict | None = None) -> dict:
    import streamlit as st

    from image_processing.preprocessing.ui_widgets import slider_with_input

    step_id = (context or {}).get("step_id", "")
    st.caption(
        "Pick a reference image and draw one or more landmark boxes on it via "
        "'Manage landmarks' below. One landmark corrects translation only; "
        "two or more also correct rotation."
    )
    # Step-scoped key prefixes — see crop.py's render_controls for why these are needed.
    threshold = slider_with_input(
        "Match confidence threshold",
        0.0,
        1.0,
        float(params.get("confidence_threshold", 0.5)),
        0.05,
        f"align_threshold_{step_id}",
        format="%.2f",
    )
    margin = slider_with_input(
        "Search margin (px)", 5, 200, int(params.get("search_margin", 40)), 1, f"align_margin_{step_id}"
    )
    st.write(f"Landmarks defined: {len(params.get('landmarks', []))}")
    return {**params, "confidence_threshold": threshold, "search_margin": margin}
