"""Crop tool: a rectangular ROI positioned by its center, not its top-left corner.

Width/Height and the center X/Y position are deliberately independent: the size sliders
are bounded only by the full image, and the position sliders are then bounded to whatever
range keeps a box of that size fully on-image. This one-way dependency (size can nudge
position, position never touches size) mirrors the fix already applied to the align tool's
landmark manager, for the same reason — the reverse order let repositioning silently
shrink an already-chosen size via Streamlit's automatic value-clamping.
"""
import cv2
import numpy as np


def default_params() -> dict:
    return {"cx": 128, "cy": 128, "w": 256, "h": 256}


def _resolve_center(params: dict) -> tuple[int, int]:
    """A config saved before crop switched to center-based positioning stored a top-left
    corner (x, y) instead of a center — convert it once so an old config still crops the
    exact same region it always did, with no migration step."""
    if "cx" in params and "cy" in params:
        return params["cx"], params["cy"]
    x = params.get("x", 0)
    y = params.get("y", 0)
    w = params.get("w", 256)
    h = params.get("h", 256)
    return x + w // 2, y + h // 2


def center_bounds(image_dim: int, box_dim: int) -> tuple[int, int]:
    """The range of center coordinates that keep a box of box_dim fully inside image_dim.

    Public (not module-private) because the align tool's landmark picker reuses this same
    region-positioning logic — see pipeline_builder._render_align_landmark_manager.
    """
    box_dim = max(1, min(box_dim, image_dim))
    min_c = box_dim // 2
    max_c = max(min_c, image_dim - (box_dim - box_dim // 2))
    return min_c, max_c


def crop_box(image_shape: tuple[int, int], params: dict) -> tuple[int, int, int, int]:
    """Resolves params against an image of this shape into (left, top, w, h) — the single
    source of truth shared by apply() and the Preprocessing page's overlay, so the box
    drawn on the input preview always matches exactly what apply() crops."""
    img_h, img_w = image_shape
    w = max(1, min(params.get("w", 256), img_w))
    h = max(1, min(params.get("h", 256), img_h))
    cx, cy = _resolve_center(params)
    min_cx, max_cx = center_bounds(img_w, w)
    min_cy, max_cy = center_bounds(img_h, h)
    cx = min(max(cx, min_cx), max_cx)
    cy = min(max(cy, min_cy), max_cy)
    return cx - w // 2, cy - h // 2, w, h


def apply(image: np.ndarray, params: dict) -> np.ndarray:
    left, top, w, h = crop_box(image.shape[:2], params)
    return image[top : top + h, left : left + w]


def draw_overlay(image: np.ndarray, params: dict) -> np.ndarray:
    """A copy of image with the current crop region outlined — for previewing where the
    crop will land before it's applied. Never used for the actual cached/trained output."""
    left, top, w, h = crop_box(image.shape[:2], params)
    overlay = image.copy()
    cv2.rectangle(overlay, (left, top), (left + w - 1, top + h - 1), (255, 0, 0), 2)
    return overlay


def render_controls(params: dict, context: dict | None = None) -> dict:
    from image_processing.preprocessing.ui_widgets import dynamic_bounded_slider

    context = context or {}
    sample = context.get("sample_image")
    step_id = context.get("step_id", "")
    img_h, img_w = (sample.shape[:2] if sample is not None else (1080, 1920))

    # Size first, against the full image — see module docstring for why this ordering
    # (not position first) is required for well-behaved, non-interfering sliders.
    # dynamic_bounded_slider (not plain slider_with_input) is required here specifically
    # because these sliders' bounds are recomputed from each other every render — see its
    # docstring for why that needs special handling to avoid the position snapping back to
    # an image edge every time Width/Height changes.
    w = dynamic_bounded_slider("Crop region width", 1, img_w, params.get("w", 256), 1, f"crop_w_{step_id}")
    h = dynamic_bounded_slider("Crop region height", 1, img_h, params.get("h", 256), 1, f"crop_h_{step_id}")

    cx0, cy0 = _resolve_center(params)
    min_cx, max_cx = center_bounds(img_w, w)
    min_cy, max_cy = center_bounds(img_h, h)
    cx = dynamic_bounded_slider("Region X position (center)", min_cx, max_cx, cx0, 1, f"crop_cx_{step_id}")
    cy = dynamic_bounded_slider("Region Y position (center)", min_cy, max_cy, cy0, 1, f"crop_cy_{step_id}")

    return {"cx": cx, "cy": cy, "w": w, "h": h}
