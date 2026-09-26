"""Anomaly-map colormap overlay, shared by the Train/Validate and Inference pages."""
import cv2
import matplotlib
import numpy as np


def overlay_heatmap(
    image: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.5, colormap: str = "inferno"
) -> np.ndarray:
    amap = anomaly_map.astype(np.float32)
    amin, amax = float(amap.min()), float(amap.max())
    norm = (amap - amin) / (amax - amin + 1e-8)

    cmap = matplotlib.colormaps[colormap]
    colored = (cmap(norm)[..., :3] * 255).astype(np.uint8)

    if colored.shape[:2] != image.shape[:2]:
        colored = cv2.resize(colored, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    blended = image.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
