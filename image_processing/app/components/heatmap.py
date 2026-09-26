"""Anomaly-map colormap overlay, shared by the Train/Validate and Inference pages.

Normalization is against a fixed (vmin, vmax) range shared across an entire run, not
computed per image. Anomalib's raw anomaly_map values are already comparable across
images (a "good" image's peak really is much lower than an anomalous image's) — the
per-image min/max normalization tried initially stretched every image's own local range
to fill the whole color scale, making even a boring, low-scoring good image look just as
"hot" as a genuine anomaly. See ml/train.py::run_training for where the range is computed.
"""
import cv2
import matplotlib
import numpy as np


def intensity_to_gamma(intensity: float) -> float:
    """Maps a 0-1 slider (default 0.5 = neutral) to a gamma exponent on a symmetric log
    scale: intensity=0.5 -> gamma=1 (unchanged), ->0 -> gamma~10 (only near-max scores
    still read as hot), ->1 -> gamma~0.1 (even modest scores expand toward hot)."""
    return 10.0 ** ((0.5 - intensity) * 2)


def overlay_heatmap(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    vmin: float,
    vmax: float,
    alpha: float = 0.5,
    intensity: float = 0.5,
    colormap: str = "inferno",
) -> np.ndarray:
    amap = anomaly_map.astype(np.float32)
    norm = np.clip((amap - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)

    # Gamma curve reshapes the transition steepness between cold and hot *within* the
    # vmin/vmax calibration above — it never moves the anchor points (0 stays 0, 1 stays
    # 1), so it can't undo that run-wide calibration, only how aggressively color ramps
    # up between the two ends.
    gamma = intensity_to_gamma(intensity)
    norm = norm**gamma

    cmap = matplotlib.colormaps[colormap]
    colored = (cmap(norm)[..., :3] * 255).astype(np.uint8)

    if colored.shape[:2] != image.shape[:2]:
        colored = cv2.resize(colored, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    blended = image.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
