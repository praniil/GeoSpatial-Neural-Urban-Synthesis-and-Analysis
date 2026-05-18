import os
from typing import Tuple

import numpy as np
from PIL import Image

COLOR_MAP = np.array([
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [0, 128, 128],
    [128, 128, 0],
    [128, 0, 128],
], dtype=np.uint8)

COLOR_TO_INDEX = {
    (0, 0, 0): 0,
    (128, 0, 0): 1,
    (0, 128, 0): 2,
    (0, 0, 128): 3,
    (0, 128, 128): 4,
    (128, 128, 0): 5,
    (128, 0, 128): 6,
}


def mask_to_color_image(pred_mask: np.ndarray, size: Tuple[int, int] = None) -> Image.Image:
    """Convert a HxW index mask to a color-coded PIL image.

    Args:
        pred_mask: ndarray with shape (H, W) and integer values 0..6
        size: optional (width, height) to resize the output image

    Returns:
        PIL.Image in RGB mode
    """
    color_mask = COLOR_MAP[pred_mask]
    img = Image.fromarray(color_mask.astype("uint8"))
    if size is not None:
        img = img.resize(size)
    return img


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rgb_mask_to_index(rgb_mask: np.ndarray) -> np.ndarray:
    """Convert HxWx3 RGB mask to HxW class index array."""
    h, w, _ = rgb_mask.shape
    flat = rgb_mask.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)

    # Map unique colors to indices with a small lookup to avoid large per-pixel loops.
    mapped = np.zeros((unique_colors.shape[0],), dtype=np.uint8)
    for i, color in enumerate(unique_colors):
        key = (int(color[0]), int(color[1]), int(color[2]))
        if key not in COLOR_TO_INDEX:
            raise ValueError(f"Unknown mask color: {key}")
        mapped[i] = COLOR_TO_INDEX[key]

    index_mask = mapped[inverse].reshape(h, w)
    return index_mask
