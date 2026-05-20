import os
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter

# Canonical mask colors are defined in BGR (OpenCV source of truth).
CLASS_COLORS_BGR = np.array([
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [0, 128, 128],
    [128, 128, 0],
    [128, 0, 128],
], dtype=np.uint8)

# RGB palette for display/control images.
CLASS_COLORS_RGB = CLASS_COLORS_BGR[:, ::-1]

COLOR_MAP = CLASS_COLORS_RGB

BGR_COLOR_TO_INDEX = {tuple(map(int, color)): idx for idx, color in enumerate(CLASS_COLORS_BGR)}


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


RGB_COLOR_TO_INDEX = {tuple(map(int, color)): idx for idx, color in enumerate(CLASS_COLORS_RGB)}


def rgb_mask_to_index(rgb_mask: np.ndarray) -> np.ndarray:
    """Convert HxWx3 RGB mask to HxW class index array."""
    h, w, _ = rgb_mask.shape
    flat = rgb_mask.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)

    # Map unique colors to indices with a small lookup to avoid large per-pixel loops.
    mapped = np.zeros((unique_colors.shape[0],), dtype=np.uint8)
    for i, color in enumerate(unique_colors):
        key = (int(color[0]), int(color[1]), int(color[2]))
        if key not in RGB_COLOR_TO_INDEX:
            raise ValueError(f"Unknown mask color: {key}")
        mapped[i] = RGB_COLOR_TO_INDEX[key]

    index_mask = mapped[inverse].reshape(h, w)
    return index_mask


def bgr_mask_to_index(bgr_mask: np.ndarray) -> np.ndarray:
    """Convert HxWx3 BGR mask to HxW class index array."""
    if bgr_mask.ndim != 3 or bgr_mask.shape[2] != 3:
        raise ValueError("bgr_mask must be HxWx3.")

    h, w, _ = bgr_mask.shape
    label_mask = np.full((h, w), fill_value=255, dtype=np.uint8)
    for idx, color in enumerate(CLASS_COLORS_BGR):
        match = np.all(bgr_mask == color, axis=-1)
        label_mask[match] = idx

    if (label_mask == 255).any():
        unknown = bgr_mask[label_mask == 255]
        unique_unknown = np.unique(unknown.reshape(-1, 3), axis=0)
        raise ValueError(f"Unknown mask colors found: {unique_unknown.tolist()}")

    return label_mask


def bgr_mask_to_rgb_image(bgr_mask: np.ndarray) -> Image.Image:
    """Convert a BGR mask array to an RGB PIL image for display."""
    rgb = bgr_mask[:, :, ::-1]
    return Image.fromarray(rgb.astype("uint8"))


def build_preserve_alpha(
    index_mask: np.ndarray,
    preserve_indices: list[int],
    feather_radius: int = 0,
    out_size: Tuple[int, int] | None = None,
) -> Image.Image:
    """Build an alpha mask for preserved classes.

    Args:
        index_mask: HxW class index mask.
        preserve_indices: list of class indices to preserve.
        feather_radius: optional Gaussian blur radius to soften edges.
        out_size: optional (width, height) to resize alpha mask.

    Returns:
        PIL.Image in L mode with 255 where preservation should apply.
    """
    if index_mask.ndim != 2:
        raise ValueError("index_mask must be HxW with class indices.")

    h, w = index_mask.shape
    if not preserve_indices:
        alpha = Image.new("L", (w, h), 0)
    else:
        preserve = np.isin(index_mask, preserve_indices).astype(np.uint8) * 255
        alpha = Image.fromarray(preserve, mode="L")

    if out_size is not None and alpha.size != out_size:
        alpha = alpha.resize(out_size, resample=Image.NEAREST)

    if feather_radius and feather_radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    return alpha


def blend_preserved_regions(
    original: Image.Image,
    generated: Image.Image,
    alpha: Image.Image,
) -> Image.Image:
    """Composite original pixels back into generated output using alpha mask.

    Args:
        original: original RGB image.
        generated: generated RGB image.
        alpha: L-mode alpha mask where 255 keeps original.

    Returns:
        Blended RGB image.
    """
    if alpha.mode != "L":
        alpha = alpha.convert("L")

    if generated.size != original.size:
        original = original.resize(generated.size, resample=Image.BICUBIC)

    if alpha.size != generated.size:
        alpha = alpha.resize(generated.size, resample=Image.NEAREST)

    return Image.composite(original, generated, alpha)
