"""
Step 5: Apply the block-letter mask to the stippled image (biased estimate).
"""

from __future__ import annotations

import numpy as np


def create_masked_stipple(
    stipple_img: np.ndarray,
    mask_img: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Remove stipples wherever the mask is dark (selection-bias region).

    Parameters
    ----------
    stipple_img : np.ndarray
        Stippled image, shape ``(H, W)``, values in ``[0, 1]``.
    mask_img : np.ndarray
        Mask image, same shape; ``0`` = letter / remove data, ``1`` = keep.
    threshold : float
        Pixels with ``mask_img < threshold`` are treated as masked and set to white.

    Returns
    -------
    np.ndarray
        Same shape as inputs; stipples cleared to ``1.0`` where the mask is dark.
    """
    if stipple_img.shape != mask_img.shape:
        raise ValueError(
            f"stipple_img shape {stipple_img.shape} does not match mask_img {mask_img.shape}"
        )
    stipple = np.asarray(stipple_img, dtype=np.float64)
    mask = np.asarray(mask_img, dtype=np.float64)
    out = np.where(mask < float(threshold), 1.0, stipple)
    return np.clip(out, 0.0, 1.0)
