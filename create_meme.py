"""
Assemble the four-panel statistics meme and save as PNG.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image


def _resize_to_shape(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize 2D grayscale ``img`` to ``(height, width)``."""
    target_h, target_w = shape
    if img.shape == (target_h, target_w):
        return np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
    g = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
    u8 = np.round(g * 255.0).astype(np.uint8)
    pil = Image.fromarray(u8, mode="L")
    pil = pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.clip(np.asarray(pil, dtype=np.float64) / 255.0, 0.0, 1.0)


def create_statistics_meme(
    original_img: np.ndarray,
    stipple_img: np.ndarray,
    block_letter_img: np.ndarray,
    masked_stipple_img: np.ndarray,
    output_path: str,
    dpi: int = 150,
    background_color: str = "white",
) -> None:
    """
    Create a 1×4 figure (Reality | Your Model | Selection Bias | Estimate) and save to disk.

    Uses a single-row ``GridSpec`` so four panels sit side by side with room for titles,
    borders, and the chosen background color. Panels are resized to match ``original_img``
    when shapes differ. ``dpi`` should be 150–300 for publication-style output.
    """
    target_shape = (int(original_img.shape[0]), int(original_img.shape[1]))
    panels = [
        _resize_to_shape(original_img, target_shape),
        _resize_to_shape(stipple_img, target_shape),
        _resize_to_shape(block_letter_img, target_shape),
        _resize_to_shape(masked_stipple_img, target_shape),
    ]
    titles = ["Reality", "Your Model", "Selection Bias", "Estimate"]

    # Wide figure + enough height so panel titles (above each image) are not clipped
    fig = plt.figure(figsize=(18.0, 5.6), facecolor=background_color)
    fig.patch.set_facecolor(background_color)

    gs = GridSpec(
        1,
        4,
        figure=fig,
        width_ratios=[1, 1, 1, 1],
        wspace=0.22,
        left=0.045,
        right=0.985,
        top=0.86,
        bottom=0.12,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    for ax, data, title in zip(axes, panels, titles):
        ax.set_facecolor(background_color)
        ax.imshow(data, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=13, fontweight="bold", color="#1a1a1a", pad=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.25)
            spine.set_edgecolor("#3a3a3a")

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.25,
        facecolor=background_color,
        edgecolor=background_color,
        format="png",
    )
    plt.close(fig)
