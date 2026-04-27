"""
Step 4: Render a block letter on a white canvas matching image dimensions.
Used as the selection-bias mask in the statistics meme.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try common bold system fonts; fall back to PIL default bitmap font."""
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    try:
        import matplotlib.font_manager as mfm

        path = mfm.findfont(mfm.FontProperties(family="sans-serif", weight="bold"))
        if path and path.lower().endswith((".ttf", ".ttc", ".otf")):
            return ImageFont.truetype(path, size=size)
    except Exception:
        pass
    return ImageFont.load_default()


def create_block_letter_s(
    height: int,
    width: int,
    letter: str = "S",
    font_size_ratio: float = 0.9,
) -> np.ndarray:
    """
    Draw a single block letter centered on a white background.

    Parameters
    ----------
    height : int
        Output image height in pixels.
    width : int
        Output image width in pixels.
    letter : str
        Character to render (default ``"S"``).
    font_size_ratio : float
        Font size as a fraction of ``min(height, width)`` (clamped for stability).

    Returns
    -------
    np.ndarray
        2D array of shape ``(height, width)`` with values in ``[0, 1]``:
        letter pixels are ``0.0`` (black), background ``1.0`` (white).
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    base = min(height, width)
    font_size = max(8, int(base * float(font_size_ratio)))
    font = _load_bold_font(font_size)

    # Shrink font until the glyph fits inside the canvas
    while font_size >= 8:
        font = _load_bold_font(font_size)
        bbox = draw.textbbox((0, 0), letter, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= width and th <= height:
            break
        font_size = max(8, int(font_size * 0.92))

    bbox = draw.textbbox((0, 0), letter, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (width - tw) / 2.0 - bbox[0]
    y = (height - th) / 2.0 - bbox[1]

    draw.text((x, y), letter, fill=0, font=font)
    out = np.asarray(img, dtype=np.float64) / 255.0
    return np.clip(out, 0.0, 1.0)
