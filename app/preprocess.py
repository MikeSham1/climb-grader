"""Image intake — normalise every upload before anything else looks at it.

Phone cameras don't rotate pixels; they set an EXIF orientation flag. So a
portrait shot arrives sideways, and if we don't fix that up front the wall
check, hold detection, and the drawn overlay all run on a rotated image and
come out wrong. We also want a single canonical RGB image that every stage
shares, so the boxes we draw line up with what the user actually sees.

`normalize_image` returns clean RGB JPEG bytes: EXIF-rotated, transparency
flattened onto white, and oversized photos downscaled. It raises `BadImage`
on anything undecodable so the UI can show a friendly message instead of a
stack trace.
"""

from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

# Cap the long edge so a 48-megapixel phone photo doesn't blow up memory or
# CLIP time. Well above what any stage needs; holds.py downscales again anyway.
_MAX_DIM = 2000


class BadImage(Exception):
    """The upload could not be decoded as an image."""


def normalize_image(raw: bytes) -> bytes:
    """Return orientation-corrected, RGB, size-capped JPEG bytes.

    Raises BadImage if `raw` is empty, truncated, or not an image.
    """
    if not raw:
        raise BadImage("empty file")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()  # force a full decode now so truncation errors surface here
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise BadImage(str(exc)) from exc

    # Honour the EXIF orientation flag, then leave the pixels correctly rotated.
    image = ImageOps.exif_transpose(image)

    # Flatten any transparency onto white so alpha PNG/WebP don't go black.
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    image = image.convert("RGB")

    # Downscale very large photos, preserving aspect ratio.
    w, h = image.size
    if max(w, h) > _MAX_DIM:
        scale = _MAX_DIM / max(w, h)
        image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))))

    out = io.BytesIO()
    image.save(out, format="JPEG", quality=90)
    return out.getvalue()
