"""Hold detection + color grouping — Stage A of route tracing.

A gym route *is* all the holds of one color, and holds are saturated colors
bolted onto a fairly neutral wall. So we can find holds and split them into
routes with no trained model: segment the photo in HSV colour space, keep the
saturated blobs, and bucket each one by its hue.

Like `grader.predict` and `detector.is_climbing_wall`, this is a swappable
seam. Later, replace the HSV segmentation with a trained hold detector — the
`Detection` return shape stays the same, so the UI doesn't change.

Note: this first pass only finds *chromatic* holds (red…pink). Black / white /
grey holds have no hue and need a separate path; that's a later improvement.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

# Hue buckets (OpenCV hue is 0..179). Each bucket is one route colour the UI
# can offer. Red wraps around the end of the hue circle, so it gets two ranges.
COLOR_BINS: dict[str, list[tuple[int, int]]] = {
    "red": [(0, 10), (170, 179)],
    "orange": [(11, 22)],
    "yellow": [(23, 33)],
    "green": [(34, 85)],
    "blue": [(86, 125)],
    "purple": [(126, 150)],
    "pink": [(151, 169)],
}

# Display RGB used when drawing each bucket on the overlay.
COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red": (239, 68, 68),
    "orange": (249, 115, 22),
    "yellow": (234, 179, 8),
    "green": (34, 197, 94),
    "blue": (59, 130, 246),
    "purple": (168, 85, 247),
    "pink": (236, 72, 153),
}

# Tunables.
_MAX_DIM = 900  # downscale long edge to this for speed + stable coordinates
_MIN_SAT = 90  # below this saturation a pixel is "wall", not a hold
_MIN_VAL = 50  # below this brightness a pixel is shadow/near-black
_MIN_AREA_FRAC = 0.0004  # a hold must cover at least this fraction of the image
_MAX_AREA_FRAC = 0.02  # ...and no more — bigger blobs are mats/roofs/banners
_MIN_FILL = 0.40  # blob area / bbox area — holds are compact, not scattered
_ASPECT_RANGE = (0.33, 3.0)  # w/h bounds — drops long thin bars (beams, edges)


@dataclass
class Hold:
    color: str
    cx: float  # centroid x, pixels (in the downscaled frame)
    cy: float  # centroid y, pixels
    area: int  # blob area in pixels
    bbox: tuple[int, int, int, int]  # x, y, w, h


@dataclass
class Route:
    color: str
    holds: list[Hold] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.holds)


@dataclass
class Detection:
    holds: list[Hold]
    routes: dict[str, Route]  # color -> Route (only colours actually found)
    width: int  # frame size the coordinates live in
    height: int


def _load_rgb(image_bytes: bytes) -> np.ndarray:
    """Decode to an RGB array, downscaled deterministically so hold
    coordinates from `detect_holds` line up with `annotate`."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size
    scale = min(1.0, _MAX_DIM / max(w, h))
    if scale < 1.0:
        image = image.resize((round(w * scale), round(h * scale)))
    return np.asarray(image)


def detect_holds(image_bytes: bytes) -> Detection:
    """Find chromatic holds and group them into per-colour routes."""
    img = _load_rgb(image_bytes)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    min_area = max(1, int(_MIN_AREA_FRAC * h * w))
    max_area = int(_MAX_AREA_FRAC * h * w)
    open_k = np.ones((3, 3), np.uint8)
    close_k = np.ones((7, 7), np.uint8)

    holds: list[Hold] = []
    for color, ranges in COLOR_BINS.items():
        mask = np.zeros((h, w), np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(
                hsv, (lo, _MIN_SAT, _MIN_VAL), (hi, 255, 255)
            )
        # Drop speckle, then close small gaps within a hold.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

        n, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):  # 0 is the background component
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not (min_area <= area <= max_area):
                continue  # too small (speckle) or too big (mat/roof/banner)
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            if not (_ASPECT_RANGE[0] <= bw / bh <= _ASPECT_RANGE[1]):
                continue  # long thin bar — a beam or edge, not a hold
            if area / (bw * bh) < _MIN_FILL:
                continue  # sparse/scattered region — a hold fills its box
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            cx, cy = float(centroids[i][0]), float(centroids[i][1])
            holds.append(Hold(color, cx, cy, area, (x, y, bw, bh)))

    routes: dict[str, Route] = {}
    for hold in holds:
        routes.setdefault(hold.color, Route(hold.color)).holds.append(hold)

    return Detection(holds=holds, routes=routes, width=w, height=h)


def annotate(
    image_bytes: bytes,
    detection: Detection,
    highlight: str | None = None,
) -> Image.Image:
    """Draw detected holds on the photo.

    With `highlight=None` every hold is boxed in its own colour. With a colour
    name, only that route's holds are drawn (the rest fade to a faint marker)
    so the user can isolate one route.
    """
    img = _load_rgb(image_bytes).copy()
    for hold in detection.holds:
        rgb = COLOR_RGB.get(hold.color, (255, 255, 255))
        x, y, bw, bh = hold.bbox
        if highlight is not None and hold.color != highlight:
            # De-emphasise other routes: thin grey dot at the centroid.
            cv2.circle(img, (int(hold.cx), int(hold.cy)), 3, (120, 120, 120), -1)
            continue
        cv2.rectangle(img, (x, y), (x + bw, y + bh), rgb, 3)
    return Image.fromarray(img)
