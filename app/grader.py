"""Grade predictor interface.

This is the seam between the UI (first page) and the model. Right now it
returns a deterministic *stub* prediction so the page is fully interactive
before any model exists. When Stage A (hold detection) and Stage B (grader)
from DESIGN.md are ready, replace `predict` internals — the return shape and
the rest of the app stay the same.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# V-scale bouldering grades the app can output.
V_GRADES = [f"V{i}" for i in range(0, 13)]


@dataclass
class GradePrediction:
    grade: str  # e.g. "V5" — the top prediction
    confidence: float  # 0..1 for the top grade
    distribution: dict[str, float] = field(default_factory=dict)  # grade -> prob
    is_stub: bool = True  # so the UI can be honest that no model ran

    @property
    def plus_minus_one(self) -> str:
        """The ±1 range that is the realistic human-agreement target."""
        i = V_GRADES.index(self.grade)
        lo = V_GRADES[max(0, i - 1)]
        hi = V_GRADES[min(len(V_GRADES) - 1, i + 1)]
        return f"{lo}–{hi}"


def _stub_distribution(seed: int) -> dict[str, float]:
    """Deterministic bell-ish distribution over grades, keyed by image bytes.

    Same photo -> same grade every time, which makes the page feel real and
    keeps the demo reproducible. No learning happens here.
    """
    center = seed % len(V_GRADES)
    weights = []
    for i in range(len(V_GRADES)):
        d = abs(i - center)
        weights.append(max(0.0, 1.0 - d * 0.35))
    total = sum(weights) or 1.0
    return {g: w / total for g, w in zip(V_GRADES, weights)}


def predict(image_bytes: bytes) -> GradePrediction:
    """Return a grade prediction for a wall photo.

    STUB: hashes the image to a reproducible pseudo-grade. Swap the body for
    the real perception -> grading pipeline when the model lands.
    """
    seed = int.from_bytes(hashlib.sha256(image_bytes).digest()[:4], "big")
    dist = _stub_distribution(seed)
    grade = max(dist, key=dist.get)
    return GradePrediction(
        grade=grade,
        confidence=dist[grade],
        distribution=dist,
        is_stub=True,
    )
