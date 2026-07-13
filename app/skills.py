"""Skills coach — what you'll need to send this wall.

Given a route's holds (from `holds.detect_holds`) and its grade, infer the
climbing skills the route demands: crimp strength, dynamic movement, precise
footwork, endurance, and so on. Each skill comes with a reason tied to *this*
route's geometry, so the grade becomes actionable, not just a number.

Same swappable-seam idea as the rest of the app: this is rule-based over hold
geometry today; swap in a learned model later without changing the UI, since
the `Skill` return shape stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grader import V_GRADES
from holds import Hold


@dataclass
class Skill:
    name: str
    reason: str  # why *this* route demands it
    level: float  # 0..1 intensity

    @property
    def label(self) -> str:
        if self.level >= 0.75:
            return "Critical"
        if self.level >= 0.5:
            return "High"
        if self.level >= 0.3:
            return "Moderate"
        return "Light"


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def skills_for(
    holds: list[Hold], grade: str, width: int, height: int
) -> list[Skill]:
    """Rank the skills a route demands, most important first (max 5)."""
    n = len(holds)
    frame_area = max(1, width * height)
    areas = np.array([h.area for h in holds], dtype=float)
    cxs = np.array([h.cx for h in holds], dtype=float)
    cys = np.array([h.cy for h in holds], dtype=float)

    # Hold size — small holds mean crimps and finger strength.
    med_area_frac = float(np.median(areas)) / frame_area if n else 0.0
    med_px = int(round((med_area_frac * frame_area) ** 0.5))
    crimp = _clamp((0.0045 - med_area_frac) / 0.0045)

    # Spacing — big gaps between nearest holds mean reachy, dynamic moves.
    if n >= 2:
        pts = np.stack([cxs, cys], axis=1)
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        nn_frac = float(np.median(d.min(axis=1))) / height
        hspread = float(cxs.max() - cxs.min()) / width
    else:
        nn_frac = hspread = 0.0
    dynamic = _clamp((nn_frac - 0.10) / 0.16)

    # Wide horizontal spread + big moves -> flexibility, high steps, flagging.
    flexibility = _clamp((hspread - 0.40) / 0.45) * (0.5 + 0.5 * dynamic)

    # Route length -> endurance; small/tight holds -> precise footwork.
    endurance = _clamp((n - 6) / 12)
    footwork = _clamp(0.6 * crimp + 0.4 * _clamp(n / 16))

    # Spread of hold sizes -> switching grip styles (crimp <-> sloper).
    variety = 0.0
    if n >= 2 and areas.mean() > 0:
        variety = _clamp((float(areas.std() / areas.mean()) - 0.55) / 0.9)

    # Grade -> commitment and body tension at the crux.
    try:
        gfrac = V_GRADES.index(grade) / (len(V_GRADES) - 1)
    except ValueError:
        gfrac = 0.4
    commitment = _clamp(0.3 + 0.55 * gfrac)

    candidates = [
        Skill("Crimp strength", f"Small holds (~{med_px}px) — you'll crimp hard.", crimp),
        Skill("Dynamic movement", "Big gaps between holds mean reachy, dynamic moves.", dynamic),
        Skill("Precise footwork", "Tight, small holds demand exact foot placement.", footwork),
        Skill("Endurance", f"{n} holds on the line — a long, pumpy route.", endurance),
        Skill("Flexibility & flagging", "Holds spread wide — expect high steps and flags.", flexibility),
        Skill("Grip adaptability", "Mixed hold sizes — switch between crimps and slopers.", variety),
        Skill("Commitment & tension", f"At {grade}, the crux rewards commitment and body tension.", commitment),
    ]

    ranked = sorted(candidates, key=lambda s: -s.level)
    picked = [s for s in ranked if s.level >= 0.33]
    if len(picked) < 3:  # always show a few, even on an easy route
        picked = ranked[:3]
    return picked[:5]
