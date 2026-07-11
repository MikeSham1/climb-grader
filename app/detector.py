"""Climbing-wall detector — the gate before grading.

Before we grade anything, we check that the photo actually shows a climbing
wall. This uses zero-shot CLIP: the image is scored against a set of
"climbing wall" captions and a set of "not a climbing wall" captions, and we
keep the combined probability on the wall side.

The model loads lazily and is cached, so the first call downloads the weights
(~600 MB) and every call after that is fast. Like `grader.predict`, this is a
seam: swap the internals for a fine-tuned classifier later without touching
the UI.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache

from PIL import Image

# CLIP checkpoint — small, fast, good enough for a coarse yes/no.
_MODEL_NAME = "openai/clip-vit-base-patch32"

# Captions describing a climbing wall...
_WALL_PROMPTS = [
    "a photo of an indoor rock climbing wall",
    "a bouldering gym wall with colorful climbing holds",
    "a person climbing a rock wall",
    "an outdoor rock climbing crag",
]

# ...and captions for the kinds of photos people upload by mistake.
_OTHER_PROMPTS = [
    "a photo of a person",
    "a screenshot of an app",
    "a photo of an animal",
    "food on a plate",
    "a landscape with no climbing wall",
    "an ordinary indoor room",
    "a document or piece of text",
]

# Minimum combined wall-probability to accept the photo. Tuned by hand; raise
# it to be stricter, lower it to let borderline photos through.
_THRESHOLD = 0.55


@dataclass
class WallCheck:
    is_wall: bool
    confidence: float  # 0..1 — combined probability the photo is a climbing wall


@lru_cache(maxsize=1)
def _load_model():
    """Load CLIP once and cache it for the life of the process."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
    model.eval()
    return model, processor


def is_climbing_wall(image_bytes: bytes) -> WallCheck:
    """Zero-shot check: is this photo a climbing wall?

    Returns a `WallCheck` with the decision and the wall-side confidence.
    """
    import torch

    model, processor = _load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    prompts = _WALL_PROMPTS + _OTHER_PROMPTS
    inputs = processor(
        text=prompts, images=image, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits_per_image  # (1, n_prompts)
    probs = logits.softmax(dim=1)[0]

    wall_prob = float(probs[: len(_WALL_PROMPTS)].sum())
    return WallCheck(is_wall=wall_prob >= _THRESHOLD, confidence=wall_prob)
