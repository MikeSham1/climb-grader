"""Climb Route Grader — first page.

Upload a photo of a bouldering wall, get a predicted V-grade.
Run with:  streamlit run app/main.py
"""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

from detector import is_climbing_wall
from grader import V_GRADES, predict
from holds import annotate, detect_holds


# Cache the heavy per-image work so clicking the route picker (which reruns the
# whole script) doesn't re-run CLIP or re-segment the photo every time.
@st.cache_data(show_spinner=False)
def cached_wall_check(image_bytes: bytes):
    return is_climbing_wall(image_bytes)


@st.cache_data(show_spinner=False)
def cached_detect(image_bytes: bytes):
    return detect_holds(image_bytes)

st.set_page_config(
    page_title="Climb Route Grader",
    page_icon="🧗",
    layout="centered",
)

# --- light custom styling -------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { max-width: 720px; }
      .grade-badge {
        font-size: 4rem; font-weight: 800; line-height: 1;
        letter-spacing: -0.02em;
      }
      .subtle { color: #8a8f98; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧗 Climb Route Grader")
st.caption("Upload a photo of a bouldering route and get a predicted V-grade.")

uploaded = st.file_uploader(
    "Wall photo",
    type=["jpg", "jpeg", "png", "webp"],
    help="A clear photo of the wall with the route's holds visible.",
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("Drop a wall photo above to grade it.", icon="📷")
    st.stop()

image_bytes = uploaded.getvalue()
image = Image.open(io.BytesIO(image_bytes))

# Gate: only climbing-wall photos get graded.
with st.spinner("Checking the photo…"):
    check = cached_wall_check(image_bytes)

if not check.is_wall:
    st.image(image, caption=uploaded.name, use_container_width=True)
    st.error(
        "That doesn't look like a climbing wall. Upload a clear photo of a "
        "bouldering or rock-climbing wall and I'll grade the route.",
        icon="🚫",
    )
    st.caption(f"Climbing-wall confidence: {check.confidence * 100:.0f}%")
    st.stop()

# Stage A — find holds and group them by colour into candidate routes.
with st.spinner("Finding holds…"):
    detection = cached_detect(image_bytes)

if not detection.routes:
    st.image(image, caption=uploaded.name, use_container_width=True)
    st.warning(
        "Couldn't pick out any coloured holds. Try a closer, brighter, "
        "straight-on photo of the wall.",
        icon="🔍",
    )
    st.stop()

# Each colour is a candidate route — let the climber pick the one they're on.
colors = sorted(detection.routes, key=lambda c: -detection.routes[c].count)
choice = st.radio(
    "Route — pick the hold colour you're climbing",
    ["All"] + colors,
    horizontal=True,
    format_func=lambda c: (
        "All holds"
        if c == "All"
        else f"{c.title()} ({detection.routes[c].count})"
    ),
)
highlight = None if choice == "All" else choice

st.image(
    annotate(image_bytes, detection, highlight),
    caption=(
        f"{len(detection.holds)} holds • {len(detection.routes)} colours"
        if highlight is None
        else f"{choice.title()} route • {detection.routes[choice].count} holds"
    ),
    use_container_width=True,
)

if not st.button("Grade this route", type="primary", use_container_width=True):
    st.stop()

with st.spinner("Analyzing route…"):
    result = predict(image_bytes)

st.divider()

left, right = st.columns([1, 1])
with left:
    st.markdown(
        f"<div class='grade-badge'>{result.grade}</div>", unsafe_allow_html=True
    )
    st.markdown(
        f"<span class='subtle'>±1 range: {result.plus_minus_one}</span>",
        unsafe_allow_html=True,
    )
with right:
    st.metric("Confidence", f"{result.confidence * 100:.0f}%")

# Probability across grades — ordinal, so we show the whole spread.
st.markdown("**Grade distribution**")
st.bar_chart(
    {g: result.distribution.get(g, 0.0) for g in V_GRADES},
    height=200,
)

if result.is_stub:
    st.warning(
        "⚠️ Stub prediction — no trained model is running yet. "
        "This is a reproducible placeholder so the page works end to end. "
        "See DESIGN.md for the model roadmap.",
        icon="🧪",
    )
