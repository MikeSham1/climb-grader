"""Climb Route Grader — first page.

Upload a photo of a bouldering wall, get a predicted V-grade.
Run with:  streamlit run app/main.py
"""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

from grader import V_GRADES, predict

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

st.image(image, caption=uploaded.name, use_container_width=True)

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
