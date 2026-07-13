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
from holds import COLOR_RGB, annotate, detect_holds
from preprocess import BadImage, normalize_image
from skills import skills_for


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

# --- custom styling -------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { max-width: 760px; padding-top: 2.4rem; }
      #MainMenu, footer { visibility: hidden; }

      /* Hero */
      .hero { text-align: center; margin-bottom: 1.4rem; }
      .hero .kicker {
        display: inline-block; font-size: .72rem; font-weight: 700;
        letter-spacing: .16em; text-transform: uppercase; color: #f97316;
        background: rgba(249,115,22,.12); border: 1px solid rgba(249,115,22,.3);
        padding: .34rem .8rem; border-radius: 999px; margin-bottom: 1rem;
      }
      .hero h1 {
        font-size: 2.8rem; font-weight: 800; line-height: 1.04;
        letter-spacing: -.03em; margin: .1rem 0 .6rem;
        background: linear-gradient(135deg,#ffffff 28%,#f97316 72%,#ec4899);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
      }
      .hero p.tag {
        color: #9aa0aa; font-size: 1.05rem; max-width: 34rem;
        margin: 0 auto; line-height: 1.5;
      }

      /* Problem card + how-it-works steps */
      .problem {
        background: linear-gradient(180deg,#161a22,#12151c);
        border: 1px solid #232a36; border-radius: 16px;
        padding: 1.25rem 1.4rem; margin: 1.6rem 0 1.4rem;
      }
      .problem h3 { margin: 0 0 .45rem; font-size: .82rem; color: #f97316;
        text-transform: uppercase; letter-spacing: .12em; }
      .problem p { margin: 0; color: #c3c8d1; line-height: 1.6; font-size: .98rem; }
      .steps { display: flex; gap: .8rem; }
      .step {
        flex: 1; background: #12151c; border: 1px solid #232a36;
        border-radius: 14px; padding: 1rem .9rem; text-align: center;
      }
      .step .n {
        display: inline-flex; align-items: center; justify-content: center;
        width: 1.9rem; height: 1.9rem; border-radius: 999px; margin-bottom: .55rem;
        background: rgba(249,115,22,.14); color: #f97316; font-weight: 800;
      }
      .step .t { font-weight: 700; font-size: .92rem; margin-bottom: .2rem; }
      .step .d { color: #8a90a0; font-size: .8rem; line-height: 1.45; }

      /* Route chip + result card */
      .route-chip {
        display: inline-flex; align-items: center; gap: .45rem; font-size: .82rem;
        color: #c3c8d1; background: #161a22; border: 1px solid #232a36;
        padding: .32rem .75rem; border-radius: 999px; margin-bottom: .9rem;
      }
      .route-chip .dot { width: .72rem; height: .72rem; border-radius: 999px; }
      .result {
        border-radius: 20px; padding: 1.5rem 1.7rem; margin: .2rem 0 1.3rem;
        border: 1px solid #2a2130;
        background: radial-gradient(130% 150% at 0% 0%,
          rgba(249,115,22,.16), rgba(236,72,153,.06) 45%, #12151c 76%);
      }
      .result .row {
        display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem;
      }
      .result .grade {
        font-size: 4.4rem; font-weight: 800; line-height: .9; letter-spacing: -.03em;
        background: linear-gradient(135deg,#f97316,#ec4899);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
      }
      .result .range { color: #9aa0aa; font-size: .9rem; margin-top: .45rem; }
      .result .conf-num { font-size: 2.1rem; font-weight: 800; text-align: right; line-height: 1; }
      .result .conf-lbl {
        color: #8a90a0; font-size: .72rem; text-transform: uppercase;
        letter-spacing: .12em; text-align: right; margin-top: .3rem;
      }
      .meter { height: .5rem; border-radius: 999px; background: #222834;
        margin-top: 1.15rem; overflow: hidden; }
      .meter > span { display: block; height: 100%; border-radius: 999px;
        background: linear-gradient(90deg,#f97316,#ec4899); }

      /* Skills coach */
      .skills-title { font-size: 1.05rem; font-weight: 700; margin: 1.4rem 0 .7rem; }
      .skill {
        background: #12151c; border: 1px solid #232a36; border-radius: 12px;
        padding: .8rem .95rem; margin-bottom: .6rem;
      }
      .skill-head { display: flex; align-items: center; justify-content: space-between; }
      .skill-name { font-weight: 700; font-size: .95rem; }
      .skill-lvl {
        font-size: .68rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: .08em; padding: .18rem .5rem; border-radius: 999px;
      }
      .lvl-critical { color: #fca5a5; background: rgba(239,68,68,.15); }
      .lvl-high { color: #fdba74; background: rgba(249,115,22,.15); }
      .lvl-moderate { color: #fde047; background: rgba(234,179,8,.14); }
      .lvl-light { color: #94a3b8; background: rgba(148,163,184,.14); }
      .skill-bar { height: .4rem; border-radius: 999px; background: #222834;
        margin: .55rem 0 .5rem; overflow: hidden; }
      .skill-bar > span { display: block; height: 100%; border-radius: 999px;
        background: linear-gradient(90deg,#f97316,#ec4899); }
      .skill-why { color: #9aa0aa; font-size: .84rem; line-height: 1.4; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <span class="kicker">🧗 AI Route Grading</span>
      <h1>Is it really a V5?</h1>
      <p class="tag">Bouldering grades are subjective — setters vary and gyms
      sandbag. Snap the wall, isolate your route by hold colour, and get a
      consistent, explainable grade.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Wall photo",
    type=["jpg", "jpeg", "png", "webp"],
    help="A clear photo of the wall with the route's holds visible.",
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown(
        """
        <div class="problem">
          <h3>The problem</h3>
          <p>Two climbers look at the same wall and guess two different grades.
          Setters vary, gyms sandbag, and newer climbers genuinely can't tell a
          V3 from a V6. Climb Grader reads the <b>actual holds</b> and gives you a
          consistent, explainable estimate — plus the skills you'll need to send it.</p>
        </div>
        <div class="steps">
          <div class="step"><div class="n">1</div>
            <div class="t">📷 Snap the wall</div>
            <div class="d">Upload a photo of your boulder problem.</div></div>
          <div class="step"><div class="n">2</div>
            <div class="t">🎯 Pick your route</div>
            <div class="d">We find the holds — choose your colour.</div></div>
          <div class="step"><div class="n">3</div>
            <div class="t">🧗 Get the beta</div>
            <div class="d">A grade, confidence, and the skills it takes.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Normalise first: fix phone-photo rotation, flatten transparency, cap size —
# so every stage below shares one clean, correctly-oriented image.
try:
    image_bytes = normalize_image(uploaded.getvalue())
except BadImage:
    st.error(
        "Couldn't read that file as an image. Try a JPG, PNG, or WebP photo "
        "of the wall.",
        icon="🚫",
    )
    st.stop()
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

# The holds we're actually grading: the picked route, or everything.
route_holds = detection.routes[choice].holds if highlight else detection.holds
dot = COLOR_RGB.get(highlight, (249, 115, 22))
route_label = "all detected holds" if highlight is None else f"the {choice} route"
conf = round(result.confidence * 100)

st.markdown(
    f"""
    <div class="route-chip">
      <span class="dot" style="background: rgb{dot};"></span>
      Grading {route_label} · {len(route_holds)} holds
    </div>
    <div class="result">
      <div class="row">
        <div>
          <div class="grade">{result.grade}</div>
          <div class="range">±1 range · {result.plus_minus_one}</div>
        </div>
        <div>
          <div class="conf-num">{conf}%</div>
          <div class="conf-lbl">confidence</div>
        </div>
      </div>
      <div class="meter"><span style="width: {conf}%;"></span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Skills coach — what this specific route demands.
profile = skills_for(route_holds, result.grade, detection.width, detection.height)
skill_rows = "".join(
    f"""
    <div class="skill">
      <div class="skill-head">
        <span class="skill-name">{s.name}</span>
        <span class="skill-lvl lvl-{s.label.lower()}">{s.label}</span>
      </div>
      <div class="skill-bar"><span style="width: {round(s.level * 100)}%;"></span></div>
      <div class="skill-why">{s.reason}</div>
    </div>
    """
    for s in profile
)
st.markdown(
    f'<div class="skills-title">💪 What you\'ll need on this wall</div>{skill_rows}',
    unsafe_allow_html=True,
)

# Probability across grades — ordinal, so we show the whole spread.
st.markdown('<div class="skills-title">Grade distribution</div>', unsafe_allow_html=True)
st.bar_chart(
    {g: result.distribution.get(g, 0.0) for g in V_GRADES},
    height=200,
    color="#f97316",
)

if result.is_stub:
    st.caption(
        "🧪 Holds and skills are computed for real from your photo. The grade "
        "itself is still a reproducible placeholder while the grading model is built."
    )
