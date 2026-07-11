"""
OCR Readiness Evaluation Platform
SNLP Department — Team: Yash (Lead), Vivek, Mansi, Krish, Tanusha
Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import cv2

from factors import run_all_factors, generate_recommendations, DISPLAY_NAMES, WEIGHTS
from factor_info import FACTOR_INFO
from storage import (save_result,load_results,compute_correlations,save_uploaded_image)
from report import generate_pdf_report
from descriptions import get_factor_description
from short_descriptions import get_short_description
from api_integration import call_all_team_apis, KNOWN_ISSUES, get_current_urls
from config_manager import load_config, save_config, build_urls, PORTS, ENDPOINTS

try:
    import pytesseract
    # ── SET YOUR TESSERACT PATH HERE ──────────────────────────────────────
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # ─────────────────────────────────────────────────────────────────────
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False

try:
    from streamlit_cropper import st_cropper
    CROPPER_OK = True
except ImportError:
    CROPPER_OK = False

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCR Readiness Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}

[data-testid="stSidebar"]{background:linear-gradient(180deg,#1A2B4A 0%,#0F1E36 100%);}
[data-testid="stSidebar"] *{color:#E8EDF5 !important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#00C4B4 !important;}

.top-banner{background:linear-gradient(90deg,#1A2B4A 0%,#0F3460 50%,#00C4B4 100%);
    padding:18px 24px;border-radius:12px;margin-bottom:20px;color:white;}
.top-banner h1{font-family:'Space Grotesk',sans-serif;font-size:24px;font-weight:700;margin:0;color:white !important;}
.top-banner p{font-size:13px;color:rgba(255,255,255,0.7);margin:4px 0 0 0;}

.metric-card{background:white;border-radius:12px;padding:16px;
    box-shadow:0 2px 8px rgba(0,0,0,0.07);border-top:3px solid #00C4B4;
    text-align:center;margin-bottom:8px;position:relative;}
.metric-card .label{font-size:13px;font-weight:700;color:#374151;
    text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;}
.metric-card .value{font-size:28px;font-weight:700;
    font-family:'Space Grotesk',sans-serif;}
.metric-card .badge{display:inline-block;padding:3px 12px;border-radius:20px;
    font-size:13px;font-weight:600;margin-top:4px;}
.badge-excellent{background:#D1FAE5;color:#065F46;}
.badge-good{background:#DBEAFE;color:#1E40AF;}
.badge-average{background:#FEF3C7;color:#92400E;}
.badge-poor{background:#FEE2E2;color:#991B1B;}
.badge-error{background:#F3F4F6;color:#6B7280;}
.metric-card .short-desc{font-size:13px;color:#4B5563;margin-top:7px;
    font-style:italic;font-weight:500;line-height:1.4;}
.metric-card .weight-src{font-size:11px;color:#9CA3AF;margin-top:5px;}

/* ── Pure CSS toggle — no JS/rerun needed ── */
.toggle-check{display:none;}
.toggle-arrow{
    display:block;margin:10px auto 0 auto;
    width:28px;height:28px;line-height:28px;text-align:center;
    background:#F3F4F6;border-radius:50%;cursor:pointer;
    font-size:14px;color:#6B7280;user-select:none;
    transition:background 0.2s,transform 0.2s;}
.toggle-arrow:hover{background:#E5E7EB;color:#374151;}
.card-info-panel{
    display:none;
    background:#EFF6FF;border-left:3px solid #3B82F6;
    border-radius:8px;padding:10px 12px;margin-top:10px;
    text-align:left;font-size:11px;color:#1E3A5F;line-height:1.7;
    overflow:hidden;word-wrap:break-word;word-break:break-word;
    white-space:normal;max-width:100%;box-sizing:border-box;}
.card-info-panel .info-title{font-weight:700;font-size:12px;
    color:#1A2B4A;margin-bottom:6px;}
.card-info-panel .info-row{margin-bottom:5px;word-wrap:break-word;
    overflow-wrap:break-word;white-space:normal;}
.card-info-panel .info-label{font-weight:600;color:#3B82F6;}
/* When checkbox is checked: show panel and rotate arrow */
.toggle-check:checked ~ .card-info-panel{display:block;}
.toggle-check:checked ~ .toggle-arrow{
    background:#DBEAFE;color:#1D4ED8;transform:rotate(180deg);}

.score-ring-wrap{display:flex;flex-direction:column;align-items:center;
    justify-content:center;padding:20px;
    background:linear-gradient(135deg,#1A2B4A,#0F3460);
    border-radius:16px;color:white;height:100%;}
.score-ring-label{font-size:12px;font-weight:600;letter-spacing:0.1em;
    text-transform:uppercase;color:#00C4B4;margin-bottom:4px;}
.score-ring-number{font-size:52px;font-weight:700;
    font-family:'Space Grotesk',sans-serif;line-height:1;}
.score-ring-status{font-size:16px;font-weight:500;color:#A8B8D0;margin-top:4px;}

.rec-box{background:#F0FDF4;border-left:4px solid #10B981;
    padding:10px 14px;border-radius:0 8px 8px 0;
    margin-bottom:8px;font-size:14px;color:#1F2937;}
.rec-box.warn{background:#FFFBEB;border-left-color:#F59E0B;}

.info-card{background:#F9FAFB;border:1px solid #E5E7EB;
    border-radius:10px;padding:16px;margin-bottom:12px;}
.info-card h4{color:#1A2B4A;margin:0 0 8px 0;
    font-family:'Space Grotesk',sans-serif;}
.owner-tag{display:inline-block;background:#EFF6FF;color:#1D4ED8;
    font-size:11px;font-weight:600;padding:2px 8px;
    border-radius:20px;margin-bottom:8px;}

.api-row{padding:6px 12px;border-radius:6px;margin-bottom:4px;font-size:12px;font-weight:500;}
.api-yash{background:#D1FAE5;color:#065F46;}
.api-mansi{background:#EDE9FE;color:#5B21B6;}
.api-krish{background:#DBEAFE;color:#1E40AF;}
.api-vivek{background:#FFEDD5;color:#9A3412;}
.api-tanusha{background:#FCE7F3;color:#9D174D;}
.api-local{background:#F3F4F6;color:#374151;}

.desc-box{background:#F8FAFC;border-left:3px solid #00C4B4;
    padding:12px 16px;border-radius:0 8px 8px 0;
    font-size:13px;color:#374151;line-height:1.6;margin-top:8px;}

.clear-btn{text-align:right;margin-bottom:8px;}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def pil_to_bgr(img):
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def score_color(s):
    if s >= 81: return "#10B981"
    if s >= 61: return "#3B82F6"
    if s >= 41: return "#F59E0B"
    return "#EF4444"

def run_tesseract(img):
    if not TESSERACT_OK:
        return None, None
    try:
        data  = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confs = [c for c in data["conf"] if isinstance(c,(int,float)) and c >= 0]
        if confs:
            return round(float(np.mean(confs)),1), pytesseract.image_to_string(img)
        return None, None
    except Exception:
        return None, None

def make_radar(factor_results):
    keys   = list(DISPLAY_NAMES.keys())
    labels = [DISPLAY_NAMES[k] for k in keys]
    scores = [factor_results.get(k,{}).get("score",0) for k in keys]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores+[scores[0]], theta=labels+[labels[0]],
        fill="toself", fillcolor="rgba(0,196,180,0.15)",
        line=dict(color="#00C4B4",width=2), name="Your Image",
        marker=dict(color="#1A2B4A",size=7),
    ))
    fig.add_trace(go.Scatterpolar(
        r=[70]*len(labels)+[70], theta=labels+[labels[0]],
        fill="none",
        line=dict(color="rgba(245,158,11,0.5)",width=1.5,dash="dot"),
        name="Good threshold (70)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#F9FAFB",
            radialaxis=dict(range=[0,100],tickfont=dict(size=9),
                gridcolor="#E5E7EB",linecolor="#D1D5DB"),
            angularaxis=dict(tickfont=dict(size=11,color="#374151"),
                linecolor="#E5E7EB"),
        ),
        showlegend=True,
        legend=dict(orientation="h",y=-0.15,font=dict(size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20,b=20,l=40,r=40),
        height=430,
    )
    return fig

# ── Session state init ────────────────────────────────────────────────────────
# This keeps the image and results alive when user switches pages
if "analysis_done"    not in st.session_state: st.session_state.analysis_done    = False
if "final_results"    not in st.session_state: st.session_state.final_results    = {}
if "api_status"       not in st.session_state: st.session_state.api_status       = {}
if "ocr_conf"         not in st.session_state: st.session_state.ocr_conf         = None
if "ocr_text"         not in st.session_state: st.session_state.ocr_text         = ""
if "image_name"       not in st.session_state: st.session_state.image_name       = ""
if "raw_pil"          not in st.session_state: st.session_state.raw_pil          = None
if "analysis_img"     not in st.session_state: st.session_state.analysis_img     = None
if "recs"             not in st.session_state: st.session_state.recs             = []
if "card_info_open"   not in st.session_state: st.session_state.card_info_open   = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 OCR Quality Score Project")
    st.markdown("---")

    nav = st.radio("Navigation", [
        "🏠 Analyse Image",
        "📊 History",
        "📖 About Factors",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Score Scale**")
    for lbl,rng,col in [
        ("Excellent","81–100","#10B981"),
        ("Good","61–80","#3B82F6"),
        ("Average","41–60","#F59E0B"),
        ("Poor","0–40","#EF4444"),
    ]:
        st.markdown(
            f'<span style="color:{col};font-weight:600;">■</span> {lbl} ({rng})',
            unsafe_allow_html=True)

    # Show status of current analysis in sidebar
    if st.session_state.analysis_done:
        st.markdown("---")
        st.markdown("**Last Analysis**")
        sc = st.session_state.final_results.get("ocr_readiness_score", 0)
        st.markdown(
            f'<span style="color:{score_color(sc)};font-size:22px;font-weight:700;">{sc}</span> '
            f'<span style="color:#A8B8D0;font-size:12px;">/ 100</span>',
            unsafe_allow_html=True)
        st.caption(f"📄 {st.session_state.image_name}")
        if st.button("🗑️ Clear Analysis", width="stretch"):
            for key in ["analysis_done","final_results","api_status",
                        "ocr_conf","ocr_text","image_name","raw_pil","analysis_img","recs"]:
                st.session_state[key] = False if key=="analysis_done" else ({} if "results" in key or "status" in key else (None if key in ["ocr_conf","raw_pil","analysis_img"] else ([] if key=="recs" else "")))
            st.rerun()


# ════════════════════════════════════════════════
# PAGE 1 — Analyse Image
# ════════════════════════════════════════════════
if "🏠 Analyse Image" in nav:

    st.markdown("""
    <div class="top-banner">
      <h1>🔍 OCR Readiness Evaluation Platform</h1>
      <p>Upload → Crop (optional) → 10 Factor Scores → OCR Readiness Score → Tesseract Validation → PDF Report</p>
    </div>""", unsafe_allow_html=True)

    # --------------------------------------------------
    # Welcome message before image upload
    # --------------------------------------------------

    if uploaded_file is None:

        st.markdown("""
    <div style="
        background:#1E2A52;
        border-radius:18px;
        padding:30px;
        text-align:center;
        margin-top:25px;
        border:1px solid #3A4F87;
    ">

    <h2 style="color:#5B8CFF;">
        📄 Welcome to the OCR Readiness Evaluation Platform
    </h2>

    <p style="
        color:white;
        font-size:18px;
        line-height:1.7;
    ">

    Upload a document image using the <b>Upload Image</b> button above to begin the analysis.

    <br><br>

    The platform will automatically evaluate the image quality, calculate OCR Readiness Score, estimate OCR Confidence, analyze all quality factors, and generate recommendations to improve OCR performance.

    <br><br>

    👆 Please upload an image to continue.

    </p>

    </div>
    """, unsafe_allow_html=True)

    st.stop()

    # ── Upload ──────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a document image",
        type=["png","jpg","jpeg","bmp","tiff","webp"],
    )

    if uploaded is not None:
        raw_pil = Image.open(uploaded).convert("RGB")
        # Only reset if a NEW image is uploaded
        if uploaded.name != st.session_state.image_name:
            st.session_state.raw_pil      = raw_pil
            st.session_state.image_name   = uploaded.name
            st.session_state.analysis_done = False
            st.session_state.final_results = {}
            st.session_state.analysis_img  = raw_pil
    elif st.session_state.raw_pil is None:
        st.info("👆 Upload a document image to begin the analysis.")
        st.stop()

    raw_pil    = st.session_state.raw_pil
    image_name = st.session_state.image_name

    # ── Step 1: Crop ─────────────────────────────
    st.markdown("### Step 1 — Select Region")
    use_crop = st.checkbox("✂️ Crop the image before analysis")

    if use_crop:
        if CROPPER_OK:
            st.markdown("**Drag the handles to select the region you want to analyse:**")
            cropped = st_cropper(
                raw_pil,
                realtime_update=True,
                box_color="#00C4B4",
                aspect_ratio=None,
            )
            col_prev, col_info = st.columns([2,1])
            with col_prev:
                st.image(cropped, caption="Selected crop region", width="stretch")
            with col_info:
                w, h = cropped.size
                st.metric("Width", f"{w} px")
                st.metric("Height", f"{h} px")
            st.session_state.analysis_img = cropped
        else:
            st.warning("streamlit-cropper not installed. Using slider crop instead.")
            c1, c2 = st.columns(2)
            with c1:
                st.image(raw_pil, caption="Original Image", width="stretch")
            with c2:
                w, h = raw_pil.size
                left   = st.slider("Left",   0, w-1, 0)
                top    = st.slider("Top",    0, h-1, 0)
                right  = st.slider("Right",  1, w,   w)
                bottom = st.slider("Bottom", 1, h,   h)
                if right  <= left:  right  = left  + 1
                if bottom <= top:   bottom = top   + 1
                cropped = raw_pil.crop((left, top, right, bottom))
                st.image(cropped, caption="Cropped Region", width="stretch")
            st.session_state.analysis_img = cropped
    else:
        st.image(raw_pil, caption="Full image — will be analysed", width="stretch")
        st.session_state.analysis_img = raw_pil

    analysis_img = st.session_state.analysis_img

    # ── Step 2: Analyse ──────────────────────────
    st.markdown("### Step 2 — Run Analysis")
    use_apis = st.checkbox(
        "🔌 Use team APIs (Vivek · Mansi · Krish · Tanusha) — falls back to local if any API is offline",
        value=True,
    )

    if st.button("🚀 Analyse Image", type="primary", width="stretch"):

        bgr = pil_to_bgr(analysis_img)

        with st.spinner("⚙️ Computing local factors…"):
            local_results = run_all_factors(bgr)

        final_results = local_results
        api_status    = {}

        if use_apis:
            with st.spinner("🔌 Calling team APIs (Vivek · Mansi · Krish · Tanusha)…"):
                final_results, api_status = call_all_team_apis(bgr, local_results)

        recs = generate_recommendations(final_results)

        with st.spinner("📝 Running Tesseract OCR…"):
            ocr_conf, ocr_text = run_tesseract(analysis_img)

        ocr_readiness = final_results["ocr_readiness_score"]

        save_uploaded_image(analysis_img,image_name)
        save_result(image_name,final_results,ocr_readiness,ocr_conf)

        # Store everything in session state
        st.session_state.analysis_done = True
        st.session_state.final_results = final_results
        st.session_state.api_status    = api_status
        st.session_state.ocr_conf      = ocr_conf
        st.session_state.ocr_text      = ocr_text or ""
        st.session_state.recs          = recs

    # ── Show results if analysis has been done ────
    if st.session_state.analysis_done:

        final_results = st.session_state.final_results
        api_status    = st.session_state.api_status
        ocr_conf      = st.session_state.ocr_conf
        ocr_text      = st.session_state.ocr_text
        recs          = st.session_state.recs
        ocr_readiness = final_results["ocr_readiness_score"]
        ocr_stat      = final_results["ocr_readiness_status"]

        # ── API source summary ───────────────────
        if api_status:
            with st.expander("🔌 Data source for each factor"):
                for key, src in api_status.items():
                    disp = DISPLAY_NAMES.get(key, key)
                    if "Yash" in src:       css = "api-yash"
                    elif "Mansi" in src:    css = "api-mansi"
                    elif "Krish" in src:    css = "api-krish"
                    elif "Vivek" in src:    css = "api-vivek"
                    elif "Tanusha" in src:  css = "api-tanusha"
                    else:                   css = "api-local"
                    st.markdown(
                        f'<div class="api-row {css}"><b>{disp}</b>: {src}</div>',
                        unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Results")

        # ── 3 score rings ────────────────────────
        r1, r2, r3 = st.columns(3)

        with r1:
            c = score_color(ocr_readiness)
            st.markdown(f"""
            <div class="score-ring-wrap">
              <div class="score-ring-label">OCR Readiness Score</div>
              <div class="score-ring-number" style="color:{c};">{ocr_readiness}</div>
              <div class="score-ring-status">{ocr_stat}</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            if ocr_conf is not None:
                c2 = score_color(ocr_conf)
                st.markdown(f"""
                <div class="score-ring-wrap">
                  <div class="score-ring-label">Tesseract Confidence</div>
                  <div class="score-ring-number" style="color:{c2};">{ocr_conf}%</div>
                  <div class="score-ring-status">Actual OCR</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="score-ring-wrap">
                  <div class="score-ring-label">Tesseract Confidence</div>
                  <div class="score-ring-number" style="color:#6B7280;">N/A</div>
                  <div class="score-ring-status">Not available</div>
                </div>""", unsafe_allow_html=True)

        with r3:
            if ocr_conf is not None:
                diff = ocr_readiness - ocr_conf
                dc   = "#10B981" if abs(diff)<10 else "#F59E0B" if abs(diff)<20 else "#EF4444"
                lbl  = "Predicted ≈ Actual ✓" if abs(diff)<10 else "Small deviation" if abs(diff)<20 else "Large deviation"
                st.markdown(f"""
                <div class="score-ring-wrap">
                  <div class="score-ring-label">Predicted vs Actual</div>
                  <div class="score-ring-number" style="color:{dc};">{diff:+.1f}</div>
                  <div class="score-ring-status">{lbl}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="score-ring-wrap">
                  <div class="score-ring-label">Predicted vs Actual</div>
                  <div class="score-ring-number" style="color:#6B7280;">—</div>
                  <div class="score-ring-status">Tesseract needed</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 10 factor cards ─────────────────────
        st.markdown("#### Factor Scores")
        cols = st.columns(4)
        for i, (key, display) in enumerate(DISPLAY_NAMES.items()):
            r    = final_results.get(key, {})
            sc   = r.get("score", 0)
            s    = r.get("status", "—")
            col  = score_color(float(sc))
            bcls = f"badge-{s.lower()}"
            src_tag = ""
            if api_status:
                src = api_status.get(key, "")
                if "✅" in src:
                    src_tag = '<span style="font-size:11px;color:#065F46;font-weight:600;">● API</span>'
                else:
                    src_tag = '<span style="font-size:11px;color:#9CA3AF;">● Local</span>'

            short_desc = get_short_description(key, float(sc))

            # Build info panel content
            info_html = ""
            if key in FACTOR_INFO:
                fi    = FACTOR_INFO[key]
                ideal = fi["ideal_range"].split(".")[0]

                # Clean text — remove newlines, bullets, special chars that break HTML
                def clean_text(text, limit=150):
                    import html
                    t = text.replace("\n", " ").replace("•", "-").replace("·", "-")
                    t = html.escape(t)   # escapes <, >, &, ", '
                    t = t[:limit] + "…" if len(t) > limit else t
                    return t

                defn  = clean_text(fi["definition"], 150)
                imp   = clean_text(fi["ocr_impact"], 150)
                ideal_clean = clean_text(ideal, 100)

                info_html = f"""
                <div class="info-row"><span class="info-label">&#128100; Owner:</span> {fi['owner']}</div>
                <div class="info-row"><span class="info-label">&#128214; What it is:</span> {defn}</div>
                <div class="info-row"><span class="info-label">&#127919; OCR Impact:</span> {imp}</div>
                <div class="info-row"><span class="info-label">&#9989; Ideal Range:</span> {ideal_clean}</div>"""

            # Unique checkbox ID for each card (pure CSS toggle)
            chk_id = f"chk_{key}"

            with cols[i % 4]:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">{display}</div>
                  <div class="value" style="color:{col};">{sc}</div>
                  <span class="badge {bcls}">{s}</span>
                  <div class="short-desc">{short_desc}</div>
                  <div class="weight-src">Weight: {int(WEIGHTS[key]*100)}% &nbsp;{src_tag}</div>

                  <input type="checkbox" class="toggle-check" id="{chk_id}">
                  <div class="card-info-panel">
                    <div class="info-title">📐 {FACTOR_INFO[key]['display_name'] if key in FACTOR_INFO else display}</div>
                    {info_html}
                  </div>
                  <label class="toggle-arrow" for="{chk_id}">∨</label>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabs ─────────────────────────────────
        tab1, tab2, tab3 = st.tabs([
            "🔬 Factor Details",
            "📝 OCR Text",
            "💡 Recommendations",
        ])

        with tab1:
            st.markdown("Click any factor to see its detailed explanation:")
            for key, display in DISPLAY_NAMES.items():
                r      = final_results.get(key, {})
                sc     = r.get("score", 0)
                st_txt = r.get("status","—")
                src    = api_status.get(key,"local") if api_status else "local"
                col    = score_color(float(sc))
                # Get rich description based on score range
                rich_desc = get_factor_description(key, float(sc))
                with st.expander(f"**{display}** — {sc}/100   ({st_txt})"):
                    ca, cb = st.columns([3,1])
                    with ca:
                        # Rich description box
                        st.markdown(
                            f'<div class="desc-box">{rich_desc}</div>',
                            unsafe_allow_html=True)
                        if r.get("raw_value") is not None:
                            st.caption(f"📊 Raw value: {r['raw_value']}  {r.get('unit','')}")
                        st.caption(f"🔌 Source: {src}")
                    with cb:
                        st.markdown(f"<div style='text-align:center;font-size:36px;font-weight:700;color:{col};'>{sc}</div>", unsafe_allow_html=True)
                        st.progress(int(min(sc, 100)))
                        st.markdown(f"<div style='text-align:center;font-size:12px;color:#6B7280;'>out of 100</div>", unsafe_allow_html=True)

        with tab2:
            if ocr_conf is not None and ocr_text:
                st.markdown(f"**Tesseract Confidence: {ocr_conf}%**")
                st.text_area("Extracted Text", ocr_text, height=220)
            else:
                st.info(
                    "Tesseract OCR is not available or could not extract text.\n\n"
                    "Make sure Tesseract is installed and the path is set correctly in app.py."
                )

        with tab3:
            for rec in recs:
                is_warn = "🔧" in rec
                box_cls = "rec-box warn" if is_warn else "rec-box"
                clean = rec.replace("**","<b>",1).replace("**","</b>",1)
                st.markdown(f'<div class="{box_cls}">{clean}</div>',
                            unsafe_allow_html=True)

        # ── PDF Export ───────────────────────────
        st.markdown("---")
        st.markdown("#### Export Report")
        pdf_bytes = generate_pdf_report(
            image_name, final_results, ocr_readiness, ocr_conf, recs
        )
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"ocr_report_{image_name.rsplit('.',1)[0]}.pdf",
            mime="application/pdf",
            width="stretch",
        )


# ════════════════════════════════════════════════
# ════════════════════════════════════════════════
# PAGE 2 — History
# ════════════════════════════════════════════════
elif "📊 History" in nav:

    st.markdown("""
    <div class="top-banner">
      <h1>📊 Analysis History</h1>
      <p>View all previous OCR analyses</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_results()

    if df is None or df.empty:
        st.info("No analysis history available.")
        st.stop()

    # -----------------------------------------
    # Keep only images that actually exist
    # -----------------------------------------

    uploads_folder = os.path.join(
        os.path.dirname(__file__),
        "uploads"
    )

    df = df[
        df["image_name"].apply(
            lambda x: os.path.exists(
                os.path.join(uploads_folder, str(x))
            )
        )
    ]

    if df.empty:
        st.warning("No valid images found in uploads folder.")
        st.stop()

    # -----------------------------------------
    # Summary
    # -----------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Total Analyses",
            len(df)
        )

    with c2:
        st.metric(
            "Average OCR Score",
            round(df["ocr_readiness_score"].mean(), 1)
        )

    with c3:
        st.metric(
            "Best OCR Score",
            round(df["ocr_readiness_score"].max(), 1)
        )

    st.divider()

    # -----------------------------------------
    # Time Sorting
    # -----------------------------------------

    left, right = st.columns([3,2])

    with left:

        search = st.text_input(
            "🔍 Search Image",
            placeholder="Enter image name..."
        )

    with right:

        order = st.radio(
            "Time",
            [
                "Descending",
                "Ascending"
            ],
            horizontal=True
        )

    ascending = order == "Ascending"

    if search.strip():

        df = df[
            df["image_name"]
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(
        "timestamp",
        ascending=ascending
    ).reset_index(drop=True)
    st.caption(f"Showing {len(df)} analyses")

    st.divider()

    # ==========================================
    # Display Table
    # ==========================================

    display_df = df.copy()

    display_df.columns = [
        c.upper().replace("_", " ")
        for c in display_df.columns
    ]

    def color_score(value):

        try:

            value = float(value)

            if value >= 81:
                return "background-color:#D1FAE5;color:#065F46;font-weight:bold;"

            elif value >= 61:
                return "background-color:#DBEAFE;color:#1E40AF;font-weight:bold;"

            elif value >= 41:
                return "background-color:#FEF3C7;color:#92400E;font-weight:bold;"

            else:
                return "background-color:#FEE2E2;color:#991B1B;font-weight:bold;"

        except:
            return ""

    if "OCR READINESS SCORE" in display_df.columns:

        styled_df = display_df.style.map(
            color_score,
            subset=["OCR READINESS SCORE"]
        )

    else:

        styled_df = display_df.style

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇ Download Results CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="results.csv",
        mime="text/csv"
    )

    st.divider()

    # ==========================================
    # Image Preview
    # ==========================================

    st.subheader("🖼 Image Preview")

    image_list = df["image_name"].tolist()

    selected_image = st.selectbox(
        "Select Image",
        image_list
    )

    image_path = os.path.join(
        uploads_folder,
        selected_image
    )

    if os.path.exists(image_path):

        preview = Image.open(image_path)

        row = df[
            df["image_name"] == selected_image
        ].iloc[-1]

        left, right = st.columns([2,1])

        with left:

            st.image(
                preview,
                caption=selected_image,
                use_container_width=True
            )

        with right:

            score = float(row["ocr_readiness_score"])

            st.markdown(f"""
<div style="
    background:#1E2A52;
    border-radius:18px;
    padding:22px;
    text-align:center;
    margin-bottom:20px;
">

<div style="
    color:white;
    font-size:20px;
    font-weight:600;
">
    OCR Readiness
</div>

<div style="
    color:#5B8CFF;
    font-size:58px;
    font-weight:700;
    margin-top:10px;
">
    {score:.1f}
</div>

</div>
""", unsafe_allow_html=True)

            factor_columns = [
                c for c in df.columns
                if c.endswith("_score")
                and c != "ocr_readiness_score"
            ]

            for factor in factor_columns:

                try:

                    value = float(row[factor])

                    st.progress(value / 100)

                    st.caption(
                        factor.replace("_score","")
                              .replace("_"," ")
                              .title()
                        + f" : {value}"
                    )

                except:
                    pass

            if "ocr_confidence" in row:

                st.markdown(f"""
                <div style="
                    background:#1E2A52;
                    border-radius:18px;
                    padding:22px;
                    text-align:center;
                    margin-top:15px;
                    margin-bottom:15px;
                ">

                <div style="
                    color:white;
                    font-size:20px;
                    font-weight:600;
                ">
                    OCR Confidence
                </div>

                <div style="
                    color:#5B8CFF;
                    font-size:58px;
                    font-weight:700;
                    margin-top:10px;
                ">
                    {float(row["ocr_confidence"]):.1f}
                </div>

                </div>
                """, unsafe_allow_html=True)

    else:

        st.error(
            "Image not found inside uploads folder."
        )


# ════════════════════════════════════════════════
# PAGE 3 — About Factors
# ════════════════════════════════════════════════
elif "📖 About" in nav:

    st.markdown("""
    <div class="top-banner">
      <h1>📖 About the 10 Quality Factors</h1>
      <p>Definition · Importance · Formula · OCR Impact · Ideal Range</p>
    </div>""", unsafe_allow_html=True)

    selected = st.selectbox(
        "Select a Quality Factor",
        options=list(FACTOR_INFO.keys()),
        format_func=lambda k: FACTOR_INFO[k]["display_name"],
    )
    info = FACTOR_INFO[selected]

    # ── Main info card using native Streamlit ──
    with st.container(border=True):
        st.markdown(f"## 📐 {info['display_name']}")
        st.markdown(
            f'<span style="display:inline-block;background:#EFF6FF;color:#1D4ED8;'
            f'font-size:12px;font-weight:600;padding:3px 10px;border-radius:20px;'
            f'margin-bottom:12px;">👤 Assigned to: {info["owner"]}</span>',
            unsafe_allow_html=True)
        st.markdown(f"**{info['definition']}**")

    st.markdown("")

    # ── 2-column detail section ──
    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown("### 📌 Why This Factor Matters")
            st.markdown(info["importance"])

        with st.container(border=True):
            st.markdown("### 🎯 Effect on OCR")
            st.markdown(info["ocr_impact"])

    with c2:
        with st.container(border=True):
            st.markdown("### 🧮 Calculation Method")
            st.code(info["formula"], language=None)

        with st.container(border=True):
            st.markdown("### ✅ Ideal Score Range")
            st.success(info["ideal_range"])

    st.markdown("---")
    st.markdown("#### All 10 Factors at a Glance")
    rows = [{
        "Factor":      v["display_name"],
        "Owner":       v["owner"],
        "Weight":      f"{int(WEIGHTS[k]*100)}%",
        "Ideal Range": v["ideal_range"].split(".")[0],
    } for k,v in FACTOR_INFO.items()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)