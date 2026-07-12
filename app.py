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

    st.markdown("### Navigation")

    nav = st.radio(
        "",
        [
            "🏠 Analyse Image",
            "📊 History"
        ],
        label_visibility="collapsed"
    )

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

    st.markdown("---")
    st.markdown("### Information")

    if st.button("📖 About Factors", use_container_width=True):
        nav = "📖 About Factors"

    if st.button("👥 About Team", use_container_width=True):
        nav = "👥 About Team"


# ════════════════════════════════════════════════
# PAGE 1 — Analyse Image
# ════════════════════════════════════════════════
if "🏠 Analyse Image" in nav:

        st.markdown("""
        <div class="top-banner">
            <h1>🔍 OCR Readiness Evaluation Platform</h1>
            <p>
            Upload a document image • Analyse 10 Quality Factors •
            Predict OCR Readiness • Validate with Tesseract •
            Export Professional PDF Report
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ---------------------------------------------------
        # Upload Section
        # ---------------------------------------------------

        uploaded = st.file_uploader(
            "📂 Upload Document Image",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"]
        )

        # ---------------------------------------------------
        # Welcome Screen
        # ---------------------------------------------------

        if uploaded is None and st.session_state.raw_pil is None:

            st.markdown("""
            <div style="
                background:linear-gradient(135deg,#1A2B4A,#0F3460);
                border-radius:20px;
                padding:45px;
                margin-top:20px;
                text-align:center;
                border:1px solid #304878;
            ">

            <h2 style="
                color:#5B8CFF;
                margin-bottom:20px;
                font-size:30px;
            ">
                👋 Welcome
            </h2>

            <p style="
                color:white;
                font-size:18px;
                line-height:1.8;
            ">

            This platform evaluates whether a document image is suitable
            for Optical Character Recognition (OCR).

            <br><br>

            It automatically analyses

            <br>

            ✅ Noise

            <br>

            ✅ Resolution

            <br>

            ✅ Blur

            <br>

            ✅ Contrast

            <br>

            ✅ Text Density

            <br>

            ✅ Stroke Width

            <br>

            ✅ Matra Continuity

            <br>

            ✅ Zone Integrity

            <br>

            ✅ Connected Components

            <br>

            ✅ Skew Detection

            <br><br>

            After analysis, you'll receive

            <br><br>

            📊 OCR Readiness Score

            <br>

            📄 OCR Confidence

            <br>

            💡 Improvement Suggestions

            <br>

            📥 Downloadable PDF Report

            <br><br>

            <span style="
                color:#5B8CFF;
                font-size:20px;
                font-weight:bold;
            ">
            ⬆ Upload an image above to begin.
            </span>

            </p>

            </div>
            """, unsafe_allow_html=True)

            st.stop()

        # ---------------------------------------------------
        # Store Uploaded Image
        # ---------------------------------------------------

        if uploaded is not None:

            raw_pil = Image.open(uploaded).convert("RGB")

            if uploaded.name != st.session_state.image_name:

                st.session_state.raw_pil = raw_pil
                st.session_state.image_name = uploaded.name
                st.session_state.analysis_done = False
                st.session_state.final_results = {}
                st.session_state.analysis_img = raw_pil

        raw_pil = st.session_state.raw_pil
        image_name = st.session_state.image_name

        # ======================================================
        # STEP 1 — Select Analysis Region
        # ======================================================

        st.markdown("---")
        st.markdown("## ✂️ Step 1 • Select Image Region")

        st.info(
            "You may analyse the complete image or crop only the required document area."
        )

        use_crop = st.toggle(
            "Crop Image Before Analysis",
            value=False
        )

        if use_crop:

            if CROPPER_OK:

                st.markdown(
                    "Drag the handles below to select the portion you want to analyse."
                )

                cropped = st_cropper(
                    raw_pil,
                    realtime_update=True,
                    box_color="#00C4B4",
                    aspect_ratio=None
                )

                left, right = st.columns([3,1])

                with left:

                    st.image(
                        cropped,
                        caption="Selected Region",
                        use_container_width=True
                    )

                with right:

                    width, height = cropped.size

                    st.metric(
                        "Width",
                        f"{width}px"
                    )

                    st.metric(
                        "Height",
                        f"{height}px"
                    )

                    st.success("Ready for Analysis")

                st.session_state.analysis_img = cropped

            else:

                st.warning(
                    "Cropper package not installed. Using manual crop sliders."
                )

                img_w, img_h = raw_pil.size

                c1, c2 = st.columns(2)

                with c1:

                    st.image(
                        raw_pil,
                        caption="Original Image",
                        use_container_width=True
                    )

                with c2:

                    left = st.slider(
                        "Left",
                        0,
                        img_w-1,
                        0
                    )

                    top = st.slider(
                        "Top",
                        0,
                        img_h-1,
                        0
                    )

                    right = st.slider(
                        "Right",
                        1,
                        img_w,
                        img_w
                    )

                    bottom = st.slider(
                        "Bottom",
                        1,
                        img_h,
                        img_h
                    )

                    if right <= left:
                        right = left + 1

                    if bottom <= top:
                        bottom = top + 1

                    cropped = raw_pil.crop(
                        (
                            left,
                            top,
                            right,
                            bottom
                        )
                    )

                    st.image(
                        cropped,
                        caption="Cropped Image",
                        use_container_width=True
                    )

                st.session_state.analysis_img = cropped

        else:

            st.image(
                raw_pil,
                caption="Original Image (Full Image Will Be Analysed)",
                use_container_width=True
            )

            st.session_state.analysis_img = raw_pil

        analysis_img = st.session_state.analysis_img

        temp_image_path = os.path.join(
            os.path.dirname(__file__),
            "uploads",
            st.session_state.image_name
        )

        # ======================================================
        # STEP 2 — Run OCR Readiness Analysis
        # ======================================================

        st.markdown("---")
        st.markdown("## 🚀 Step 2 • Analyse Image")

        with st.container(border=True):

            st.subheader("🚀 Ready to Analyse")

            st.write(
                "Click the **Analyse Image** button below to begin the OCR Readiness Evaluation."
            )

            st.markdown("""
        ### The platform will automatically

        - ✅ Analyse all **10 OCR Quality Factors**
        - ✅ Calculate the **OCR Readiness Score**
        - ✅ Estimate **OCR Confidence**
        - ✅ Generate **Personalized Recommendations**
        - ✅ Save this analysis into **History**
        - ✅ Generate a **Professional PDF Report**

        ---
        Once the analysis is complete, detailed factor scores, OCR text, and improvement suggestions will be displayed automatically.
        """)

        analyse = st.button(
            "🚀 Analyse Image",
            use_container_width=True,
            type="primary"
        )

        if analyse:

            with st.spinner("Preparing image..."):

                bgr = pil_to_bgr(analysis_img)

            with st.spinner("Calculating OCR Quality Factors..."):

                local_results = run_all_factors(bgr)

            with st.spinner("Connecting with Team APIs..."):

                final_results, api_status = call_all_team_apis(
                    bgr,
                    local_results
                )

            with st.spinner("Running Tesseract OCR..."):

                ocr_conf, ocr_text = run_tesseract(
                    analysis_img
                )

            with st.spinner("Generating Recommendations..."):

                recs = generate_recommendations(
                    final_results
                )

            ocr_readiness = final_results["ocr_readiness_score"]

            # -----------------------------------
            # Save Image
            # -----------------------------------

            save_uploaded_image(
                analysis_img,
                image_name
            )

            temp_image_path = os.path.join(
                os.path.dirname(__file__),
                "uploads",
                image_name
            )

            # -----------------------------------
            # Save CSV Result
            # -----------------------------------

            save_result(
                image_name=image_name,
                factor_results=final_results,
                ocr_readiness=ocr_readiness,
                ocr_confidence=ocr_conf
            )

            # -----------------------------------
            # Save Session
            # -----------------------------------

            st.session_state.analysis_done = True

            st.session_state.final_results = final_results

            st.session_state.api_status = api_status

            st.session_state.ocr_conf = ocr_conf

            st.session_state.ocr_text = ocr_text if ocr_text else ""

            st.session_state.recs = recs

            st.success("✅ Analysis completed successfully.")

            # ======================================================
        # RESULTS
        # ======================================================

        if st.session_state.analysis_done:

            final_results = st.session_state.final_results
            api_status = st.session_state.api_status
            ocr_conf = st.session_state.ocr_conf
            ocr_text = st.session_state.ocr_text
            recs = st.session_state.recs

            ocr_readiness = final_results["ocr_readiness_score"]
            ocr_status = final_results["ocr_readiness_status"]

            st.markdown("---")

            st.markdown("""
            <h2 style="
                text-align:center;
                color:#1A2B4A;
                margin-bottom:25px;
            ">
            📊 Analysis Results
            </h2>
            """, unsafe_allow_html=True)

            # ==================================================
            # API Summary
            # ==================================================

            with st.expander("🔌 Team API Status"):

                for key, src in api_status.items():

                    disp = DISPLAY_NAMES.get(key, key)

                    if "Yash" in src:
                        css = "api-yash"

                    elif "Vivek" in src:
                        css = "api-vivek"

                    elif "Mansi" in src:
                        css = "api-mansi"

                    elif "Krish" in src:
                        css = "api-krish"

                    elif "Tanusha" in src:
                        css = "api-tanusha"

                    else:
                        css = "api-local"

                    st.markdown(
                        f'<div class="api-row {css}"><b>{disp}</b> : {src}</div>',
                        unsafe_allow_html=True
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # ==================================================
            # SCORE CARDS
            # ==================================================

            c1, c2, c3 = st.columns(3)

            # -----------------------------------------

            with c1:

                clr = score_color(ocr_readiness)

                st.markdown(f"""
                <div style="
                    background:#1A2B4A;
                    border-radius:18px;
                    padding:25px;
                    text-align:center;
                    min-height:240px;
                ">

                <div style="
                    color:white;
                    font-size:18px;
                    font-weight:600;
                ">
                    OCR Readiness Score
                </div>

                <div style="
                    color:{clr};
                    font-size:70px;
                    font-weight:700;
                    margin-top:15px;
                ">
                    {ocr_readiness}
                </div>

                <div style="
                    color:#A7B7D9;
                    font-size:18px;
                ">
                    {ocr_status}
                </div>

                </div>
                """, unsafe_allow_html=True)

            # -----------------------------------------

            with c2:

                if ocr_conf is not None:

                    clr = score_color(ocr_conf)

                    st.markdown(f"""
                    <div style="
                        background:#1A2B4A;
                        border-radius:18px;
                        padding:25px;
                        text-align:center;
                        min-height:240px;
                    ">

                    <div style="
                        color:white;
                        font-size:18px;
                        font-weight:600;
                    ">
                        OCR Confidence
                    </div>

                    <div style="
                        color:{clr};
                        font-size:70px;
                        font-weight:700;
                        margin-top:15px;
                    ">
                        {ocr_conf}
                    </div>

                    <div style="
                        color:#A7B7D9;
                        font-size:18px;
                    ">
                        Tesseract OCR
                    </div>

                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.info("OCR Confidence unavailable.")

            # -----------------------------------------

            with c3:

                if ocr_conf is not None:

                    diff = round(
                        abs(
                            ocr_readiness - ocr_conf
                        ),
                        1
                    )

                    if diff < 10:
                        colour = "#10B981"
                        txt = "Excellent Prediction"

                    elif diff < 20:
                        colour = "#F59E0B"
                        txt = "Acceptable Prediction"

                    else:
                        colour = "#EF4444"
                        txt = "Needs Improvement"

                    st.markdown(f"""
                    <div style="
                        background:#1A2B4A;
                        border-radius:18px;
                        padding:25px;
                        text-align:center;
                        min-height:240px;
                    ">

                    <div style="
                        color:white;
                        font-size:18px;
                        font-weight:600;
                    ">
                        Prediction Accuracy
                    </div>

                    <div style="
                        color:{colour};
                        font-size:70px;
                        font-weight:700;
                        margin-top:15px;
                    ">
                        {diff}
                    </div>

                    <div style="
                        color:#A7B7D9;
                        font-size:18px;
                    ">
                        {txt}
                    </div>

                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.info("Prediction unavailable.")

            # ======================================================
            # FACTOR SCORES
            # ======================================================

            st.markdown("---")
            st.markdown("""
            <h2 style="
            text-align:center;
            color:#1A2B4A;
            margin-bottom:25px;
            ">
            📈 OCR Quality Factor Scores
            </h2>
            """, unsafe_allow_html=True)

            cols = st.columns(2)

            for i, (key, display) in enumerate(DISPLAY_NAMES.items()):

                result = final_results.get(key, {})

                score = float(result.get("score", 0))

                status = result.get("status", "-")

                colour = score_color(score)

                badge = f"badge-{status.lower()}"

                short_desc = get_short_description(key, score)

                owner = "-"
                definition = "-"
                impact = "-"
                ideal = "-"

                if key in FACTOR_INFO:

                    owner = FACTOR_INFO[key]["owner"]

                    definition = FACTOR_INFO[key]["definition"]

                    impact = FACTOR_INFO[key]["ocr_impact"]

                    ideal = FACTOR_INFO[key]["ideal_range"]

                with cols[i % 2]:

                    with st.container(border=True):

                        top1, top2 = st.columns([3,1])

                        with top1:

                            st.markdown(f"### {display}")

                            st.caption(short_desc)

                        with top2:

                            st.markdown(
                                f"<h2 style='color:{colour};text-align:right;'>{score:.1f}</h2>",
                                unsafe_allow_html=True
                            )

                        st.progress(score/100)

                        st.markdown(
                            f"""
            <span class="badge {badge}">
            {status}
            </span>
            """,
                            unsafe_allow_html=True
                        )

                        st.caption(
                            f"Weight : {int(WEIGHTS[key]*100)}%"
                        )

                        with st.expander("ℹ More Information"):

                            st.markdown(
                                f"""
            **👤 Owner**

            {owner}

            ---

            **📖 Definition**

            {definition}

            ---

            **🎯 OCR Impact**

            {impact}

            ---

            **✅ Ideal Range**

            {ideal}
            """
            )

            # ======================================================
            # ANALYSIS DETAILS
            # ======================================================

            st.markdown("---")

            tab1, tab2, tab3 = st.tabs([
            "📄 OCR Text",
            "💡 Recommendations",
            "📘 Factor Details"
        ])

            # ======================================================
            # TAB 1
            # ======================================================

            with tab1:

                st.subheader("OCR Text Extraction")

                if ocr_conf is not None:

                    st.success(
                        f"OCR Confidence : {ocr_conf:.1f}%"
                    )

                else:

                    st.warning(
                        "OCR Confidence could not be calculated."
                    )

                st.text_area(
                    "Extracted Text",
                    ocr_text,
                    height=350
                )

            # ======================================================
            # TAB 2
            # ======================================================

            with tab2:

                st.subheader("💡 OCR Improvement Recommendations")

                # Remove empty recommendations
                recs = [
                    r.strip()
                    for r in recs
                    if isinstance(r, str) and r.strip()
                ]

                if not recs:

                    st.success("🎉 Excellent Image Quality")

                    st.markdown("""
Your uploaded document already satisfies the recommended OCR quality standards.

### No improvements are required.

You can directly use this image for OCR processing with a high probability of obtaining accurate text extraction.
""")

                else:

                    st.info(
                        f"{len(recs)} recommendation(s) generated for improving OCR accuracy."
                    )

                    for i, rec in enumerate(recs, start=1):

                        st.markdown(f"""
<div style="
background:#F8FAFC;
border-left:6px solid #00C4B4;
padding:18px;
border-radius:12px;
margin-bottom:15px;
">

<h4 style="margin-bottom:10px;color:#1A2B4A;">
Recommendation {i}
</h4>

<p style="
font-size:15px;
line-height:1.8;
color:#374151;
margin-bottom:0px;
">
{rec}
</p>

</div>
""", unsafe_allow_html=True)

                    st.success(
                        "✔ Apply the above suggestions and re-analyse the image for improved OCR performance."
                    )

            # ======================================================
            # TAB 3
            # ======================================================

            with tab3:

                st.subheader("Detailed Factor Information")

                for key, display in DISPLAY_NAMES.items():

                    result = final_results.get(key, {})

                    score = float(
                        result.get("score", 0)
                    )

                    status = result.get(
                        "status",
                        "-"
                    )

                    with st.expander(
                        f"{display}   •   {score:.1f}/100"
                    ):

                        st.progress(score / 100)

                        st.markdown(
                            f"### Status : {status}"
                        )

                        st.markdown(
                            get_factor_description(
                                key,
                                score
                            ),
                            unsafe_allow_html=True
                        )

                        if key in FACTOR_INFO:

                            st.markdown("---")

                            st.markdown(
                                f"**Owner** : {FACTOR_INFO[key]['owner']}"
                            )

                            st.markdown(
                                f"**Ideal Range** : {FACTOR_INFO[key]['ideal_range']}"
                            )

                            st.markdown(
                                f"**Formula**"
                            )

                            st.code(
                                FACTOR_INFO[key]["formula"]
                            )

            # ======================================================
            # EXPORT REPORT
            # ======================================================

            st.markdown("---")

            st.subheader("📄 Export Report")

            left, right = st.columns([3,1])

            with left:

                st.info(
                    """
            Generate a professional PDF report containing

            • OCR Readiness Score

            • OCR Confidence

            • Radar Chart

            • All Factor Scores

            • Recommendations

            • OCR Text

            • Uploaded Image
            """
                )

            with right:

                pdf_bytes = generate_pdf_report(
                    image_name=st.session_state.image_name,
                    factor_results=final_results,
                    ocr_readiness=ocr_readiness,
                    ocr_confidence=ocr_conf,
                    recommendations=recs,
                    image_path=os.path.join(
                        os.path.dirname(__file__),
                        "uploads",
                        st.session_state.image_name
                    ),
                    ocr_text=ocr_text
                )
                

                st.download_button(
                    "⬇ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{image_name.split('.')[0]}_OCR_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            st.success("✅ Analysis Completed Successfully.")


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
                        factor.replace("_score", "")
                              .replace("_", " ")
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
        st.error("Image not found inside uploads folder.")


    # ════════════════════════════════════════════════
    # PAGE 3 — About Factors
    # ════════════════════════════════════════════════
elif nav == "📖 About Factors":

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

# ═══════════════════════════════════════
# PAGE 4 — ABOUT TEAM
# ═══════════════════════════════════════

elif nav == "👥 About Team":

    st.markdown("""
    <div class="top-banner">
        <h1>👥 Project Team</h1>
        <p>OCR Readiness Evaluation Platform • SNLP Department</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Meet Our Team")

    team = [

        {
            "Name":"Yash Rajput",
            "Role":"Team Lead",
            "College":"Symbiosis Skills and Professional University, Pune",
            "Duration of internship":"2 Months",
            "Phone":"9811518972",
            "Photo":"team/yash.jpg"
        },

        {
            "Name":"Vivek",
            "Role":"Backend Developer",
            "College":"KCC Institute of Technology and Management",
            "Duration of internship":"2 Months",
            "Phone":"8527890733",
            "Photo":"team/vivek.jpg"
        },

        {
            "Name":"Mansi",
            "Role":"Backend Developer",
            "College":"Indira Gandhi Delhi Technical University for Women (IGDTUW)",
            "Duration of internship":"6 Weeks",
            "Phone":"9210720688",
            "Photo":"team/mansi.jpg"
        },

        {
            "Name":"Krish",
            "Role":"Backend Developer",
            "College":"Thapar Institute of Engineering and Technology",
            "Duration of internship":"1 Month",
            "Phone":"9417976174",
            "Photo":"team/krish.jpg"
        },

        {
            "Name":"Tanusha",
            "Role":"Backend Developer",
            "College":"Jaypee Institute of Information Technology, Noida",
            "Duration of internship":"2 Months",
            "Phone":"8505938377",
            "Photo":"team/tanusha.jpg"
        }

    ]

    for member in team:

        col1, col2 = st.columns([1,3])

        with col1:

            if os.path.exists(member["photo"]):
                st.image(member["photo"], width=170)
            else:
                st.image("https://placehold.co/170x210")

        with col2:

            st.markdown(f"### {member['name']}")
            st.markdown(f"**Role :** {member['role']}")
            st.markdown(f"**College :** {member['college']}")
            st.markdown(f"**Internship Duration :** {member['duration']}")
            st.markdown(f"**Contact :** {member['phone']}")

        st.divider()

