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
from storage import save_result, load_results, compute_correlations
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
    text-align:center;height:100%;margin-bottom:8px;position:relative;}
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
.info-btn{position:absolute;top:8px;right:10px;background:none;border:none;
    cursor:pointer;font-size:15px;color:#9CA3AF;padding:2px 5px;
    border-radius:50%;transition:background 0.2s;}
.info-btn:hover{background:#F3F4F6;color:#374151;}
.card-info-panel{background:#EFF6FF;border-left:3px solid #3B82F6;
    border-radius:0 0 10px 10px;padding:10px 12px;margin-top:10px;
    text-align:left;font-size:12px;color:#1E3A5F;line-height:1.6;}
.card-info-panel .info-title{font-weight:700;font-size:13px;
    color:#1A2B4A;margin-bottom:6px;}
.card-info-panel .info-row{margin-bottom:4px;}
.card-info-panel .info-label{font-weight:600;color:#3B82F6;}

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
    st.markdown("## 🔍 OCR Readiness")
    st.markdown("**SNLP Department**")
    st.markdown("---")

    nav = st.radio("Navigation", [
        "🏠 Analyse Image",
        "📊 History & Correlation",
        "📖 About Factors",
        "🔌 API Status",
        "⚙️ Settings",
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

        save_result(image_name, final_results, ocr_readiness, ocr_conf)

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

            # Info toggle button — ℹ️ in top-right of card
            info_key   = f"info_{key}"
            is_open    = st.session_state.card_info_open.get(key, False)

            with cols[i % 4]:
                # Toggle button ABOVE the card HTML (Streamlit button)
                btn_label = "✕ Close" if is_open else "ℹ️"
                if st.button(btn_label, key=info_key):
                    st.session_state.card_info_open[key] = not is_open
                    st.rerun()

                # Build info panel HTML if open
                info_html = ""
                if is_open and key in FACTOR_INFO:
                    fi = FACTOR_INFO[key]
                    ideal = fi["ideal_range"].split(".")[0]
                    info_html = f"""
                    <div class="card-info-panel">
                      <div class="info-title">📐 {fi['display_name']}</div>
                      <div class="info-row"><span class="info-label">Owner:</span> {fi['owner']}</div>
                      <div class="info-row"><span class="info-label">Definition:</span> {fi['definition'][:120]}…</div>
                      <div class="info-row"><span class="info-label">OCR Impact:</span> {fi['ocr_impact'][:120]}…</div>
                      <div class="info-row"><span class="info-label">Ideal Range:</span> {ideal}</div>
                    </div>"""

                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">{display}</div>
                  <div class="value" style="color:{col};">{sc}</div>
                  <span class="badge {bcls}">{s}</span>
                  <div class="short-desc">{short_desc}</div>
                  <div class="weight-src">Weight: {int(WEIGHTS[key]*100)}% &nbsp;{src_tag}</div>
                  {info_html}
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
# PAGE 2 — History & Correlation
# ════════════════════════════════════════════════
elif "📊 History" in nav:

    st.markdown("""
    <div class="top-banner">
      <h1>📊 History & Correlation Analysis</h1>
      <p>All past analyses · CSV download · Factor vs OCR accuracy correlation</p>
    </div>""", unsafe_allow_html=True)

    df = load_results()
    if df is None:
        st.info("No results yet. Analyse some images first!")
        st.stop()

    st.markdown(f"**{len(df)} analyses stored in results.csv**")
    st.dataframe(df, width="stretch")

    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV", csv_bytes, "results.csv", "text/csv")

    st.markdown("---")
    st.markdown("### Factor ↔ OCR Confidence Correlation")

    corr = compute_correlations()
    if corr is None:
        st.info("Need at least 3 analyses with Tesseract data to compute correlations.")
    else:
        fig = go.Figure(go.Bar(
            x=corr.values,
            y=[DISPLAY_NAMES.get(k,k) for k in corr.index],
            orientation="h",
            marker=dict(color=[score_color(abs(v)*100) for v in corr.values]),
        ))
        fig.update_layout(
            title="Pearson Correlation with OCR Confidence",
            xaxis=dict(range=[-1,1], title="Correlation Coefficient"),
            yaxis=dict(autorange="reversed"),
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("Closer to +1.0 means the factor is a stronger positive predictor of OCR accuracy.")

        corr_df = pd.DataFrame({
            "Factor":      [DISPLAY_NAMES.get(k,k) for k in corr.index],
            "Correlation": [round(v,3) for v in corr.values],
            "Strength":    ["Strong" if abs(v)>0.7 else "Moderate" if abs(v)>0.4 else "Weak"
                            for v in corr.values],
        })
        st.dataframe(corr_df, width="stretch", hide_index=True)


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
        "Select a factor to explore",
        options=list(FACTOR_INFO.keys()),
        format_func=lambda k: FACTOR_INFO[k]["display_name"],
    )
    info = FACTOR_INFO[selected]

    st.markdown(f"""
    <div class="info-card">
      <h4>📐 {info['display_name']}</h4>
      <span class="owner-tag">👤 Assigned to: {info['owner']}</span>
      <p style="color:#374151;margin:0;">{info['definition']}</p>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Why it matters**")
        st.markdown(info["importance"])
        st.markdown("**Effect on OCR**")
        st.markdown(info["ocr_impact"])
    with c2:
        st.markdown("**Calculation Method**")
        st.code(info["formula"], language=None)
        st.markdown("**Ideal Score Range**")
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


# ════════════════════════════════════════════════
# PAGE 4 — API Status
# ════════════════════════════════════════════════
elif "🔌 API Status" in nav:

    st.markdown("""
    <div class="top-banner">
      <h1>🔌 Team API Status</h1>
      <p>Configuration · Live ping · Expected request & response formats</p>
    </div>""", unsafe_allow_html=True)

    urls = get_current_urls()
    st.markdown("### API Configuration")
    api_map = pd.DataFrame([
        {"Member":"Vivek",   "Factors":"Stroke Width, Text Density",
         "URL":urls["vivek"],   "Method":"POST", "Field":"file", "Port":"8001"},
        {"Member":"Mansi",   "Factors":"Blur, Contrast",
         "URL":urls["mansi"],   "Method":"POST", "Field":"file", "Port":"8000"},
        {"Member":"Krish",   "Factors":"Matra Continuity, Zone Integrity",
         "URL":urls["krish"],   "Method":"POST", "Field":"file", "Port":"8002"},
        {"Member":"Tanusha", "Factors":"CC Stability, Skew Penalty",
         "URL":urls["tanusha"], "Method":"POST", "Field":"file", "Port":"9001"},
        {"Member":"Yash",    "Factors":"Noise, Resolution",
         "URL":"Local (built-in)", "Method":"—", "Field":"—", "Port":"—"},
    ])
    st.dataframe(api_map, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("### Live Connectivity Check")
    if st.button("🔄 Ping All APIs Now"):
        import requests as req
        tests = [
            ("Vivek",   urls["vivek"]),
            ("Mansi",   urls["mansi"]),
            ("Krish",   urls["krish"]),
            ("Tanusha", urls["tanusha"]),
        ]
        for name, url in tests:
            # Ping the actual API endpoint directly using HEAD/GET
            # Do NOT strip the port — keep full URL including port number
            try:
                r = req.get(url, timeout=5)
                st.success(f"**{name}** ({url}): Server reachable ✅")
            except req.exceptions.ConnectionError:
                st.error(f"**{name}** ({url}): Unreachable ❌ — Server not running")
            except req.exceptions.Timeout:
                st.error(f"**{name}** ({url}): Unreachable ❌ — Connection timed out")
            except Exception as e:
                # 405/422 means server IS running but endpoint needs POST not GET — that's fine!
                if hasattr(e, 'response') and e.response is not None:
                    st.success(f"**{name}** ({url}): Server reachable ✅")
                else:
                    st.error(f"**{name}** ({url}): Unreachable ❌ — {type(e).__name__}")

    st.markdown("---")
    st.markdown("### Expected Response Formats")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Vivek** `POST /analyze-image`")
        st.code('{\n  "filename": "image.png",\n  "stroke_width": 72.5,\n  "text_density": 85.1\n}', language="json")
    with c2:
        st.markdown("**Mansi** `POST /analyze`")
        st.code('{\n  "blur_score": 88.0,\n  "contrast_score": 74.3\n}', language="json")
    with c3:
        st.markdown("**Krish** `POST /scores`")
        st.code('{\n  "matra_continuity_score": 91.2,\n  "zone_integrity_score": 78.5\n}', language="json")
    with c4:
        st.markdown("**Tanusha** `POST /analyze`")
        st.code('{\n  "connected_component_stability_score": 84.0,\n  "skew_penalty_score": 91.5\n}', language="json")

    st.markdown("---")
    st.info("ℹ️ If any API is offline during analysis, the platform automatically falls back to the local algorithm for that factor.")

# ════════════════════════════════════════════════
# PAGE 5 — Settings
# ════════════════════════════════════════════════
elif "⚙️ Settings" in nav:

    st.markdown("""
    <div class="top-banner">
      <h1>⚙️ Settings — Team IP Addresses</h1>
      <p>Update your teammates' IP addresses here — no code editing needed ever again</p>
    </div>""", unsafe_allow_html=True)

    cfg = load_config()

    st.markdown("### How to find a teammate's IP address")
    st.code("ipconfig        # Windows — look for IPv4 Address\nifconfig        # Mac / Linux", language="bash")
    st.info("📌 Ask each teammate to run the command above and send you their **IPv4 Address** (looks like 192.168.x.x)")

    st.markdown("---")
    st.markdown("### Enter IP Addresses")
    st.markdown("Leave as `127.0.0.1` if that person is running on **your laptop**.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🟢 Vivek** — Port 8001 — Stroke Width & Text Density")
        vivek_ip = st.text_input("Vivek's IP Address", value=cfg["vivek_ip"],
                                  placeholder="e.g. 192.168.1.105")

        st.markdown("**🟡 Mansi** — Port 8000 — Blur & Contrast")
        mansi_ip = st.text_input("Mansi's IP Address", value=cfg["mansi_ip"],
                                  placeholder="e.g. 192.168.1.108")

    with col2:
        st.markdown("**🔵 Krish** — Port 8002 — Matra & Zone Integrity")
        krish_ip = st.text_input("Krish's IP Address", value=cfg["krish_ip"],
                                  placeholder="e.g. 192.168.1.112")

        st.markdown("**🩷 Tanusha** — Port 9001 — CC Stability & Skew")
        tanusha_ip = st.text_input("Tanusha's IP Address", value=cfg["tanusha_ip"],
                                    placeholder="e.g. 192.168.1.115")

    st.markdown("---")
    st.markdown("**Preview — URLs that will be used after saving:**")
    preview = build_urls({
        "vivek_ip": vivek_ip, "mansi_ip": mansi_ip,
        "krish_ip": krish_ip, "tanusha_ip": tanusha_ip
    })
    for name, url in preview.items():
        st.code(f"{name.capitalize()}: {url}")

    if st.button("💾 Save IP Addresses", type="primary", width="stretch"):
        new_cfg = {
            "vivek_ip":   vivek_ip.strip(),
            "mansi_ip":   mansi_ip.strip(),
            "krish_ip":   krish_ip.strip(),
            "tanusha_ip": tanusha_ip.strip(),
        }
        save_config(new_cfg)
        st.success("✅ Saved! The app will now use these IPs automatically — no restart needed.")

    st.markdown("---")
    st.markdown("### Test Connections after Saving")
    if st.button("🔄 Test All Connections Now", width="stretch"):
        import requests as req
        saved_urls = get_current_urls()
        for name, url in saved_urls.items():
            try:
                r = req.get(url, timeout=5)
                st.success(f"**{name.capitalize()}** ({url}): ✅ Reachable")
            except req.exceptions.ConnectionError:
                st.error(f"**{name.capitalize()}** ({url}): ❌ Server not running")
            except req.exceptions.Timeout:
                st.error(f"**{name.capitalize()}** ({url}): ❌ Timed out")
            except Exception as e:
                if hasattr(e, 'response') and e.response is not None:
                    st.success(f"**{name.capitalize()}** ({url}): ✅ Reachable")
                else:
                    st.error(f"**{name.capitalize()}** ({url}): ❌ {type(e).__name__}")
