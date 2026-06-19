"""
OCR Readiness Evaluation Platform
SNLP Department  Team: Yash (Lead), Vivek, Mansi, Krish, Tanusha
Run: streamlit run app.py
"""

import sys, os
import shutil
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
    def _find_tesseract_cmd():
        env_cmd = os.environ.get("TESSERACT_CMD")
        if env_cmd and os.path.isfile(env_cmd):
            return env_cmd

        path_cmd = shutil.which("tesseract")
        if path_cmd:
            return path_cmd

        program_files = [
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("LocalAppData"),
        ]
        candidates = [
            r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
            r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
            r"C:\\Users\\Yash Rajput\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe",
        ]
        for base_dir in program_files:
            if base_dir:
                candidates.append(os.path.join(base_dir, "Tesseract-OCR", "tesseract.exe"))
                candidates.append(os.path.join(base_dir, "Programs", "Tesseract-OCR", "tesseract.exe"))

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return None

    tesseract_cmd = _find_tesseract_cmd()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        TESSERACT_OK = True
    else:
        TESSERACT_OK = False
except ImportError:
    TESSERACT_OK = False

try:
    from streamlit_cropper import st_cropper
    CROPPER_OK = True
except ImportError:
    CROPPER_OK = False

#  Page config 
st.set_page_config(
    page_title="OCR Readiness Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  CSS 
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
    text-align:center;height:100%;margin-bottom:8px;}
.metric-card .label{font-size:11px;font-weight:600;color:#6B7280;
    text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;}
.metric-card .value{font-size:28px;font-weight:700;
    font-family:'Space Grotesk',sans-serif;}
.metric-card .badge{display:inline-block;padding:2px 10px;border-radius:20px;
    font-size:11px;font-weight:600;margin-top:4px;}
.badge-excellent{background:#D1FAE5;color:#065F46;}
.badge-good{background:#DBEAFE;color:#1E40AF;}
.badge-average{background:#FEF3C7;color:#92400E;}
.badge-poor{background:#FEE2E2;color:#991B1B;}
.badge-error{background:#F3F4F6;color:#6B7280;}

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


#  Helpers 
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
        confs = []
        for c in data.get("conf", []):
            try:
                conf = float(c)
            except (TypeError, ValueError):
                continue
            if conf >= 0:
                confs.append(conf)
        if confs:
            return round(float(np.mean(confs)),1), pytesseract.image_to_string(img)
        return None, None
    except Exception:
        return None, None

#  Session state init 
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

#  Sidebar 
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
if nav == "🏠 Analyse Image":

    st.markdown("""
    <div class="top-banner">
      <h1>🔍 OCR Readiness Evaluation Platform</h1>
      <p>Upload → Crop (optional) → Factor Scores → OCR Readiness Score → Tesseract Validation → PDF Report</p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload a document image", type=["png","jpg","jpeg","bmp","tiff","webp"]) 
    if uploaded is not None:
        raw_pil = Image.open(uploaded).convert("RGB")
        if uploaded.name != st.session_state.image_name:
            st.session_state.raw_pil = raw_pil
            st.session_state.image_name = uploaded.name
            st.session_state.analysis_done = False
            st.session_state.final_results = {}
            st.session_state.analysis_img = raw_pil
    elif st.session_state.raw_pil is None:
        st.info("👆 Upload a document image to begin the analysis.")
        st.stop()

    raw_pil = st.session_state.raw_pil
    image_name = st.session_state.image_name

    st.markdown("### Step 1 — Select Region")
    use_crop = st.checkbox("✂️ Crop the image before analysis")
    if use_crop and CROPPER_OK:
        cropped = st_cropper(raw_pil, realtime_update=True, box_color="#00C4B4", aspect_ratio=None)
        st.image(cropped, caption="Selected region", use_column_width=True)
        st.session_state.analysis_img = cropped
    else:
        st.image(raw_pil, caption="Full image — will be analysed", use_column_width=True)
        st.session_state.analysis_img = raw_pil

    analysis_img = st.session_state.analysis_img

    st.markdown("### Step 2 — Run Analysis")
    use_apis = st.checkbox("🔌 Use team APIs (falls back to local if any API is offline)", value=True)

    if st.button("🚀 Analyse Image"):
        bgr = pil_to_bgr(analysis_img)
        with st.spinner("Computing local factors…"):
            local_results = run_all_factors(bgr)

        final_results = local_results
        api_status = {}
        if use_apis:
            with st.spinner("Calling team APIs…"):
                try:
                    final_results, api_status = call_all_team_apis(bgr, local_results)
                except Exception:
                    api_status = {}

        recs = generate_recommendations(final_results)
        ocr_conf, ocr_text = run_tesseract(analysis_img)
        ocr_readiness = final_results.get("ocr_readiness_score", 0)

        save_result(image_name, final_results, ocr_readiness, ocr_conf)

        st.session_state.analysis_done = True
        st.session_state.final_results = final_results
        st.session_state.api_status = api_status
        st.session_state.ocr_conf = ocr_conf
        st.session_state.ocr_text = ocr_text or ""
        st.session_state.recs = recs

    # Show results
    if st.session_state.analysis_done:
        final_results = st.session_state.final_results
        api_status = st.session_state.api_status
        ocr_conf = st.session_state.ocr_conf
        ocr_text = st.session_state.ocr_text
        recs = st.session_state.recs
        ocr_readiness = final_results.get("ocr_readiness_score", 0)

        st.markdown("---")
        st.markdown("### Results (summary)")
        st.metric("OCR Readiness Score", f"{ocr_readiness}/100")
        st.markdown("#### Factor scores")
        st.json(final_results)

        st.markdown("#### Tesseract")
        if ocr_conf is not None:
            st.write(f"Tesseract confidence: {ocr_conf}%")
            st.text_area("Extracted text", ocr_text, height=200)
        else:
            st.info("Tesseract not available or could not extract text.")

        st.markdown("---")
        st.markdown("#### Export Report")
        try:
            pdf_bytes = generate_pdf_report(image_name, final_results, ocr_readiness, ocr_conf, recs)
            st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name=f"ocr_report_{image_name.rsplit('.',1)[0]}.pdf", mime="application/pdf")
        except Exception as e:
            st.warning(f"Could not generate PDF: {e}")


# ════════════════════════════════════════════════
# PAGE 2 — History & Correlation
# ════════════════════════════════════════════════
elif nav == "📊 History & Correlation":
    st.markdown("""
    <div class="top-banner">
      <h1>📊 History & Correlation Analysis</h1>
      <p>All past analyses · CSV download · Factor vs OCR accuracy correlation</p>
    </div>""", unsafe_allow_html=True)

    df = load_results()
    if df is None or df.empty:
        st.info("No results yet. Analyse some images first!")
        st.stop()

    st.write(f"**{len(df)} analyses stored**")
    st.dataframe(df)
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV", csv_bytes, "results.csv", "text/csv")

    st.markdown("---")
    corr = compute_correlations()
    if corr is None:
        st.info("Need at least 3 analyses with Tesseract data to compute correlations.")
    else:
        fig = go.Figure(go.Bar(x=corr.values, y=[DISPLAY_NAMES.get(k,k) for k in corr.index], orientation="h"))
        fig.update_layout(title="Pearson Correlation with OCR Confidence", height=400)
        st.plotly_chart(fig)


# ════════════════════════════════════════════════
# PAGE 3 — About Factors
# ════════════════════════════════════════════════
elif nav == "📖 About Factors":
    st.header("About Factors")
    for key, info in FACTOR_INFO.items():
        st.subheader(info.get("display", key))
        st.write(info.get("description", ""))


# ════════════════════════════════════════════════
# PAGE 4 — API Status
# ════════════════════════════════════════════════
elif nav == "🔌 API Status":
    st.header("API Status")
    try:
        urls = get_current_urls()
        st.json(urls)
    except Exception:
        st.write("Could not fetch current URLs. Known issues:")
        st.write(KNOWN_ISSUES)


# ════════════════════════════════════════════════
# PAGE 5 — Settings
# ════════════════════════════════════════════════
elif nav == "⚙️ Settings":
    st.header("Settings")
    try:
        cfg = load_config()
        st.json(cfg)
    except Exception as e:
        st.write(f"Could not load config: {e}")
