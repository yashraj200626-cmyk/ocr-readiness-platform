"""
Team API Integration Layer
===========================
Mansi   → POST http://localhost:8000/analyze        (blur_score, contrast_score)
Krish   → POST http://localhost:8002/scores         (matra_continuity_score, zone_integrity_score)
Vivek   → POST http://127.0.0.1:8001/analyze-image  (stroke_width, text_density)
Tanusha → POST http://127.0.0.1:9001/analyze        (connected_component_stability_score, skew_penalty_score)
Yash    → Local                                     (noise_score, resolution_score)
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, Optional, Tuple

# ── API URLs ───────────────────────────────────────────────────────────────
MANSI_API_URL   = "http://localhost:8000/analyze"
KRISH_API_URL   = "http://localhost:8002/scores"
VIVEK_API_URL   = "http://127.0.0.1:8001/analyze-image"
TANUSHA_API_URL = "http://127.0.0.1:9001/analyze"

API_TIMEOUT   = 10
FIELD_NAME    = "file"

# ── No known issues — all APIs confirmed ──────────────────────────────────
KNOWN_ISSUES  = []

# ── Helpers ────────────────────────────────────────────────────────────────
def _bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def _pil_to_bytes(img_pil: Image.Image, fmt="PNG") -> Tuple[bytes, str]:
    buf = io.BytesIO()
    img_pil.save(buf, format=fmt)
    buf.seek(0)
    return buf.read(), f"image/{fmt.lower()}"

def _post_image(url: str, img_bgr: np.ndarray) -> Optional[Dict]:
    try:
        img_bytes, mime = _pil_to_bytes(_bgr_to_pil(img_bgr))
        files = {FIELD_NAME: ("image.png", img_bytes, mime)}
        resp  = requests.post(url, files=files, timeout=API_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None

def _normalize(val) -> Optional[float]:
    try:
        return max(0.0, min(100.0, float(val)))
    except (TypeError, ValueError):
        return None

def _classify(score: float) -> str:
    if score >= 81: return "Excellent"
    if score >= 61: return "Good"
    if score >= 41: return "Average"
    return "Poor"

# ── Parsers ────────────────────────────────────────────────────────────────

def _parse_mansi(data: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Mansi response:
    { "blur_score": <0-100>, "contrast_score": <0-100> }
    """
    blur_result = contrast_result = None

    raw = _normalize(data.get("blur_score"))
    if raw is not None:
        blur_result = {
            "factor_name": "blur_score",
            "score":       round(raw, 1),
            "status":      _classify(raw),
            "description": data.get("blur_description",
                           f"Blur score from Mansi's API: {raw:.1f}"),
            "source":      "mansi_api",
            "raw_value":   data.get("blur_score"),
            "unit":        "",
        }

    raw = _normalize(data.get("contrast_score"))
    if raw is not None:
        contrast_result = {
            "factor_name": "contrast_score",
            "score":       round(raw, 1),
            "status":      _classify(raw),
            "description": data.get("contrast_description",
                           f"Contrast score from Mansi's API: {raw:.1f}"),
            "source":      "mansi_api",
            "raw_value":   data.get("contrast_score"),
            "unit":        "",
        }

    return blur_result, contrast_result


def _parse_krish(data: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Krish response:
    { "matra_continuity_score": <0-100>, "zone_integrity_score": <0-100> }
    """
    matra_result = zone_result = None

    raw = _normalize(data.get("matra_continuity_score"))
    if raw is not None:
        matra_result = {
            "factor_name": "matra_continuity_score",
            "score":       round(raw, 1),
            "status":      _classify(raw),
            "description": data.get("matra_description",
                           f"Matra continuity from Krish's API: {raw:.1f}"),
            "source":      "krish_api",
            "raw_value":   data.get("matra_continuity_score"),
            "unit":        "",
        }

    raw = _normalize(data.get("zone_integrity_score"))
    if raw is not None:
        zone_result = {
            "factor_name": "zone_integrity_score",
            "score":       round(raw, 1),
            "status":      _classify(raw),
            "description": data.get("zone_description",
                           f"Zone integrity from Krish's API: {raw:.1f}"),
            "source":      "krish_api",
            "raw_value":   data.get("zone_integrity_score"),
            "unit":        "",
        }

    return matra_result, zone_result


def _parse_vivek(data: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Vivek response:
    {
        "filename":     "image.png",   <- ignored
        "stroke_width": <value>,       <- mapped to stroke_width_score
        "text_density": <value>        <- mapped to text_density_score
    }
    """
    stroke_result = density_result = None

    # Vivek uses "stroke_width" not "stroke_width_score"
    raw = _normalize(data.get("stroke_width"))
    if raw is not None:
        stroke_result = {
            "factor_name": "stroke_width_score",
            "score":       round(raw, 1),
            "status":      _classify(raw),
            "description": f"Stroke width consistency from Vivek's API: {raw:.1f}",
            "source":      "vivek_api",
            "raw_value":   data.get("stroke_width"),
            "unit":        "",
        }

    # Vivek uses "text_density" not "text_density_score"
    raw = _normalize(data.get("text_density"))
    if raw is not None:
        density_result = {
            "factor_name": "text_density_score",
            "score":       round(raw, 1),
            "status":      _classify(raw),
            "description": f"Text density from Vivek's API: {raw:.1f}",
            "source":      "vivek_api",
            "raw_value":   data.get("text_density"),
            "unit":        "",
        }

    return stroke_result, density_result


def _parse_tanusha(data: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Tanusha response (POST /analyze):
    {
        "connected_component_stability_score": <0-100>,
        "skew_penalty_score": <0-100>
    }
    """
    ccs_result = skew_result = None

    raw = _normalize(data.get("connected_component_stability_score"))
    if raw is not None:
        ccs_result = {
            "factor_name": "connected_component_stability_score",
            "score":       round(raw, 1),
            "status":      _classify(raw),
            "description": data.get("ccs_description",
                           f"Connected component stability from Tanusha's API: {raw:.1f}"),
            "source":      "tanusha_api",
            "raw_value":   data.get("connected_component_stability_score"),
            "unit":        "",
        }

    raw = _normalize(data.get("skew_penalty_score"))
    if raw is not None:
        skew_result = {
            "factor_name": "skew_penalty_score",
            "score":       round(raw, 1),
            "status":      _classify(raw),
            "description": data.get("skew_description",
                           f"Skew penalty from Tanusha's API: {raw:.1f}"),
            "source":      "tanusha_api",
            "raw_value":   data.get("skew_penalty_score"),
            "unit":        "",
        }

    return ccs_result, skew_result


# ── Main orchestrator ──────────────────────────────────────────────────────

def call_all_team_apis(
    img_bgr: np.ndarray,
    local_results: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Calls all 4 team APIs. Falls back to local computation if any API
    is unreachable or returns unexpected data.

    Returns:
        merged_results  — all 10 factor results (API or local fallback)
        api_status      — per-factor source label shown in the dashboard
    """
    merged = dict(local_results)
    status = {}

    # ── Mansi — blur + contrast ───────────────────
    data = _post_image(MANSI_API_URL, img_bgr)
    if data is not None:
        blur_r, contrast_r = _parse_mansi(data)
        if blur_r is not None:
            merged["blur_score"]     = blur_r
            status["blur_score"]     = "api ✅ Mansi"
        else:
            status["blur_score"]     = "local ⚡ (Mansi returned unexpected format)"
        if contrast_r is not None:
            merged["contrast_score"] = contrast_r
            status["contrast_score"] = "api ✅ Mansi"
        else:
            status["contrast_score"] = "local ⚡ (Mansi returned unexpected format)"
    else:
        status["blur_score"]     = "local ⚡ (Mansi API offline)"
        status["contrast_score"] = "local ⚡ (Mansi API offline)"

    # ── Krish — matra + zone ──────────────────────
    data = _post_image(KRISH_API_URL, img_bgr)
    if data is not None:
        matra_r, zone_r = _parse_krish(data)
        if matra_r is not None:
            merged["matra_continuity_score"] = matra_r
            status["matra_continuity_score"] = "api ✅ Krish"
        else:
            status["matra_continuity_score"] = "local ⚡ (Krish returned unexpected format)"
        if zone_r is not None:
            merged["zone_integrity_score"]   = zone_r
            status["zone_integrity_score"]   = "api ✅ Krish"
        else:
            status["zone_integrity_score"]   = "local ⚡ (Krish returned unexpected format)"
    else:
        status["matra_continuity_score"] = "local ⚡ (Krish API offline)"
        status["zone_integrity_score"]   = "local ⚡ (Krish API offline)"

    # ── Vivek — stroke_width + text_density ───────
    data = _post_image(VIVEK_API_URL, img_bgr)
    if data is not None:
        stroke_r, density_r = _parse_vivek(data)
        if stroke_r is not None:
            merged["stroke_width_score"] = stroke_r
            status["stroke_width_score"] = "api ✅ Vivek"
        else:
            status["stroke_width_score"] = "local ⚡ (Vivek returned unexpected format)"
        if density_r is not None:
            merged["text_density_score"] = density_r
            status["text_density_score"] = "api ✅ Vivek"
        else:
            status["text_density_score"] = "local ⚡ (Vivek returned unexpected format)"
    else:
        status["stroke_width_score"] = "local ⚡ (Vivek API offline)"
        status["text_density_score"] = "local ⚡ (Vivek API offline)"

    # ── Yash — always local ───────────────────────
    status["noise_score"]      = "local ✅ Yash"
    status["resolution_score"] = "local ✅ Yash"

    # ── Tanusha — connected component stability + skew ──
    data = _post_image(TANUSHA_API_URL, img_bgr)
    if data is not None:
        ccs_r, skew_r = _parse_tanusha(data)
        if ccs_r is not None:
            merged["connected_component_stability_score"] = ccs_r
            status["connected_component_stability_score"] = "api ✅ Tanusha"
        else:
            status["connected_component_stability_score"] = "local ⚡ (Tanusha returned unexpected format)"
        if skew_r is not None:
            merged["skew_penalty_score"] = skew_r
            status["skew_penalty_score"] = "api ✅ Tanusha"
        else:
            status["skew_penalty_score"] = "local ⚡ (Tanusha returned unexpected format)"
    else:
        status["connected_component_stability_score"] = "local ⚡ (Tanusha API offline)"
        status["skew_penalty_score"] = "local ⚡ (Tanusha API offline)"

    # ── Recompute OCR Readiness with final scores ─
    from factors import WEIGHTS, _clamp
    ocr_readiness = sum(
        merged[k]["score"] * w
        for k, w in WEIGHTS.items()
        if k in merged
    )
    merged["ocr_readiness_score"]  = round(_clamp(ocr_readiness), 1)
    merged["ocr_readiness_status"] = _classify(merged["ocr_readiness_score"])

    return merged, status
