"""
OCR Readiness Evaluation Platform
Factor computation engine — all 8 quality factors.

Each function accepts a numpy BGR image (OpenCV format) and returns
a dict with: score (0-100), status, description, details.

Team API integration notes:
  - Vivek  → stroke_width_score, text_density_score
  - Mansi  → blur_score, contrast_score
  - Krish  → matra_continuity_score, zone_integrity_score
  - Yash   → noise_score, resolution_score  (+ integration)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _classify(score: float) -> str:
    if score >= 81:
        return "Excellent"
    elif score >= 61:
        return "Good"
    elif score >= 41:
        return "Average"
    return "Poor"


def _clamp(v: float, lo=0.0, hi=100.0) -> float:
    return float(max(lo, min(hi, v)))


# ──────────────────────────────────────────────
# YASH — Noise Score
# ──────────────────────────────────────────────

def noise_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Estimates image noise using the high-frequency residual method.
    A Gaussian-blurred copy is subtracted from the original; the
    standard deviation of the residual is the noise estimate.
    Score = 100 − clamp(noise_std / 0.5, 0, 100)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray - blurred
    noise_std = float(np.std(residual))
    # Typical noise_std: 0 = perfect, ~50 = very noisy
    score = _clamp(100.0 - (noise_std / 0.50))
    return {
        "factor_name": "noise_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description": f"Estimated noise σ = {noise_std:.2f}. "
                       + ("Low noise — good OCR candidate." if score >= 70
                          else "Moderate noise detected." if score >= 45
                          else "High noise — apply denoising filter."),
        "raw_value": round(noise_std, 3),
        "unit": "σ (std of residual)",
    }


# ──────────────────────────────────────────────
# YASH — Resolution Score
# ──────────────────────────────────────────────

def resolution_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates pixel dimensions for OCR suitability.
    OCR works best at ≥300 DPI equivalent; heuristic: images ≥1500px
    on the long side score 100, images <200px score 0.
    Score = clamp((long_side − 200) / 13, 0, 100)
    """
    h, w = img_bgr.shape[:2]
    long_side = max(h, w)
    score = _clamp((long_side - 200) / 13.0)
    mp = (h * w) / 1_000_000
    return {
        "factor_name": "resolution_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description": f"{w}×{h} px ({mp:.2f} MP). "
                       + ("High resolution — text should be sharp." if score >= 70
                          else "Moderate resolution." if score >= 45
                          else "Low resolution — capture at higher DPI."),
        "raw_value": long_side,
        "unit": "px (long side)",
    }


# ──────────────────────────────────────────────
# MANSI — Blur Score
# ──────────────────────────────────────────────

def blur_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Variance of Laplacian method.
    A sharp image has high edge variance; a blurry one is low.
    Score = clamp(var_laplacian / 5, 0, 100)
    Threshold tuned for typical document scans (var ~50–500).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var_lap = float(lap.var())
    score = _clamp(var_lap / 5.0)
    return {
        "factor_name": "blur_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description": f"Laplacian variance = {var_lap:.1f}. "
                       + ("Sharp image — good edge definition." if score >= 70
                          else "Slight blur present." if score >= 45
                          else "Blurry image — improve camera focus."),
        "raw_value": round(var_lap, 2),
        "unit": "Laplacian variance",
    }


# ──────────────────────────────────────────────
# MANSI — Contrast Score
# ──────────────────────────────────────────────

def contrast_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Histogram spread method.
    Contrast = (p95 − p5) of the pixel intensity histogram.
    Score = clamp((p95 − p5) / 2.55, 0, 100)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    p5 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    spread = p95 - p5
    score = _clamp(spread / 2.55)
    return {
        "factor_name": "contrast_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description": f"Intensity range p5={p5:.0f} → p95={p95:.0f} (spread={spread:.0f}/255). "
                       + ("High contrast — text clearly distinguishable." if score >= 70
                          else "Moderate contrast." if score >= 45
                          else "Low contrast — apply thresholding or brightness adjustment."),
        "raw_value": round(spread, 2),
        "unit": "intensity spread (0-255)",
    }


# ──────────────────────────────────────────────
# VIVEK — Stroke Width Score
# ──────────────────────────────────────────────

def stroke_width_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Estimates stroke width via skeleton of binarised text regions.
    Distance transform on the skeleton gives local stroke radii;
    median × 2 = estimated stroke width.
    Ideal OCR stroke width: 1-5 px.
    Score = 100 − clamp(|median_sw − 3| × 15, 0, 100)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Distance transform on text pixels
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    # Skeleton-like: keep pixels where dist is local max
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(dist, kernel)
    skeleton_mask = (dist == dilated) & (binary > 0)
    stroke_radii = dist[skeleton_mask]
    if len(stroke_radii) == 0:
        return {
            "factor_name": "stroke_width_score",
            "score": 50.0,
            "status": "Average",
            "description": "Could not detect text strokes.",
            "raw_value": 0,
            "unit": "px",
        }
    median_sw = float(np.median(stroke_radii)) * 2  # radius → width
    score = _clamp(100.0 - abs(median_sw - 3.0) * 15.0)
    return {
        "factor_name": "stroke_width_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description": f"Median stroke width ≈ {median_sw:.1f} px. "
                       + ("Ideal stroke width for OCR." if score >= 70
                          else "Stroke width slightly outside optimal range." if score >= 45
                          else "Stroke too thin or thick — may affect character recognition."),
        "raw_value": round(median_sw, 2),
        "unit": "px (stroke width)",
    }


# ──────────────────────────────────────────────
# VIVEK — Text Density Score
# ──────────────────────────────────────────────

def text_density_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Ratio of text pixels (dark foreground) to total image pixels.
    Uses Otsu thresholding.
    Ideal range for a text document: 5–35 % text pixel coverage.
    Score peaks at 20 % coverage and falls off towards 0 or 100 %.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_pixels = int(np.sum(binary > 0))
    total_pixels = binary.size
    density_pct = 100.0 * text_pixels / total_pixels
    # Bell curve centred at 20 %, std ~15 %
    score = _clamp(100.0 * np.exp(-0.5 * ((density_pct - 20.0) / 15.0) ** 2))
    return {
        "factor_name": "text_density_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description": f"Text coverage = {density_pct:.1f}% of image. "
                       + ("Optimal text density." if score >= 70
                          else "Text is sparse or very dense." if score >= 45
                          else "Extremely sparse or overcrowded text region."),
        "raw_value": round(density_pct, 2),
        "unit": "% text pixel coverage",
    }


# ──────────────────────────────────────────────
# KRISH — Matra Continuity Score
# ──────────────────────────────────────────────

def matra_continuity_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Measures continuity of the Devanagari Shirorekha (header line /
    matra). A strong horizontal run in the upper-third of each text
    line indicates a well-preserved matra.
    Method:
      1. Binarise + find horizontal runs per row.
      2. In the top ~30 % of each detected text band, measure max
         run length as a fraction of line width.
      3. Average across text bands → continuity %.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape

    # Horizontal projection profile to find text bands
    row_sums = np.sum(binary > 0, axis=1).astype(float)
    threshold_row = row_sums.max() * 0.05
    in_band = row_sums > threshold_row

    # Find contiguous bands
    bands = []
    start = None
    for i, val in enumerate(in_band):
        if val and start is None:
            start = i
        elif not val and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, h))

    if not bands:
        return {
            "factor_name": "matra_continuity_score",
            "score": 50.0,
            "status": "Average",
            "description": "No text bands detected.",
            "raw_value": 0,
            "unit": "% continuity",
        }

    continuities = []
    for (r0, r1) in bands:
        band_h = r1 - r0
        matra_zone = binary[r0: r0 + max(1, band_h // 3), :]  # top 33 %
        for row in matra_zone:
            runs = []
            count = 0
            for px in row:
                if px > 0:
                    count += 1
                else:
                    if count:
                        runs.append(count)
                    count = 0
            if count:
                runs.append(count)
            if runs:
                continuities.append(max(runs) / w)

    if not continuities:
        avg_cont = 0.5
    else:
        avg_cont = float(np.mean(continuities))

    score = _clamp(avg_cont * 130.0)  # scale: 0.77 run ratio → 100
    return {
        "factor_name": "matra_continuity_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description": f"Avg Shirorekha run = {avg_cont*100:.1f}% of line width. "
                       + ("Matra well-preserved." if score >= 70
                          else "Some matra breaks detected." if score >= 45
                          else "Significant matra breaks — poor Devanagari OCR expected."),
        "raw_value": round(avg_cont * 100, 2),
        "unit": "% avg run length",
    }


# ──────────────────────────────────────────────
# KRISH — Zone Integrity Score
# ──────────────────────────────────────────────

def zone_integrity_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Devanagari characters have three zones:
      Upper  — vowel marks / matras above shirorekha
      Middle — main body of characters
      Lower  — descenders / vowel marks below
    Method:
      For each detected text band, check that all three vertical
      thirds contain pixel mass ≥ threshold.  The fraction of bands
      where all three zones are intact = integrity ratio.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape

    row_sums = np.sum(binary > 0, axis=1).astype(float)
    threshold_row = row_sums.max() * 0.05
    in_band = row_sums > threshold_row

    bands = []
    start = None
    for i, val in enumerate(in_band):
        if val and start is None:
            start = i
        elif not val and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, h))

    if not bands:
        return {
            "factor_name": "zone_integrity_score",
            "score": 50.0,
            "status": "Average",
            "description": "No text bands found for zone analysis.",
            "raw_value": 0,
            "unit": "% bands with 3 intact zones",
        }

    intact = 0
    min_mass = w * 0.02  # at least 2 % of row width must have pixels
    for (r0, r1) in bands:
        bh = r1 - r0
        third = max(1, bh // 3)
        upper = binary[r0:          r0 + third, :]
        middle = binary[r0 + third:  r0 + 2*third, :]
        lower  = binary[r0 + 2*third: r1, :]
        if (upper.sum() / 255 >= min_mass and
                middle.sum() / 255 >= min_mass and
                lower.sum() / 255 >= min_mass):
            intact += 1

    ratio = intact / len(bands)
    score = _clamp(ratio * 100.0)
    return {
        "factor_name": "zone_integrity_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description": f"{intact}/{len(bands)} bands have all 3 zones intact ({ratio*100:.0f}%). "
                       + ("All character zones present." if score >= 70
                          else "Some zone components missing." if score >= 45
                          else "Significant zone data missing — structural integrity poor."),
        "raw_value": round(ratio * 100, 2),
        "unit": "% complete bands",
    }


# ──────────────────────────────────────────────
# TANUSHA — Connected Component Stability Score
# ──────────────────────────────────────────────

def connected_component_stability_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Measures how uniform/stable the detected character (connected
    component) sizes are. Stable, well-segmented text has components
    of fairly consistent area; noisy or broken text produces a wide
    spread of component sizes.

    Method:
      1. Binarise (Otsu).
      2. Find connected components, drop very tiny noise specks
         (< 4 px area) and the background label.
      3. Compute coefficient of variation (CV = std/mean) of
         component areas.
      4. Score = clamp(100 − CV*40, 0, 100)
         Lower CV (more uniform components) → higher score.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # stats[:,4] = area; label 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    areas = areas[areas >= 4]  # drop tiny specks

    if len(areas) < 2:
        return {
            "factor_name": "connected_component_stability_score",
            "score": 50.0,
            "status": "Average",
            "description": "Not enough text components detected to evaluate stability.",
            "raw_value": len(areas),
            "unit": "components",
        }

    mean_area = float(np.mean(areas))
    std_area  = float(np.std(areas))
    cv_val    = std_area / mean_area if mean_area > 0 else 0

    score = _clamp(100.0 - cv_val * 40.0)
    return {
        "factor_name": "connected_component_stability_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description": f"{len(areas)} components detected, size CV = {cv_val:.2f}. "
                       + ("Character sizes very consistent — clean segmentation." if score >= 70
                          else "Moderate variation in character sizes." if score >= 45
                          else "Highly inconsistent character sizes — likely noise or broken glyphs."),
        "raw_value": round(cv_val, 3),
        "unit": "coefficient of variation",
    }


# ──────────────────────────────────────────────
# TANUSHA — Skew Penalty Score
# ──────────────────────────────────────────────

def skew_penalty_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Estimates the skew (rotation) angle of the text lines using the
    minimum-area bounding box of all foreground pixels, and converts
    the deviation from 0° into a penalty score.

    Method:
      1. Binarise (Otsu) and collect foreground pixel coordinates.
      2. cv2.minAreaRect on those points gives an angle in [-90, 0).
      3. Normalise angle to the smallest deviation from horizontal
         (0° or 90°), giving skew_deg in [0, 45].
      4. Score = clamp(100 − skew_deg * 6, 0, 100)
         0° skew → 100; 16.7°+ skew → 0.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return {
            "factor_name": "skew_penalty_score",
            "score": 50.0,
            "status": "Average",
            "description": "Not enough foreground pixels to estimate skew.",
            "raw_value": 0,
            "unit": "degrees",
        }

    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    # Normalise to [0, 45] = deviation from nearest horizontal/vertical axis
    if angle < -45:
        angle = -(90 + angle)
    skew_deg = abs(angle)
    if skew_deg > 45:
        skew_deg = 90 - skew_deg

    score = _clamp(100.0 - skew_deg * 6.0)
    return {
        "factor_name": "skew_penalty_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description": f"Estimated skew = {skew_deg:.2f}°. "
                       + ("Image is well-aligned." if score >= 70
                          else "Slight tilt detected." if score >= 45
                          else "Significant skew — deskew before OCR."),
        "raw_value": round(skew_deg, 2),
        "unit": "° skew angle",
    }


# ──────────────────────────────────────────────
# Master scorer
# ──────────────────────────────────────────────

WEIGHTS = {
    "noise_score":                       0.11,
    "resolution_score":                  0.11,
    "blur_score":                        0.15,
    "contrast_score":                    0.11,
    "stroke_width_score":                0.07,
    "text_density_score":                0.07,
    "matra_continuity_score":            0.07,
    "zone_integrity_score":              0.03,
    "connected_component_stability_score": 0.09,
    "skew_penalty_score":                0.09,
}

FACTOR_FUNCTIONS = {
    "noise_score":            noise_score,
    "resolution_score":       resolution_score,
    "blur_score":             blur_score,
    "contrast_score":         contrast_score,
    "stroke_width_score":     stroke_width_score,
    "text_density_score":     text_density_score,
    "matra_continuity_score": matra_continuity_score,
    "zone_integrity_score":   zone_integrity_score,
    "connected_component_stability_score": connected_component_stability_score,
    "skew_penalty_score":     skew_penalty_score,
}

DISPLAY_NAMES = {
    "noise_score":            "Noise",
    "resolution_score":       "Resolution",
    "blur_score":             "Blur",
    "contrast_score":         "Contrast",
    "stroke_width_score":     "Stroke Width",
    "text_density_score":     "Text Density",
    "matra_continuity_score": "Matra Continuity",
    "zone_integrity_score":   "Zone Integrity",
    "connected_component_stability_score": "CC Stability",
    "skew_penalty_score":     "Skew Penalty",
}


def run_all_factors(img_bgr: np.ndarray) -> Dict[str, Any]:
    results = {}
    for key, fn in FACTOR_FUNCTIONS.items():
        try:
            results[key] = fn(img_bgr)
        except Exception as e:
            results[key] = {
                "factor_name": key,
                "score": 50.0,
                "status": "Error",
                "description": f"Computation error: {e}",
                "raw_value": None,
                "unit": "",
            }

    # OCR Readiness Score
    ocr_readiness = sum(
        results[k]["score"] * w for k, w in WEIGHTS.items()
    )
    ocr_readiness = round(_clamp(ocr_readiness), 1)
    results["ocr_readiness_score"] = ocr_readiness
    results["ocr_readiness_status"] = _classify(ocr_readiness)
    return results


# ──────────────────────────────────────────────
# Recommendation engine
# ──────────────────────────────────────────────

RECOMMENDATIONS = {
    "noise_score": {
        "threshold": 50,
        "text": "🔧 **Noise too high** — Apply a Gaussian or median denoising filter before OCR.",
    },
    "resolution_score": {
        "threshold": 50,
        "text": "🔧 **Low resolution** — Rescan or recapture the document at ≥300 DPI.",
    },
    "blur_score": {
        "threshold": 50,
        "text": "🔧 **Image blurry** — Improve camera focus, use a tripod, or apply unsharp mask.",
    },
    "contrast_score": {
        "threshold": 50,
        "text": "🔧 **Low contrast** — Increase brightness/contrast or apply adaptive thresholding.",
    },
    "stroke_width_score": {
        "threshold": 50,
        "text": "🔧 **Stroke width suboptimal** — Use a font/print size closer to 10–14 pt at scan DPI.",
    },
    "text_density_score": {
        "threshold": 40,
        "text": "🔧 **Text density unusual** — Crop tightly around text or check for blank regions.",
    },
    "matra_continuity_score": {
        "threshold": 50,
        "text": "🔧 **Matra breaks detected** — Clean document, remove folds/tears, or deskew.",
    },
    "zone_integrity_score": {
        "threshold": 50,
        "text": "🔧 **Zone integrity low** — Ensure full character visibility; avoid clipping text edges.",
    },
    "connected_component_stability_score": {
        "threshold": 50,
        "text": "🔧 **Inconsistent character segmentation** — Clean speckle noise and ensure uniform stroke thickness before OCR.",
    },
    "skew_penalty_score": {
        "threshold": 50,
        "text": "🔧 **Image is skewed** — Deskew/rotate the document so text lines are horizontal before OCR.",
    },
}


def generate_recommendations(results: Dict[str, Any]) -> list:
    recs = []
    for key, cfg in RECOMMENDATIONS.items():
        if key in results and results[key]["score"] < cfg["threshold"]:
            recs.append(cfg["text"])
    if not recs:
        recs.append("✅ All factors are within acceptable ranges. Image looks ready for OCR!")
    return recs
