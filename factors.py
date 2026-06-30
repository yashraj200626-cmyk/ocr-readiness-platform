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
# ------------------------------------------------
# Noise Score (OCR Calibrated)
# ------------------------------------------------
def noise_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Estimates random image noise.

    Method:
    1. Convert image to grayscale.
    2. Apply Gaussian blur to create a smooth version.
    3. Find difference between original and smoothed image.
    4. Ignore strong edges to avoid treating text as noise.
    5. Convert noise value into a 0–100 OCR score.

    Higher score = cleaner image.
    """

    # Validate image
    if img_bgr is None or img_bgr.size == 0:
        return {
            "factor_name": "noise_score",
            "score": 0,
            "status": "Poor",
            "description": "No image data available.",
            "raw_value": 0,
            "unit": "Noise level"
        }

    # Convert to grayscale
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()

    # Smooth image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Difference image (contains noise + edges)
    residual = cv2.absdiff(gray, blurred)

    # Detect strong text edges
    edges = cv2.Canny(gray, 80, 150)

    # Remove text edge regions from noise calculation
    background_mask = edges == 0

    # If very few background pixels exist, use full image
    if np.sum(background_mask) > 100:
        noise_value = float(np.mean(residual[background_mask]))
    else:
        noise_value = float(np.mean(residual))

    # OCR-based calibration
    # 0   -> perfect
    # 25+ -> extremely noisy
    score = 100 - (noise_value * 4)

    # Keep score within 0–100
    score = float(np.clip(score, 0, 100))

    return {
        "factor_name": "noise_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description":
            f"Estimated noise level = {noise_value:.2f}. " +
            (
                "Low noise detected. Image is suitable for OCR."
                if score >= 70 else
                "Moderate noise detected. OCR may require preprocessing."
                if score >= 35 else
                "High noise detected. Apply denoising before OCR."
            ),
        "raw_value": round(noise_value, 2),
        "unit": "Average residual intensity"
    }


# ------------------------------------------------
# Resolution Score (OCR Calibrated)
# ------------------------------------------------
def resolution_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates image resolution for OCR.

    Uses total image pixels (megapixels)
    instead of only one dimension.

    Higher score = more character detail.
    """

    # Validate image
    if img_bgr is None or img_bgr.size == 0:
        return {
            "factor_name": "resolution_score",
            "score": 0,
            "status": "Poor",
            "description": "No image data available.",
            "raw_value": 0,
            "unit": "Megapixels"
        }

    # Get image size
    height, width = img_bgr.shape[:2]

    # Calculate megapixels
    megapixels = (height * width) / 1_000_000


    # OCR resolution calibration
    if megapixels >= 2.0:
        score = 100

    elif megapixels >= 1.0:
        # 1–2 MP → 70–100 score
        score = 70 + ((megapixels - 1.0) * 30)

    elif megapixels >= 0.3:
        # 0.3–1 MP → 35–70 score
        score = 35 + ((megapixels - 0.3) / 0.7) * 35

    else:
        # Very low resolution
        score = (megapixels / 0.3) * 35


    score = float(np.clip(score, 0, 100))


    return {
        "factor_name": "resolution_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description":
            f"Image size = {width} × {height} pixels "
            f"({megapixels:.2f} MP). " +
            (
                "High resolution with sufficient text details."
                if score >= 70 else
                "Moderate resolution. OCR may have minor issues."
                if score >= 35 else
                "Low resolution. Capture a higher-quality image."
            ),
        "raw_value": round(megapixels, 2),
        "unit": "Megapixels"
    }

# ──────────────────────────────────────────────
# Blur Score (OCR calibrated)
# ──────────────────────────────────────────────
def blur_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Measure text sharpness using Laplacian variance.
    Higher score = sharper text.
    """

    if img_bgr is None or img_bgr.size == 0:
        return {
            "factor_name": "blur_score",
            "score": 0,
            "status": "Poor",
            "description": "No image data available.",
            "raw_value": 0,
            "unit": "Laplacian variance"
        }

    # Convert to grayscale
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()

    # Slight denoising to avoid noise creating fake edges
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Calculate edge strength
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Better scaling for documents
    # 0–20   : very blurry
    # 20–100 : moderate blur
    # 100+   : sharp text
    score = (
        (np.log1p(lap_var) - np.log1p(3)) /
        (np.log1p(600) - np.log1p(3))
    ) * 100

    score = float(np.clip(score, 0, 100))

    return {
        "factor_name": "blur_score",
        "score": round(score, 1),
        "status": _classify(score),
        "description": (
            f"Laplacian variance = {lap_var:.2f}. " +
            (
                "Text edges are sharp and OCR readability is high."
                if score >= 70 else
                "Some blur is present. OCR may have minor errors."
                if score >= 35 else
                "Heavy blur detected. OCR accuracy may be poor."
            )
        ),
        "raw_value": round(lap_var, 2),
        "unit": "Laplacian variance"
    }

# ──────────────────────────────────────────────
# Contrast Score (OCR calibrated)
# ──────────────────────────────────────────────
def contrast_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    OCR-aware contrast score.
    Measures text/background separation and adjusts for low resolution.
    Higher score = better OCR readability.
    """

    if img_bgr is None or img_bgr.size == 0:
        return {
            "factor_name": "contrast_score",
            "score": 0,
            "status": "Poor",
            "description": "No image data available.",
            "raw_value": 0,
            "unit": "Intensity difference"
        }

    # Convert to grayscale
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()

    h, w = gray.shape
    pixels = h * w

    # Robust contrast using percentiles
    p5 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    contrast = p95 - p5

    # Normalize contrast to 0-100
    # 20 = very poor, 180 = excellent
    base_score = ((contrast - 20) / (180 - 20)) * 100
    base_score = float(np.clip(base_score, 0, 100))

    # --------------------------------------------------
    # Resolution penalty for OCR
    # Small images lose character details
    # --------------------------------------------------
    if pixels >= 1_000_000:
        res_factor = 1.0      # High resolution
    elif pixels >= 500_000:
        res_factor = 0.90
    elif pixels >= 250_000:
        res_factor = 0.80
    else:
        res_factor = 0.70     # Very small documents

    # Keep some contrast credit even for low resolution
    final_score = base_score * (0.5 + 0.5 * res_factor)
    final_score = float(np.clip(final_score, 0, 100))

    return {
        "factor_name": "contrast_score",
        "score": round(final_score, 1),
        "status": _classify(final_score),
        "description": (
            f"Contrast difference = {contrast:.1f}. "
            + (
                "Excellent text/background separation."
                if final_score >= 80 else
                "Moderate contrast suitable for OCR."
                if final_score >= 50 else
                "Low contrast; OCR accuracy may decrease."
            )
        ),
        "raw_value": round(contrast, 2),
        "unit": "Intensity difference"
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
                       + ("text density too high" if score >= 70
                          else "Text is sparse or very dense." 
                          if score >= 45
                          else "Extremely sparse"),
        "raw_value": round(density_pct, 2),
        "unit": "% text pixel coverage",
    }

#matra scoring

def matra_continuity_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Self-contained MCS using the same logic as the full pipeline.
    Fixed for handwriting and printed Hindi text.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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
            "factor_name": "matra_continuity_score",
            "score": 50.0,
            "status": "Average",
            "description": "No text bands detected.",
            "raw_value": 0,
            "unit": "MCS (0-100)",
        }

    band_scores = []
    for (r0, r1) in bands:
        band_h = r1 - r0
        if band_h < 5:
            continue

        shiro_top    = r0
        shiro_bottom = r0 + max(1, int(band_h * 0.15))
        upper_top    = r0
        upper_bottom = r0 + max(1, int(band_h * 0.20))
        lower_top    = r0 + max(1, int(band_h * 0.75))
        lower_bottom = r1
        middle_top   = r0 + max(1, int(band_h * 0.35))
        middle_bottom= r0 + max(1, int(band_h * 0.75))

        shiro_zone  = binary[shiro_top:shiro_bottom, :]
        upper_zone  = binary[upper_top:upper_bottom, :]
        lower_zone  = binary[lower_top:lower_bottom, :]
        middle_zone = binary[middle_top:middle_bottom, :]
        full_line   = binary[r0:r1, :]

        # ── SCS: Shirorekha Continuity Score ────────────────────────
        if shiro_zone.size > 0:
            col_ink = shiro_zone.sum(axis=0)
            zone_h  = shiro_zone.shape[0]
            # FIX 1: relaxed threshold 0.30 → 0.15
            active  = col_ink >= max(1, int(zone_h * 0.15) * 255)

            max_gap = 0
            current_gap = 0
            for val in active:
                if not val:
                    current_gap += 1
                    max_gap = max(max_gap, current_gap)
                else:
                    current_gap = 0
            # Dynamic gap threshold based on image width
            gap_threshold = max(20, int(w * 0.02))  # 2% of image width
            if max_gap <= gap_threshold:
                scs = 100.0
            else:
                # FIX 2: softer penalty 5.0 → 2.0
                penalty = min(100.0, (max_gap - gap_threshold) * 2.0)
                scs = max(0.0, 100.0 - penalty)
        else:
            scs = 50.0

        # ── MVS: Matra Visibility Score ──────────────────────────────
        def mvs(zone):
            if zone.size == 0:
                return 50.0
            n, _, stats, _ = cv2.connectedComponentsWithStats(
                zone, connectivity=8)
            if n <= 1:
                return 100.0  # FIX 3: clean zone not a failure
            min_area = 4
            valid = sum(1 for i in range(1, n)
                        if stats[i, cv2.CC_STAT_AREA] >= min_area)
            total = n - 1
            if total == 0:
                return 100.0  # FIX 3: clean zone not a failure
            return float(np.clip((valid / total) * 100, 0, 100))

        mvs_upper = mvs(upper_zone)
        mvs_lower = mvs(lower_zone)

        # ── RLR: Run-Length Regularity ───────────────────────────────
        runs = []
        for row in range(full_line.shape[0]):
            in_run = False
            length = 0
            for px in full_line[row]:
                if px > 0:
                    length += 1
                    in_run = True
                else:
                    if in_run and length >= 2:
                        runs.append(length)
                    length = 0
                    in_run = False
            if in_run and length >= 2:
                runs.append(length)

        if runs:
            runs_arr = np.array(runs, dtype=np.float32)
            mean_r = runs_arr.mean()
            if mean_r > 0:
                    cov = runs_arr.std() / mean_r
                    rlr = float(np.clip(100.0 * np.exp(-cov * 0.5), 0, 100))
            else:
                rlr = 0.0
        else:
            rlr = 0.0

        # ── BS: Baseline Stability ───────────────────────────────────
        if middle_zone.size > 0:
            ink = (middle_zone > 0).astype(np.float32)
            col_sum = ink.sum(axis=0) + 1e-6
            row_idx = np.arange(middle_zone.shape[0], dtype=np.float32)
            coms = (ink * row_idx[:, None]).sum(axis=0) / col_sum
            var = float(coms.var())
            # FIX 4: relaxed variance max 5.0 → 30.0
            bs = float(np.clip(100.0 * np.exp(-var / 30.0), 0, 100))
        else:
            bs = 50.0

        # ── NS: Noise Suppression ────────────────────────────────────
        n_cc, _, stats_ns, _ = cv2.connectedComponentsWithStats(
            full_line, connectivity=8)
        if n_cc > 1:
            noise_count = sum(1 for i in range(1, n_cc)
                              if stats_ns[i, cv2.CC_STAT_AREA] <= 12)
            ns = float(np.clip(
                100.0 * (1.0 - noise_count / max(1, n_cc - 1)), 0, 100))
        else:
            ns = 100.0

        # ── SS: Stroke Straightness ──────────────────────────────────
        import math
        edges = cv2.Canny(full_line, 50, 150)
        lines_h = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                   threshold=20,
                                   minLineLength=10,
                                   maxLineGap=5)
        if lines_h is not None and len(lines_h) > 0:
            angles = []
            for seg in lines_h:
                x1, y1, x2, y2 = seg[0]
                angles.append(
                    math.degrees(math.atan2(y2 - y1, x2 - x1)))
            std = float(np.array(angles).std())
            # FIX 5: relaxed angle std 5.0 → 10.0
            ss = float(np.clip(100.0 * np.exp(-std / 10.0), 0, 100))
        else:
            ss = 50.0

        # ── MCS formula — rebalanced weights ─────────────────────────
        line_mcs = (0.22 * scs +
            0.28 * mvs_upper +  # ← increased
            0.10 * rlr +        # ← reduced significantly
            0.18 * mvs_lower +  # ← increased
            0.12 * bs +
            0.06 * ns +
            0.04 * ss)
        band_scores.append(float(np.clip(line_mcs, 0, 100)))

    if not band_scores:
        score = 50.0
    else:
        score = float(np.mean(band_scores))
        score = float(np.clip(score, 0, 100))

    status = ("Excellent" if score >= 81 else "Good" if score >= 61
              else "Average" if score >= 41 else "Poor")
    return {
        "factor_name": "matra_continuity_score",
        "score": round(score, 1),
        "status": status,
        "description": f"Shirorekha continuity and matra visibility = {score:.1f}/100. "
                       + ("Matra well-preserved." if score >= 70
                          else "Matra continuity acceptable." if score >= 45
                          else "Significant matra breaks — poor Devanagari OCR expected."),
        "raw_value": round(score, 2),
        "unit": "MCS (0-100)",
    }


# ──────────────────────────────────────────────
# KRISH — Zone Integrity Score (self-contained)
# ──────────────────────────────────────────────

def zone_integrity_score(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Self-contained ZIS using the same logic as the full pipeline.
    Scores each Devanagari zone (upper, shiro, middle, lower) on
    FRAG, STROKE, FILL, SHARP and combines with Pal-Chaudhuri weights.
    Fixed: nan guards, division by zero protection.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape

    # ── Detect text lines ────────────────────────────────────────────
    row_sums = np.sum(binary > 0, axis=1).astype(float)
    if row_sums.max() == 0:
        return {
            "factor_name": "zone_integrity_score",
            "score": 50.0,
            "status": "Average",
            "description": "No ink detected in image.",
            "raw_value": 0,
            "unit": "ZIS (0-100)",
        }

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
            "unit": "ZIS (0-100)",
        }

    # ── Zone scoring functions ───────────────────────────────────────
    def frag_score(patch):
        if patch is None or patch.size == 0:
            return 50.0
        try:
            n, _, stats, _ = cv2.connectedComponentsWithStats(
                patch, connectivity=8)
            valid = [i for i in range(1, n)
                     if stats[i, cv2.CC_STAT_AREA] >= 3]
            if not valid:
                return 50.0  # FIX: neutral not 0
            total_ink = sum(stats[i, cv2.CC_STAT_AREA] for i in valid)
            mean_cc = total_ink / max(1, len(valid))
            rel = mean_cc / max(1, patch.size)  # FIX: avoid /0
            result = float(np.clip(rel * 10000, 0, 100))
            return result if not np.isnan(result) else 50.0
        except Exception:
            return 50.0

    def stroke_score(patch):
        if patch is None or patch.size == 0:
            return 50.0
        try:
            if patch.sum() == 0:
                return 50.0  # FIX: no ink = neutral
            dist = cv2.distanceTransform(patch, cv2.DIST_L2, 5)
            vals = dist[dist > 0]
            if len(vals) < 5:
                return 50.0
            mean_val = float(vals.mean())
            if mean_val == 0:
                return 50.0  # FIX: avoid /0
            cov = float(vals.std() / mean_val)
            if cov <= 0.4:
                return 100.0
            excess = cov - 0.4
            result = float(np.clip(100.0 * np.exp(-excess * 3), 0, 100))
            return result if not np.isnan(result) else 50.0
        except Exception:
            return 50.0

    def fill_score(patch):
        if patch is None or patch.size == 0:
            return 50.0
        try:
            ratio = float(np.count_nonzero(patch)) / max(1, patch.size)  # FIX
            lo, hi = 0.05, 0.70
            if ratio < lo:
                result = float(np.clip((ratio / lo) * 100, 0, 100))
            elif ratio > hi:
                excess = (ratio - hi) / (1.0 - hi + 1e-6)
                result = float(np.clip(100.0 * (1.0 - excess), 0, 100))
            else:
                result = 100.0
            return result if not np.isnan(result) else 50.0
        except Exception:
            return 50.0

    def sharp_score(patch):
        if patch is None or patch.size == 0:
            return 50.0
        try:
            lap = cv2.Laplacian(patch.astype(np.float32), cv2.CV_32F)
            energy = float((lap ** 2).mean())
            if np.isnan(energy) or np.isinf(energy):
                return 50.0  # FIX: guard nan/inf
            result = float(np.clip(
                100.0 * (1.0 - np.exp(-energy / 50.0)), 0, 100))
            return result if not np.isnan(result) else 50.0
        except Exception:
            return 50.0

    def zone_score(patch):
        try:
            f  = frag_score(patch)
            s  = stroke_score(patch)
            fi = fill_score(patch)
            sh = sharp_score(patch)
            total = (0.35 * f + 0.25 * s + 0.20 * fi + 0.20 * sh)
            result = float(np.clip(total, 0, 100))
            return result if not np.isnan(result) else 50.0
        except Exception:
            return 50.0

    # ── Per-line ZIS ─────────────────────────────────────────────────
    line_scores = []
    for (r0, r1) in bands:
        band_h = r1 - r0
        if band_h < 5:
            continue

        try:
            upper_patch  = binary[r0: r0 + max(1, int(band_h * 0.20)), :]
            shiro_patch  = binary[r0 + max(1, int(band_h * 0.20)):
                                   r0 + max(1, int(band_h * 0.35)), :]
            middle_patch = binary[r0 + max(1, int(band_h * 0.35)):
                                   r0 + max(1, int(band_h * 0.75)), :]
            lower_patch  = binary[r0 + max(1, int(band_h * 0.75)): r1, :]

            zs_upper  = zone_score(upper_patch)
            zs_shiro  = zone_score(shiro_patch)
            zs_middle = zone_score(middle_patch)
            zs_lower  = zone_score(lower_patch)

            # Final ZIS with Pal-Chaudhuri weights
            zis = (0.40 * zs_shiro +
                   0.30 * zs_middle +
                   0.20 * zs_upper +
                   0.10 * zs_lower)

            zis = float(np.clip(zis, 0, 100))

            # FIX: only append valid scores
            if not np.isnan(zis) and not np.isinf(zis):
                line_scores.append(zis)

        except Exception:
            continue

    # ── Final score ──────────────────────────────────────────────────
    if not line_scores:
        score = 50.0
    else:
        # FIX: filter nan before averaging
        valid_scores = [s for s in line_scores
                        if not np.isnan(s) and not np.isinf(s)]
        if not valid_scores:
            score = 50.0
        else:
            score = float(np.mean(valid_scores))
            score = float(np.clip(score, 0, 100))

    # Final nan guard
    if np.isnan(score) or np.isinf(score):
        score = 50.0

    status = ("Excellent" if score >= 81 else "Good" if score >= 61
              else "Average" if score >= 41 else "Poor")

    return {
        "factor_name": "zone_integrity_score",
        "score": round(score, 1),
        "status": status,
        "description": f"Devanagari zone structural integrity = {score:.1f}/100. "
                       + ("All zones intact." if score >= 70
                          else "Some zone degradation." if score >= 45
                          else "Significant zone damage detected."),
        "raw_value": round(score, 2),
        "unit": "ZIS (0-100)",
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
    "noise_score":                       0.12,
    "resolution_score":                  0.12,
    "blur_score":                        0.15,
    "contrast_score":                    0.12,
    "stroke_width_score":                0.08,
    "text_density_score":                0.08,
    "matra_continuity_score":            0.08,
    "zone_integrity_score":              0.05,
    "connected_component_stability_score": 0.10,
    "skew_penalty_score":                0.10,
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
