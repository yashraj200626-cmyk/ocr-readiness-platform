"""
Educational content for the About Factor section.
Each entry has: definition, importance, formula, ocr_impact, ideal_range.
"""

FACTOR_INFO = {
    "noise_score": {
        "display_name": "Noise Score",
        "owner": "Yash",
        "definition": (
            "Image noise refers to random variation of brightness or colour in pixels, "
            "caused by sensor limitations, low light, or compression artefacts. "
            "It appears as graininess or speckles overlaid on the true image content."
        ),
        "importance": (
            "Noise introduces false edges and distorts character shapes. OCR engines "
            "misread strokes that are interrupted or thickened by noise, leading to "
            "substitution errors (e.g. 'a' read as 'o', 'l' as 'i')."
        ),
        "formula": (
            "1. Blur the image with a 5×5 Gaussian kernel.\n"
            "2. Subtract the blurred image from the original → residual.\n"
            "3. Noise σ = std(residual).\n"
            "4. Score = clamp(100 − σ / 0.5, 0, 100)."
        ),
        "ocr_impact": (
            "High noise → OCR confidence drops sharply, especially for small fonts. "
            "Studies show denoising alone can improve character accuracy by 10–25 %."
        ),
        "ideal_range": "Score ≥ 70 (σ < 15). Excellent: Score ≥ 81.",
    },

    "resolution_score": {
        "display_name": "Resolution Score",
        "owner": "Yash",
        "definition": (
            "Resolution describes the pixel density of the image — how many pixels "
            "represent each unit of physical area. For scanned documents, this is "
            "expressed as DPI (dots per inch)."
        ),
        "importance": (
            "OCR engines need sufficient pixel detail to distinguish similar characters "
            "('rn' vs 'm', '0' vs 'O'). Below ~150 DPI, accuracy degrades rapidly."
        ),
        "formula": (
            "1. Find long_side = max(width, height) in pixels.\n"
            "2. Score = clamp((long_side − 200) / 13, 0, 100).\n"
            "   Calibrated so ~1500 px long side → Score ≈ 100."
        ),
        "ocr_impact": (
            "Tesseract is trained on 300 DPI data. Images below 200 DPI cause "
            "character segmentation errors. Higher resolution beyond 600 DPI "
            "shows diminishing OCR returns."
        ),
        "ideal_range": "Score ≥ 70 (≥ 1100 px long side, ~300 DPI equivalent). Excellent: ≥ 1500 px.",
    },

    "blur_score": {
        "display_name": "Blur Score",
        "owner": "Mansi",
        "definition": (
            "Blur is the loss of sharpness caused by camera defocus, motion during "
            "capture, or post-processing smoothing. It manifests as soft, indistinct "
            "edges between text strokes and background."
        ),
        "importance": (
            "OCR relies on detecting precise character edges. Blur spreads those edges, "
            "making thin strokes disappear and thick strokes merge."
        ),
        "formula": (
            "Variance of Laplacian method:\n"
            "1. Convert to greyscale.\n"
            "2. Apply Laplacian operator (edge detector).\n"
            "3. Compute variance of the resulting image.\n"
            "4. Score = clamp(variance / 5, 0, 100)."
        ),
        "ocr_impact": (
            "Blur is the single strongest predictor of OCR failure (correlation ~0.91 "
            "in published benchmarks). A Laplacian variance below 100 reliably "
            "indicates OCR accuracy below 80 %."
        ),
        "ideal_range": "Score ≥ 70 (Laplacian var ≥ 350). Excellent: var ≥ 405.",
    },

    "contrast_score": {
        "display_name": "Contrast Score",
        "owner": "Mansi",
        "definition": (
            "Contrast is the difference in luminance between the darkest and brightest "
            "regions of an image. High contrast means text strokes are clearly darker "
            "than the page background."
        ),
        "importance": (
            "Low contrast forces OCR to work harder to separate foreground (ink) from "
            "background (paper), increasing mis-segmentation and missed characters."
        ),
        "formula": (
            "Histogram spread method:\n"
            "1. Convert to greyscale.\n"
            "2. p5 = 5th percentile of pixel intensities.\n"
            "3. p95 = 95th percentile.\n"
            "4. spread = p95 − p5.\n"
            "5. Score = clamp(spread / 2.55, 0, 100)."
        ),
        "ocr_impact": (
            "Contrast below 50 (spread < 128/255) is associated with 15–30 % "
            "accuracy loss. Adaptive histogram equalisation (CLAHE) is a reliable fix."
        ),
        "ideal_range": "Score ≥ 70 (intensity spread ≥ 178/255). Excellent: spread ≥ 204.",
    },

    "stroke_width_score": {
        "display_name": "Stroke Width Score",
        "owner": "Vivek",
        "definition": (
            "Stroke width is the thickness of the lines that form character shapes. "
            "It depends on font weight, print size, and scanning resolution. "
            "For OCR, a consistent stroke width of 1–5 pixels is ideal."
        ),
        "importance": (
            "Very thin strokes break easily under noise or compression; very thick "
            "strokes cause character fills and merges adjacent characters."
        ),
        "formula": (
            "Distance-transform skeleton method:\n"
            "1. Binarise image (Otsu).\n"
            "2. Distance transform on text pixels.\n"
            "3. Extract skeleton (local maxima of distance transform).\n"
            "4. median_radius = median of distance values on skeleton.\n"
            "5. stroke_width = median_radius × 2.\n"
            "6. Score = clamp(100 − |stroke_width − 3| × 15, 0, 100)."
        ),
        "ocr_impact": (
            "Optimal OCR stroke width is 2–4 px. Strokes below 1 px or above 8 px "
            "consistently lower Tesseract confidence by 10–20 points."
        ),
        "ideal_range": "Score ≥ 70 (stroke width 2–4 px). Excellent: ~3 px.",
    },

    "text_density_score": {
        "display_name": "Text Density Score",
        "owner": "Vivek",
        "definition": (
            "Text density is the proportion of image pixels that belong to text "
            "(foreground / ink) versus the total image area. It captures how "
            "tightly packed the content is."
        ),
        "importance": (
            "Too sparse (< 5 %) suggests the crop includes a lot of blank space. "
            "Too dense (> 50 %) suggests touching characters, severe clutter, "
            "or an incorrectly thresholded image."
        ),
        "formula": (
            "1. Binarise image (Otsu thresholding).\n"
            "2. density = (text pixels / total pixels) × 100.\n"
            "3. Score = 100 × exp(−0.5 × ((density − 20) / 15)²).\n"
            "   Bell curve peaked at 20 % coverage, std = 15 %."
        ),
        "ocr_impact": (
            "OCR performs best when text occupies 10–30 % of the image. "
            "Extreme values indicate pre-processing problems that degrade accuracy."
        ),
        "ideal_range": "Score ≥ 70 (density ≈ 10–30 %). Excellent: ≈ 15–25 %.",
    },

    "matra_continuity_score": {
        "display_name": "Matra Continuity Score",
        "owner": "Krish",
        "definition": (
            "In Devanagari script, the Shirorekha (शिरोरेखा) is the horizontal "
            "headline that runs across the top of most characters, connecting them "
            "into visual word units. 'Matra' broadly refers to this connecting "
            "stroke and associated vowel marks. Continuity measures how unbroken "
            "this line is across a text band."
        ),
        "importance": (
            "Breaks in the Shirorekha confuse Devanagari OCR engines that rely on "
            "it for word segmentation. A broken matra often causes the engine to "
            "split one word into two or more fragments."
        ),
        "formula": (
            "1. Binarise and find horizontal text bands via row projection.\n"
            "2. For the top third of each band (Shirorekha zone), find the longest "
            "   continuous horizontal run per row.\n"
            "3. continuity = average(max_run / image_width) across all matra rows.\n"
            "4. Score = clamp(continuity × 130, 0, 100)."
        ),
        "ocr_impact": (
            "Devanagari OCR accuracy drops 20–40 % when matra continuity score is "
            "below 50, as the engine cannot reliably segment words."
        ),
        "ideal_range": "Score ≥ 70 (avg run > 54 % of line width). Excellent: > 77 %.",
    },

    "zone_integrity_score": {
        "display_name": "Zone Integrity Score",
        "owner": "Krish",
        "definition": (
            "Devanagari characters are structured in three vertical zones:\n"
            "• Upper zone — vowel marks (ā, i, ī, u, …) attached above the Shirorekha.\n"
            "• Middle zone — main body of the consonant/akshara.\n"
            "• Lower zone — descending vowel marks and some consonant forms.\n"
            "Zone integrity checks whether all three zones carry pixel content in "
            "each detected text band."
        ),
        "importance": (
            "Missing zone data (e.g. clipped upper zone) means the OCR engine "
            "cannot see vowel signs, causing transcription errors at every "
            "affected character."
        ),
        "formula": (
            "1. Binarise and find text bands via row projection.\n"
            "2. Split each band into three equal vertical thirds.\n"
            "3. A band is 'intact' if all three thirds contain ≥ 2 % pixel mass.\n"
            "4. integrity = (intact bands / total bands) × 100.\n"
            "5. Score = clamp(integrity, 0, 100)."
        ),
        "ocr_impact": (
            "Zone clipping is particularly harmful for Devanagari. Even a single "
            "missing upper-zone vowel mark changes word meaning entirely."
        ),
        "ideal_range": "Score ≥ 70 (≥ 70 % of bands fully intact). Excellent: ≥ 81 %.",
    },

    "connected_component_stability_score": {
        "display_name": "Connected Component Stability Score",
        "owner": "Tanusha",
        "definition": (
            "A 'connected component' is a group of touching foreground pixels — "
            "typically a single character or character part. Stability measures "
            "how uniform the sizes of these components are across the image."
        ),
        "importance": (
            "Clean, well-printed text produces characters of fairly consistent "
            "size. Noise, broken strokes, smudges, or stray marks create extra "
            "components of very different sizes, which confuses the OCR "
            "segmentation step."
        ),
        "formula": (
            "1. Binarise image (Otsu).\n"
            "2. Find all connected components and their pixel areas "
            "   (drop specks < 4 px).\n"
            "3. CV = std(areas) / mean(areas)  (coefficient of variation).\n"
            "4. Score = clamp(100 − CV × 40, 0, 100)."
        ),
        "ocr_impact": (
            "High variance in component sizes (CV > 1.25) is strongly correlated "
            "with broken or merged characters, which directly reduces OCR "
            "accuracy through mis-segmentation."
        ),
        "ideal_range": "Score ≥ 70 (CV ≤ 0.75). Excellent: CV ≤ 0.5.",
    },

    "skew_penalty_score": {
        "display_name": "Skew Penalty Score",
        "owner": "Tanusha",
        "definition": (
            "Skew is the rotation angle of the text lines relative to the "
            "horizontal axis, usually caused by a tilted scan or photo capture."
        ),
        "importance": (
            "OCR engines assume text runs horizontally. Even a small skew angle "
            "causes character baselines to drift, leading to incorrect line "
            "segmentation and reading-order errors."
        ),
        "formula": (
            "1. Binarise image (Otsu) and collect foreground pixel coordinates.\n"
            "2. Compute cv2.minAreaRect over those points to get an angle.\n"
            "3. Normalise to skew_deg ∈ [0°, 45°] (deviation from horizontal/vertical).\n"
            "4. Score = clamp(100 − skew_deg × 6, 0, 100)."
        ),
        "ocr_impact": (
            "Skew greater than ~5° measurably increases word error rate; beyond "
            "15° most OCR engines fail to segment lines correctly without a "
            "deskew preprocessing step."
        ),
        "ideal_range": "Score ≥ 70 (skew ≤ 5°). Excellent: skew ≤ 1.7°.",
    },
}
