"""
Short one-line descriptions shown directly under each factor score card.
"""

SHORT_DESC = {
    "noise_score": {
        "excellent": "Very low noise — clean image",
        "good":      "Slight noise — acceptable",
        "average":   "Moderate noise in image",
        "poor":      "High noise — very noisy image",
    },
    "resolution_score": {
        "excellent": "High resolution — very clear",
        "good":      "Good resolution for OCR",
        "average":   "Low resolution — may blur text",
        "poor":      "Very low resolution — too small",
    },
    "blur_score": {
        "excellent": "Very sharp — crisp edges",
        "good":      "Mostly sharp image",
        "average":   "Noticeable blur in image",
        "poor":      "Heavily blurred image",
    },
    "contrast_score": {
        "excellent": "High contrast — text clearly visible",
        "good":      "Good contrast between text & background",
        "average":   "Low contrast — text fading",
        "poor":      "Very low contrast — text barely visible",
    },
    "stroke_width_score": {
        "excellent": "Stroke width is ideal for OCR",
        "good":      "Stroke width is acceptable",
        "average":   "Stroke width slightly off",
        "poor":      "Stroke too thin or too thick",
    },
    "text_density_score": {
        "excellent": "Optimal text coverage in image",
        "good":      "Text density is acceptable",
        "average":   "Text too sparse or too dense",
        "poor":      "Extreme text density — check image",
    },
    "matra_continuity_score": {
        "excellent": "Shirorekha fully continuous",
        "good":      "Matra mostly intact",
        "average":   "Matra breaks detected",
        "poor":      "Severely broken matra line",
    },
    "zone_integrity_score": {
        "excellent": "All 3 character zones intact",
        "good":      "Most zones present",
        "average":   "Some zones missing",
        "poor":      "Most zones are missing",
    },
    "connected_component_stability_score": {
        "excellent": "Character sizes very consistent",
        "good":      "Character sizes mostly uniform",
        "average":   "Inconsistent character sizes",
        "poor":      "Highly inconsistent character blobs",
    },
    "skew_penalty_score": {
        "excellent": "Document is well aligned",
        "good":      "Very slight tilt — acceptable",
        "average":   "Noticeable skew detected",
        "poor":      "Severely skewed document",
    },
}


def get_short_description(factor_key: str, score: float) -> str:
    if score >= 81:   level = "excellent"
    elif score >= 61: level = "good"
    elif score >= 41: level = "average"
    else:             level = "poor"
    return SHORT_DESC.get(factor_key, {}).get(level, "")
