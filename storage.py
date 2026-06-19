"""
CSV storage and correlation analysis for OCR Readiness Platform.
"""

import csv
import os
from io import StringIO
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

CSV_PATH = os.path.join(os.path.dirname(__file__), "results.csv")

COLUMNS = [
    "timestamp",
    "image_name",
    "noise_score",
    "resolution_score",
    "blur_score",
    "contrast_score",
    "stroke_width_score",
    "text_density_score",
    "matra_continuity_score",
    "zone_integrity_score",
    "connected_component_stability_score",
    "skew_penalty_score",
    "ocr_readiness_score",
    "ocr_confidence",
]

FACTOR_COLS = [
    "noise_score",
    "resolution_score",
    "blur_score",
    "contrast_score",
    "stroke_width_score",
    "text_density_score",
    "matra_continuity_score",
    "zone_integrity_score",
    "connected_component_stability_score",
    "skew_penalty_score",
]


def save_result(
    image_name: str,
    factor_results: Dict[str, Any],
    ocr_readiness: float,
    ocr_confidence: Optional[float],
) -> None:
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_name": image_name,
            "ocr_readiness_score": ocr_readiness,
            "ocr_confidence": ocr_confidence if ocr_confidence is not None else "",
        }
        for col in FACTOR_COLS:
            row[col] = factor_results.get(col, {}).get("score", "")
        writer.writerow(row)


def load_results() -> Optional[pd.DataFrame]:
    if not os.path.isfile(CSV_PATH):
        return None
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    cleaned_lines = []
    header_line = None
    for line in raw_lines:
        if line.startswith("<<<") or line.startswith("===") or line.startswith(">>>"):
            continue
        if line == ",".join(COLUMNS):
            header_line = line
            continue
        cleaned_lines.append(line)

    if header_line is None:
        header_line = ",".join(COLUMNS)

    csv_text = "\n".join([header_line, *cleaned_lines])
    df = pd.read_csv(StringIO(csv_text))
    df.columns = [str(col).strip() for col in df.columns]
    if df.empty:
        return None
    return df


def compute_correlations() -> Optional[pd.Series]:
    df = load_results()
    if df is None:
        return None
    required_cols = set(FACTOR_COLS + ["ocr_confidence"])
    if not required_cols.issubset(df.columns):
        return None
    df = df.dropna(subset=["ocr_confidence"])
    if len(df) < 3:
        return None
    numeric = df[FACTOR_COLS + ["ocr_confidence"]].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr()["ocr_confidence"].drop("ocr_confidence")
    return corr.sort_values(ascending=False)
