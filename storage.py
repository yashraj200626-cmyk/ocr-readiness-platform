"""
CSV storage and history management for OCR Readiness Platform
"""

import csv
import os
import shutil
from datetime import datetime
from io import StringIO
from typing import Dict, Any, Optional

import pandas as pd

BASE_DIR = os.path.dirname(__file__)

CSV_PATH = os.path.join(BASE_DIR, "results.csv")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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


def save_uploaded_image(image, image_name):
    """
    Save uploaded image into uploads folder.
    """

    path = os.path.join(UPLOAD_FOLDER, image_name)

    if not os.path.exists(path):
        image.save(path)

    return path


def save_result(
    image_name: str,
    factor_results: Dict[str, Any],
    ocr_readiness: float,
    ocr_confidence: Optional[float],
):

    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=COLUMNS)

        if not file_exists:
            writer.writeheader()

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_name": image_name,
            "ocr_readiness_score": ocr_readiness,
            "ocr_confidence": "" if ocr_confidence is None else ocr_confidence,
        }

        for col in FACTOR_COLS:
            row[col] = factor_results.get(col, {}).get("score", "")

        writer.writerow(row)


def load_results():

    if not os.path.exists(CSV_PATH):
        return None

    try:
        df = pd.read_csv(CSV_PATH)

    except Exception:
        return None

    if df.empty:
        return None

    df.columns = df.columns.str.strip()

    # Remove rows whose image doesn't exist anymore
    keep_rows = []

    for _, row in df.iterrows():

        img_path = os.path.join(UPLOAD_FOLDER, str(row["image_name"]))

        if os.path.exists(img_path):
            keep_rows.append(row)

    if len(keep_rows) == 0:
        return None

    clean_df = pd.DataFrame(keep_rows)

    # Rewrite CSV automatically
    clean_df.to_csv(CSV_PATH, index=False)

    return clean_df


def compute_correlations():

    df = load_results()

    if df is None:
        return None

    required = FACTOR_COLS + ["ocr_confidence"]

    if not all(col in df.columns for col in required):
        return None

    df = df.dropna(subset=["ocr_confidence"])

    if len(df) < 3:
        return None

    numeric = df[required].apply(pd.to_numeric, errors="coerce")

    corr = numeric.corr()["ocr_confidence"].drop("ocr_confidence")

    return corr.sort_values(ascending=False)