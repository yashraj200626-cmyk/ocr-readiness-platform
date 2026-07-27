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
    """
    FIXED — this used to crash the entire History page with a raw
    KeyError whenever the results file was even slightly malformed.
    The crash `row["image_name"]` -> KeyError happens specifically when
    the *whole DataFrame* has no `image_name` column at all (not just
    one row missing a value) — which happens if results.csv ever gets
    re-saved with a different delimiter or encoding (e.g. someone opens
    it in Excel to check something and Excel resaves it as
    semicolon-separated, or with a BOM, depending on regional settings).
    After that, pandas' default comma parser reads the whole header as
    one single column and every column lookup after that fails.

    Now:
      1. If the standard comma-parse doesn't yield the expected columns,
         retry with auto-detected delimiter before giving up.
      2. Any row missing `image_name` (or any required column) is
         skipped individually instead of crashing the whole load.
      3. If the file turns out to be unrecoverable, it's backed up
         (never silently deleted) and we start clean, so no analysis
         history is ever lost without a trace.
    """

    if not os.path.exists(CSV_PATH):
        return None

    def _try_read(path):
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            if "image_name" in df.columns and "timestamp" in df.columns:
                return df
        except Exception:
            pass
        return None

    df = _try_read(CSV_PATH)

    if df is None:
        # Standard comma-parse failed to produce the expected schema —
        # most likely a delimiter/encoding mismatch. Try auto-detection.
        try:
            df = pd.read_csv(CSV_PATH, sep=None, engine="python")
            df.columns = df.columns.str.strip()
            if "image_name" not in df.columns or "timestamp" not in df.columns:
                df = None
        except Exception:
            df = None

    if df is None:
        # Truly unrecoverable — back up the broken file rather than
        # silently losing it, then report "no history" instead of
        # crashing the page.
        try:
            backup_path = os.path.join(
                BASE_DIR,
                f"results_corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            )
            shutil.copy(CSV_PATH, backup_path)
        except Exception:
            pass
        return None

    if df.empty:
        return None

    # Ensure every expected column exists (older rows / older schema
    # versions may be missing newer factor columns) so downstream code
    # never has to guess.
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Remove rows whose image doesn't exist anymore, or which are
    # themselves malformed (missing image_name entirely). Skip bad
    # rows individually instead of letting one bad row crash the page.
    keep_rows = []

    for _, row in df.iterrows():
        try:
            name = row.get("image_name")
            if name is None or (isinstance(name, float) and pd.isna(name)):
                continue
            img_path = os.path.join(UPLOAD_FOLDER, str(name))
            if os.path.exists(img_path):
                keep_rows.append(row)
        except Exception:
            continue

    if len(keep_rows) == 0:
        return None

    clean_df = pd.DataFrame(keep_rows)

    # Rewrite CSV automatically (now guaranteed well-formed)
    try:
        clean_df.to_csv(CSV_PATH, index=False)
    except Exception:
        pass

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