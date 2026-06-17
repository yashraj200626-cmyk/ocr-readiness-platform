# OCR Readiness Evaluation Platform
**SNLP Department**
Team: Yash (Lead), Vivek, Mansi, Krish, Tanusha

---

## Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Set Tesseract path
Open `app.py` and find this section near the top (after imports):

```python
# ── TESSERACT PATH — update this to your installation ──
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

Uncomment those 2 lines and set the correct path.
Find your path by running in terminal:
```
where tesseract        # Windows
which tesseract        # Mac / Linux
```

### Step 3 — Start team APIs
Make sure your teammates have their APIs running:
| Member | Command (example)           | Port |
|--------|-----------------------------|------|
| Vivek  | `uvicorn main:app --port 8001` | 8001 |
| Mansi  | `uvicorn main:app --port 8000` | 8000 |
| Krish  | `uvicorn main:app --port 8002` | 8002 |
| Tanusha| `uvicorn main:app --port 9001` | 9001 |

> If any API is offline, the platform automatically uses local fallback — it still works.

### Step 4 — Run the platform
```bash
streamlit run app.py
```

---

## Project Structure
```
ocr_platform/
├── app.py              # Main Streamlit application (4 pages)
├── factors.py          # All 8 factor algorithms (local fallback)
├── api_integration.py  # Calls team APIs, falls back if offline
├── factor_info.py      # Educational content for About page
├── storage.py          # CSV storage + correlation analysis
├── report.py           # PDF report generator
├── requirements.txt    # Python dependencies
└── results.csv         # Auto-created after first analysis
```

---

## Team API Details
| Member | Endpoint                              | Returns                                      |
|--------|---------------------------------------|----------------------------------------------|
| Vivek  | POST http://127.0.0.1:8001/analyze-image | stroke_width, text_density               |
| Mansi  | POST http://localhost:8000/analyze    | blur_score, contrast_score                   |
| Krish  | POST http://localhost:8002/scores     | matra_continuity_score, zone_integrity_score |
| Tanusha| POST http://127.0.0.1:9001/analyze    | connected_component_stability_score, skew_penalty_score |

---

## OCR Readiness Score Weights
| Factor              | Weight |
|---------------------|--------|
| Blur                | 15%    |
| Noise               | 11%    |
| Resolution          | 11%    |
| Contrast            | 11%    |
| CC Stability        | 9%     |
| Skew Penalty        | 9%     |
| Stroke Width        | 7%     |
| Text Density        | 7%     |
| Matra Continuity    | 7%     |
| Zone Integrity      | 3%     |
