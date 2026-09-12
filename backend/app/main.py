"""
FastAPI backend. Implements the API contract agreed in Section 7 of the
project doc, so the backend/frontend teammate can build against this
response shape immediately, independent of OCR engine setup.

Run with:
    uvicorn main:app --reload --port 8000

Test with:
    curl -X POST "http://localhost:8000/scan?category=Food" \
         -F "file=@sample_label.jpg"
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel

from app.extract import extract_fields
from app.rule_engine import load_rules, check_compliance
from app.ocr import run_ocr

from app.categorize import detect_category
from app.label_detector import detect_and_crop

app = FastAPI(title="SIH26034 Legal Metrology Compliance Checker")


class FieldVerdict(BaseModel):
    value: str | None
    verdict: str
    clause: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/scan", response_model=dict[str, FieldVerdict])
async def scan_label(
    file: UploadFile = File(...),
    category: str | None = Query(None, description="Product category, e.g. Food, Cosmetics"),
):
    # 1. Save uploaded image to a temp path
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Stage 2-3: detect + crop label region (no-op until trained model is added)
        tmp_path = detect_and_crop(tmp_path)
        # 2. OCR  (Stage 4 of the pipeline — label detection/crop, Stage 3,
        #    is skipped in this Phase-1 scaffold; assumes a pre-cropped PDP photo)
        ocr_text = run_ocr(tmp_path)

        # Auto-detect category via keyword/DistilBERT if not explicitly provided
        category = category or detect_category(ocr_text)

        # 3. Field extraction (Stage 5 — regex primary)
        fields = extract_fields(ocr_text, category)

        # 4. Rule engine (Stage 7 — config-driven)
        rules = load_rules(category)
        verdicts = check_compliance(fields, rules)

        return verdicts

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/scan-text", response_model=dict[str, FieldVerdict])
async def scan_text(text: str, category: str | None = None):
    """
    Debug/dev endpoint: skip OCR entirely, pass raw text straight to
    extraction + rule engine. Useful for the backend teammate to build
    the UI/report layer before OCR is wired up, and for your own testing.
    """
    try:
        # Auto-detect category if caller didn't specify one
        category = category or detect_category(text)
        fields = extract_fields(text, category)
        rules = load_rules(category)
        return check_compliance(fields, rules)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
