# SIH26034 — Phase 1 Prototype

Covers the Phase-1 MVP loop from the project doc: **image upload → OCR →
regex field extraction → config-driven rule engine → PASS/FAIL JSON**,
for the `Food` category, 6 mandatory fields.

## What's here

```
app/
  rules.json      # Stage 7 — LM(PC) Rules 2011, config not code
  extract.py       # Stage 5 — regex field extraction (tested, works standalone)
  rule_engine.py   # Stage 7 — check_compliance(), tested with PASS + FAIL cases
  ocr.py           # Stage 4 — PaddleOCR primary, Tesseract fallback
  main.py          # FastAPI app: /scan (image) and /scan-text (debug, no OCR)
  requirements.txt
```

`extract.py` and `rule_engine.py` have **no external dependencies** — they
already ran and passed locally (see the terminal output when this was
built). `ocr.py` and `main.py` need packages installed (see below).

## Setup (on your own machine — this environment has no internet)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# PaddleOCR install can be slow/fiddly on some machines. If it gives you
# trouble, comment it out of requirements.txt and use the Tesseract
# fallback instead:
#   pip install pytesseract pillow
#   sudo apt-get install tesseract-ocr   (Linux)
#   brew install tesseract               (Mac)
```

## Run it

```bash
uvicorn app.main:app --reload --port 8000
```

Test without needing a real label photo yet (debug endpoint, skips OCR):

```bash
curl -X POST "http://localhost:8000/scan-text?category=Food" \
     -H "Content-Type: application/json" \
     -d '"MRP Rs. 45.00  Net Qty: 100g  Mfg Date: 05/2026 FSSAI Lic No. 12345678901234 Manufactured by: XYZ Foods Pvt Ltd, Gurgaon Consumer Care: 1800-123-4567"'
```

Test with a real photo once OCR is installed:

```bash
curl -X POST "http://localhost:8000/scan?category=Food" \
     -F "file=@sample_label.jpg"
```

Both return the Section-7 API contract shape:
```json
{
  "MRP": {"value": "Rs. 45.00", "verdict": "PASS", "clause": "..."},
  "Mfg_Date": {"value": null, "verdict": "FAIL - missing", "clause": "..."}
}
```

## Immediate next steps, in priority order (per Section 12/13 of the doc)

1. **Take 5–10 real label photos** (phone camera is fine) and run them
   through `/scan` to see how real OCR output differs from the clean
   sample text used here — this is where you'll spend most of your
   regex-tuning time. Real OCR is noisy (line breaks, misreads,
   inconsistent spacing); expect to adjust the patterns in `extract.py`.
2. **Hand `main.py`'s contract to your teammate today** — they can build
   the FastAPI routes/DB schema/frontend against this JSON shape without
   waiting on OCR to be perfect.
3. Once the core loop works end-to-end on real photos, move to the
   **font-size checker** (Stage 8, your "wow factor") — new module,
   doesn't touch this code.
4. YOLOv8 label detection and DistilBERT categorization come after that,
   per your own priority ordering — both need datasets/training time
   that's better spent once the core loop is proven.

## Known limitations of this scaffold (by design, for Phase 1)

- No label detection/crop step (Stage 3) — assumes you feed it an
  already-cropped PDP photo. Add YOLOv8 later per your Phase plan.
- Only the `Food` category is fully fleshed out in `rules.json`;
  `Cosmetics` is a partial stub — extend the JSON, not the code, to add
  more fields/categories.
- Categorization is not implemented — `category` is passed manually as
  a query param for now (matches your documented keyword-match/DistilBERT
  fallback slot; wire it in whenever you're ready).
- No PDF report generation (Stage 9/ReportLab) yet — that's a clean next
  addition once the JSON verdict shape is stable.
