"""
Field extraction from raw OCR text using regex (primary method).
Deterministic and explainable, per the project's design decision to avoid
a single end-to-end LLM/VLM call for numeric legal fields (MRP, net qty).

NER (spaCy) for free-text address extraction is a documented lower-priority
backup and is intentionally NOT implemented in this Phase-1 scaffold.
"""

import re

# Each extractor returns the *normalized* string that should match the
# corresponding pattern in rules.json, or None if not found.

def extract_mrp(text: str):
    m = re.search(r'MRP\D{0,10}?(Rs\.?\s*[\d,]+(?:\.\d{1,2})?)', text, re.IGNORECASE)
    if not m:
        return None
    return re.sub(r'\s+', ' ', m.group(1)).replace(',', '').strip()


def extract_net_quantity(text: str):
    m = re.search(r'Net\s*Qty:?\s*(\d+(?:\.\d+)?\s*(?:g|kg|ml|l))', text, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).replace(' ', '').lower()


def extract_mfg_date(text: str):
    m = re.search(r'Mfg\.?\s*Date:?\s*(\d{2}/\d{4})', text, re.IGNORECASE)
    if not m:
        return None
    return m.group(1)


def extract_fssai(text: str):
    m = re.search(r'FSSAI\D{0,10}?(\d{14})', text, re.IGNORECASE)
    return m.group(1) if m else None


# Labels that mark the start of the *next* field on a crowded label — used as
# lookahead stop-points so one field's regex doesn't swallow the next field's
# text when OCR gives us one unbroken line with no punctuation between them.
_NEXT_FIELD_LOOKAHEAD = r'(?=\s*(?:MRP|Net\s*Qty|Mfg\.?\s*Date|FSSAI|Consumer Care|Customer Care|\.|\n|$))'


def extract_manufacturer_name(text: str):
    pattern = r'(?:Manufactured|Marketed|Packed)\s*by:?\s*(.+?)' + _NEXT_FIELD_LOOKAHEAD
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip().rstrip(',') if m else None


def extract_consumer_care(text: str):
    pattern = r'(?:Consumer Care|Customer Care)[:\s]*(.+?)' + _NEXT_FIELD_LOOKAHEAD
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip().rstrip(',') if m else None


EXTRACTORS = {
    "MRP": extract_mrp,
    "Net_Quantity": extract_net_quantity,
    "Mfg_Date": extract_mfg_date,
    "FSSAI_No": extract_fssai,
    "Manufacturer_Name": extract_manufacturer_name,
    "Consumer_Care": extract_consumer_care,
}


def extract_fields(ocr_text: str, category: str = "Food") -> dict:
    """Run all extractors relevant to a category and return {field: value|None}."""
    return {field: fn(ocr_text) for field, fn in EXTRACTORS.items()}


if __name__ == "__main__":
    sample_text = (
        "MRP Rs. 45.00 (Incl. of all taxes)  Net Qty: 100g  Mfg Date: 05/2026 "
        "FSSAI Lic No. 12345678901234 "
        "Manufactured by: XYZ Foods Pvt Ltd, Plot 12, Sector 5, Gurgaon "
        "Consumer Care: 1800-123-4567, care@xyzfoods.com"
    )
    fields = extract_fields(sample_text)
    for k, v in fields.items():
        print(f"{k:20s} -> {v}")
