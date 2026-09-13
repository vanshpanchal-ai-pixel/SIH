"""
Terminal demo: run a real product-label photo (or front+back photo pair)
through the full pipeline (label detect/crop -> OCR -> categorize ->
extract -> rule engine) and print a clean PASS/FAIL compliance report.

Usage:
    python demo.py path/to/photo.jpg
    python demo.py path/to/front.jpg path/to/back.jpg
    python demo.py path/to/front.jpg path/to/back.jpg Food   # force category
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # so `app.*` imports work when run directly

from app.label_detector import detect_and_crop
from app.ocr import run_ocr
from app.categorize import detect_category
from app.extract import extract_fields
from app.rule_engine import load_rules, check_compliance


KNOWN_CATEGORIES = {"Food", "Cosmetics"}


def line(char="-", n=60):
    print(char * n)


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo.py <image_path> [image_path_2] [category]")
        sys.exit(1)

    args = sys.argv[1:]

    # Last arg might be a category override rather than a file path
    forced_category = None
    if args[-1] in KNOWN_CATEGORIES:
        forced_category = args[-1]
        args = args[:-1]

    image_paths = args  # 1 or 2 image paths (e.g. front + back of label)
    for p in image_paths:
        if not Path(p).exists():
            print(f"File not found: {p}")
            sys.exit(1)

    print("\nSIH26034 — Legal Metrology Compliance Checker (Demo)")
    line("=")

    print(f"[1/5] Loading image(s): {', '.join(image_paths)}")

    print("[2/5] Label detection + crop (YOLOv8 + OpenCV)...")
    ocr_text_parts = []
    for img_path in image_paths:
        cropped_path = detect_and_crop(img_path)
        if cropped_path != img_path:
            print(f"      -> {img_path}: label cropped -> {cropped_path}")
        else:
            print(f"      -> {img_path}: no trained model yet, using full photo")

        print(f"[3/5] Running OCR on {img_path}...")
        text = run_ocr(cropped_path)
        print(f"      -> {len(text)} chars extracted")
        ocr_text_parts.append(text)

    # Combine text from both sides (front + back) before extraction
    ocr_text = "\n".join(ocr_text_parts)
    print(f"\n      Combined text preview:")
    print(f"      \"{ocr_text[:200]}{'...' if len(ocr_text) > 200 else ''}\"")

    category = forced_category or detect_category(ocr_text)
    print(f"[4/5] Category: {category}"
          f"{' (auto-detected)' if not forced_category else ' (forced)'}")

    print("[5/5] Extracting fields + checking compliance...\n")
    fields = extract_fields(ocr_text, category)
    rules = load_rules(category)
    verdicts = check_compliance(fields, rules)

    line("=")
    print("COMPLIANCE REPORT")
    line("=")

    passed = sum(1 for v in verdicts.values() if v["verdict"] == "PASS")
    total = len(verdicts)

    for field, result in verdicts.items():
        status = result["verdict"]
        mark = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{mark}] {field}")
        print(f"         Value : {result['value']}")
        print(f"         Clause: {result['clause']}")
        line()

    print(f"\nResult: {passed}/{total} declarations compliant")
    if passed == total:
        print("Overall: COMPLIANT")
    else:
        print("Overall: NON-COMPLIANT — missing/incorrect declarations above")
    print()


if __name__ == "__main__":
    main()