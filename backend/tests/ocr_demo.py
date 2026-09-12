"""
Minimal OCR-only demo. Just photo in -> extracted text out.
No rule engine, no FastAPI, no regex — pure "does OCR work" check.

Usage:
    python ocr_demo.py path\to\label_photo.jpg
"""

import sys
from paddleocr import PaddleOCR


def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr_demo.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("Loading PaddleOCR (first run downloads model files, needs internet)...")
    # enable_mkldnn=False works around a known CPU-backend crash
    # ("ConvertPirAttribute2RuntimeAttribute ... onednn_instruction.cc")
    # seen on some CPUs with paddlepaddle 3.x's default oneDNN acceleration.
    ocr = PaddleOCR(use_textline_orientation=True, lang="en", enable_mkldnn=False)

    print(f"Running OCR on: {image_path}")
    result = ocr.predict(image_path)

    print("\n--- Extracted text ---")
    for res in result:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        for text, score in zip(texts, scores):
            print(f"[{score:.2f}] {text}")

    print("\n--- Full combined text (what would be fed to regex extraction) ---")
    all_text = " ".join(
        t for res in result for t in res.get("rec_texts", [])
    )
    print(all_text)


if __name__ == "__main__":
    main()