"""
OCR wrapper. Tries PaddleOCR first (better multilingual/Indic support per
Section 8), falls back to Tesseract if PaddleOCR isn't installed/available.

NOTE: Neither engine is installed in the dev sandbox this scaffold was
generated in (no internet access there). Install locally with:

    pip install paddleocr paddlepaddle
    # or, for the fallback:
    pip install pytesseract pillow
    # + system package: sudo apt-get install tesseract-ocr

This file is written so main.py can import `run_ocr` right now — it will
raise a clear RuntimeError telling you what to install if neither engine
is available, rather than failing silently.
"""

from typing import Optional

_paddle_engine = None


def _get_paddle_engine():
    global _paddle_engine
    if _paddle_engine is None:
        from paddleocr import PaddleOCR  # lazy import
        _paddle_engine = PaddleOCR(use_angle_cls=True, lang="en")
    return _paddle_engine


def _run_paddle(image_path: str) -> Optional[str]:
    try:
        engine = _get_paddle_engine()
    except ImportError:
        return None
    result = engine.ocr(image_path, cls=True)
    lines = []
    for page in result:
        for line in page:
            lines.append(line[1][0])  # (box, (text, confidence))
    return " ".join(lines)


def _run_tesseract(image_path: str) -> Optional[str]:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None
    return pytesseract.image_to_string(Image.open(image_path))


def run_ocr(image_path: str) -> str:
    """Run OCR on an image, PaddleOCR first, Tesseract fallback."""
    text = _run_paddle(image_path)
    if text:
        return text

    text = _run_tesseract(image_path)
    if text:
        return text

    raise RuntimeError(
        "No OCR engine available. Install one of:\n"
        "  pip install paddleocr paddlepaddle\n"
        "  pip install pytesseract pillow  (+ apt-get install tesseract-ocr)"
    )
