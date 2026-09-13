"""
OCR wrapper. Tries PaddleOCR first (better multilingual/Indic support per
Section 8), falls back to Tesseract if PaddleOCR isn't installed/available.

NOTE: PaddleOCR is temporarily disabled (see run_ocr) due to a
paddlepaddle/paddlex version-compatibility issue on Windows.
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
    result = engine.ocr(image_path)
    lines = []
    for page in result:
        for line in page:
            lines.append(line[1][0])  # (box, (text, confidence))
    return " ".join(lines)


def _run_tesseract(image_path: str) -> Optional[str]:
    import pytesseract
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd = r"D:\SIH\downloads\tesseract.exe"
    return pytesseract.image_to_string(Image.open(image_path))


def run_ocr(image_path: str) -> str:
    """Run OCR on an image. Tesseract-only for now — PaddleOCR disabled
    due to a paddlepaddle/paddlex version-compatibility issue on Windows
    (tracked separately; re-enable by uncommenting the _run_paddle call)."""
    # text = _run_paddle(image_path)   # disabled — see note above
    # if text:
    #     return text

    text = _run_tesseract(image_path)
    if text:
        return text

    raise RuntimeError(
        "No OCR engine available. Install one of:\n"
        "  pip install paddleocr paddlepaddle\n"
        "  pip install pytesseract pillow  (+ apt-get install tesseract-ocr)"
    )