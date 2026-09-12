"""
Stage 2-3 of the pipeline: detect the product label region in a raw photo
and crop it before OCR.

Until a trained YOLOv8 model (best.pt) is available, this module falls
back to returning the original image untouched — so /scan keeps working
end-to-end today, and swapping in the real model later requires no
changes anywhere else in the codebase.
"""

from pathlib import Path
import cv2

_MODEL_PATH = Path(__file__).parent / "models" / "label_yolov8.pt"
_model = None
_model_load_attempted = False


def _get_model():
    """Lazy-load the YOLOv8 model. Returns None if not trained/available yet."""
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True
    if not _MODEL_PATH.exists():
        return None
    try:
        from ultralytics import YOLO
        _model = YOLO(str(_MODEL_PATH))
    except Exception:
        _model = None
    return _model


def _preprocess(image):
    """OpenCV cleanup applied to the cropped label before OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    return denoised


def detect_and_crop(image_path: str) -> str:
    """
    Detect the label in the photo at image_path, crop it, run OpenCV
    preprocessing, and save the result to a temp file. Returns the path
    to the (possibly cropped) image to feed into OCR.

    Falls back to the original image_path untouched if no trained model
    is available yet (Phase-1 behaviour).
    """
    model = _get_model()
    if model is None:
        return image_path  # no model yet — OCR runs on the full photo

    image = cv2.imread(image_path)
    if image is None:
        return image_path

    results = model(image)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return image_path  # nothing detected — fall back to full image

    # Take the highest-confidence detection
    best = boxes[boxes.conf.argmax()]
    x1, y1, x2, y2 = map(int, best.xyxy[0])
    cropped = image[y1:y2, x1:x2]

    processed = _preprocess(cropped)

    out_path = str(Path(image_path).with_suffix("")) + "_cropped.jpg"
    cv2.imwrite(out_path, processed)
    return out_path