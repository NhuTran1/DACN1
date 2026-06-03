"""Ultralytics YOLO inference wrapper for expiration-date region detection."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import cv2


LEFT_PAD_RATIO = 0.8
RIGHT_PAD_RATIO = 0.3
TOP_PAD_RATIO = 0.4
BOTTOM_PAD_RATIO = 0.4
_MODEL_CACHE_LOCK = Lock()
_MODEL_FINGERPRINTS: dict[str, tuple[int, int]] = {}


def _empty_result(image_path: str, warning: str) -> dict[str, Any]:
    return {
        "detection_success": False,
        "bbox": None,
        "expanded_bbox": None,
        "confidence": None,
        "class_name": None,
        "roi_path": None,
        "source_image_path": image_path,
        "metadata": {
            "padded_crop": False,
            "padding": None,
        },
        "warnings": [warning],
    }


def _default_model_path() -> Path:
    repository_root = Path(__file__).resolve().parents[2]
    return repository_root / "models" / "detection" / "best.pt"


@lru_cache(maxsize=4)
def _load_model(model_path: str, modified_time_ns: int, file_size: int) -> Any:
    from ultralytics import YOLO

    return YOLO(model_path)


def _get_model(model_path: Path) -> tuple[Any, str, dict[str, Any]]:
    resolved_path = str(model_path.resolve())
    model_stat = model_path.stat()
    fingerprint = (model_stat.st_mtime_ns, model_stat.st_size)

    with _MODEL_CACHE_LOCK:
        previous_fingerprint = _MODEL_FINGERPRINTS.get(resolved_path)
        model = _load_model(resolved_path, *fingerprint)
        if previous_fingerprint is None:
            cache_event = "loaded"
        elif previous_fingerprint == fingerprint:
            cache_event = "cached"
        else:
            cache_event = "reloaded"
        _MODEL_FINGERPRINTS[resolved_path] = fingerprint

    if cache_event != "cached":
        print(f"YOLO model {cache_event}: {resolved_path}")
    return model, cache_event, {
        "path": resolved_path,
        "modified_time_ns": fingerprint[0],
        "file_size": fingerprint[1],
        "cache_event": cache_event,
    }


def clear_model_cache() -> None:
    """Clear cached YOLO models so the next detection reloads weights."""
    with _MODEL_CACHE_LOCK:
        _load_model.cache_clear()
        _MODEL_FINGERPRINTS.clear()
    print("YOLO model cache cleared.")


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def detect_expiration_region(
    image_path: str,
    model_path: str | None = None,
    conf_threshold: float = 0.25,
) -> dict[str, Any]:
    """Detect and crop the highest-confidence expiration-date region."""
    source_image = Path(image_path)
    resolved_model_path = Path(model_path) if model_path else _default_model_path()

    if not 0.0 <= conf_threshold <= 1.0:
        return _empty_result(image_path, "Confidence threshold must be between 0 and 1.")
    if not source_image.is_file():
        return _empty_result(image_path, f"Source image does not exist: {image_path}")
    if not resolved_model_path.is_file():
        return _empty_result(
            image_path,
            f"Detection model does not exist: {resolved_model_path}",
        )

    image = cv2.imread(str(source_image))
    if image is None:
        return _empty_result(image_path, f"Unable to read source image: {image_path}")

    try:
        model, model_cache_event, model_metadata = _get_model(resolved_model_path)
        results = model.predict(
            source=str(source_image),
            conf=conf_threshold,
            verbose=False,
        )
    except Exception as error:
        return _empty_result(image_path, f"YOLO detection failed: {error}")

    best_box = None
    best_result = None
    best_confidence = -1.0
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence > best_confidence:
                best_box = box
                best_result = result
                best_confidence = confidence

    if best_box is None or best_result is None:
        return _empty_result(
            image_path,
            f"No expiration-date detection found above confidence threshold {conf_threshold:.2f}.",
        )

    image_height, image_width = image.shape[:2]
    raw_x1, raw_y1, raw_x2, raw_y2 = best_box.xyxy[0].tolist()
    x1 = max(0, min(image_width, round(raw_x1)))
    y1 = max(0, min(image_height, round(raw_y1)))
    x2 = max(0, min(image_width, round(raw_x2)))
    y2 = max(0, min(image_height, round(raw_y2)))
    bbox = [x1, y1, x2, y2]

    if x2 <= x1 or y2 <= y1:
        return _empty_result(image_path, f"Detected bounding box is invalid after clipping: {bbox}")

    bbox_width = x2 - x1
    bbox_height = y2 - y1
    expanded_x1 = max(0, round(x1 - bbox_width * LEFT_PAD_RATIO))
    expanded_y1 = max(0, round(y1 - bbox_height * TOP_PAD_RATIO))
    expanded_x2 = min(image_width, round(x2 + bbox_width * RIGHT_PAD_RATIO))
    expanded_y2 = min(image_height, round(y2 + bbox_height * BOTTOM_PAD_RATIO))
    expanded_bbox = [expanded_x1, expanded_y1, expanded_x2, expanded_y2]

    roi = image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
    repository_root = Path(__file__).resolve().parents[2]
    output_dir = repository_root / "outputs" / "cropped"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_image.stem}_{uuid4().hex[:8]}_roi.png"
    if not cv2.imwrite(str(output_path), roi):
        return _empty_result(image_path, f"Unable to write cropped ROI image: {output_path}")

    class_id = int(best_box.cls[0])
    return {
        "detection_success": True,
        "bbox": bbox,
        "expanded_bbox": expanded_bbox,
        "confidence": best_confidence,
        "class_name": _class_name(best_result.names, class_id),
        "roi_path": str(output_path),
        "source_image_path": image_path,
        "metadata": {
            "padded_crop": True,
            "padding": {
                "left": LEFT_PAD_RATIO,
                "right": RIGHT_PAD_RATIO,
                "top": TOP_PAD_RATIO,
                "bottom": BOTTOM_PAD_RATIO,
            },
            "model": model_metadata,
            "model_cache_event": model_cache_event,
        },
        "warnings": [],
    }
