"""Lightweight PaddleOCR wrapper for processed expiration-date ROIs."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any


os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def _empty_result(image_path: str, warning: str) -> dict[str, Any]:
    return {
        "success": False,
        "raw_text": "",
        "text": "",  # Compatibility alias for the current pipeline.
        "lines": [],
        "confidence": 0.0,
        "source_image_path": image_path,
        "warnings": [warning],
    }


@lru_cache(maxsize=1)
def _get_ocr_engine() -> Any:
    repository_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault(
        "PADDLE_PDX_CACHE_HOME",
        str(repository_root / "outputs" / "cache" / "paddlex"),
    )

    from paddleocr import PaddleOCR

    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        enable_mkldnn=False,
    )


def _line(text: Any, confidence: Any, bbox: Any = None) -> dict[str, Any] | None:
    normalized_text = str(text).strip()
    if not normalized_text:
        return None

    try:
        normalized_confidence = float(confidence)
    except (TypeError, ValueError):
        normalized_confidence = 0.0

    result = {
        "text": normalized_text,
        "confidence": normalized_confidence,
    }
    if bbox is not None:
        result["bbox"] = bbox.tolist() if hasattr(bbox, "tolist") else bbox
    return result


def _result_mapping(result: Any) -> dict[str, Any] | None:
    if isinstance(result, dict):
        return result

    json_result = getattr(result, "json", None)
    if callable(json_result):
        json_result = json_result()
    if isinstance(json_result, dict):
        return json_result.get("res", json_result)
    return None


def _extract_modern_lines(result: Any) -> list[dict[str, Any]] | None:
    mapping = _result_mapping(result)
    if mapping is None:
        return None

    texts = mapping.get("rec_texts")
    scores = mapping.get("rec_scores")
    boxes = mapping.get("rec_boxes")
    if boxes is None:
        boxes = mapping.get("dt_polys")
    if texts is None or scores is None:
        return None

    extracted_lines: list[dict[str, Any]] = []
    for index, (text, confidence) in enumerate(zip(texts, scores)):
        bbox = boxes[index] if boxes is not None and index < len(boxes) else None
        extracted_line = _line(text, confidence, bbox)
        if extracted_line:
            extracted_lines.append(extracted_line)
    return extracted_lines


def _extract_legacy_lines(result: Any) -> list[dict[str, Any]]:
    extracted_lines: list[dict[str, Any]] = []
    if not isinstance(result, (list, tuple)):
        return extracted_lines

    if (
        len(result) == 2
        and isinstance(result[1], (list, tuple))
        and len(result[1]) >= 2
        and isinstance(result[1][0], str)
    ):
        extracted_line = _line(result[1][0], result[1][1], result[0])
        if extracted_line:
            extracted_lines.append(extracted_line)
        return extracted_lines

    for item in result:
        extracted_lines.extend(_extract_legacy_lines(item))
    return extracted_lines


def _extract_lines(results: Any) -> list[dict[str, Any]]:
    if not isinstance(results, (list, tuple)):
        results = [results]

    extracted_lines: list[dict[str, Any]] = []
    for result in results:
        modern_lines = _extract_modern_lines(result)
        if modern_lines is not None:
            extracted_lines.extend(modern_lines)
        else:
            extracted_lines.extend(_extract_legacy_lines(result))
    return extracted_lines


def run_ocr(image_path: str) -> dict[str, Any]:
    """Recognize text in a processed ROI and return structured OCR output."""
    mock_text = os.getenv("MOCK_OCR_TEXT")
    if mock_text:
        return {
            "success": True,
            "raw_text": mock_text,
            "text": mock_text,  # Compatibility alias for the current pipeline.
            "lines": [{"text": mock_text, "confidence": 1.0}],
            "confidence": 1.0,
            "source_image_path": image_path,
            "warnings": ["Using MOCK_OCR_TEXT instead of PaddleOCR inference."],
        }

    if not Path(image_path).is_file():
        return _empty_result(image_path, f"OCR source image does not exist: {image_path}")

    try:
        engine = _get_ocr_engine()
    except ModuleNotFoundError:
        return _empty_result(image_path, "PaddleOCR is not installed. Install the paddleocr package.")
    except Exception as error:
        return _empty_result(image_path, f"Unable to initialize PaddleOCR: {error}")

    try:
        if hasattr(engine, "predict"):
            results = engine.predict(image_path)
        else:
            results = engine.ocr(image_path, cls=False)
        lines = _extract_lines(results)
    except Exception as error:
        return _empty_result(image_path, f"PaddleOCR inference failed: {error}")

    warnings: list[str] = []
    if not lines:
        warnings.append("PaddleOCR completed but did not recognize any text.")

    raw_text = "\n".join(line["text"] for line in lines)
    confidence = (
        sum(line["confidence"] for line in lines) / len(lines)
        if lines
        else 0.0
    )
    return {
        "success": True,
        "raw_text": raw_text,
        "text": raw_text,  # Compatibility alias for the current pipeline.
        "lines": lines,
        "confidence": confidence,
        "source_image_path": image_path,
        "warnings": warnings,
    }
