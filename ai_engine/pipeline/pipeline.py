"""End-to-end expiration-date recognition pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ai_engine.detection.predict import detect_expiration_region
from ai_engine.ocr.paddle_ocr import run_ocr
from ai_engine.ocr.parse_date import parse_expiration_date
from ai_engine.ocr.preprocess import preprocess_roi
from ai_engine.pipeline.expiration_logic import evaluate_expiration


def _base_result(image_path: str) -> dict[str, Any]:
    return {
        "success": False,
        "image_path": str(image_path),
        "status": "needs_review",
        "days_remaining": None,
        "parsed_date": None,
        "selected_raw": None,
        "detected_format": None,
        "ocr_text": "",
        "ocr_confidence": 0.0,
        "detection_success": False,
        "detection_confidence": None,
        "bbox": None,
        "expanded_bbox": None,
        "roi_path": None,
        "processed_roi_path": None,
        "candidate_results": [],
        "warnings": [],
    }


def _validate_stage_result(
    result: dict[str, Any],
    stage_result: Any,
    stage_name: str,
) -> bool:
    if isinstance(stage_result, dict):
        return True
    result["warnings"].append(f"{stage_name} returned an invalid result.")
    return False


def _extend_warnings(result: dict[str, Any], stage_result: dict[str, Any]) -> None:
    warnings = stage_result.get("warnings")
    if isinstance(warnings, list):
        result["warnings"].extend(str(warning) for warning in warnings)


def _json_value(value: Any) -> Any:
    """Convert stage output into JSON-serializable values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    return str(value)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    try:
        parsed_value = None if value is None else float(value)
    except (TypeError, ValueError):
        return None
    return parsed_value if parsed_value is None or math.isfinite(parsed_value) else None


def _optional_list(value: Any) -> list | None:
    return _json_value(value) if isinstance(value, (list, tuple)) else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _ocr_attempt_rank(attempt: dict[str, Any]) -> tuple[bool, float, float]:
    return (
        attempt["parsed_success"],
        attempt["parser_confidence"],
        attempt["ocr_confidence"],
    )


def _run_ocr_attempt(
    variant_name: str,
    variant_path: str,
) -> dict[str, Any]:
    attempt = {
        "variant": variant_name,
        "image_path": variant_path,
        "ocr_success": False,
        "ocr_text": "",
        "ocr_confidence": 0.0,
        "parsed_success": False,
        "parsed_date": None,
        "selected_raw": None,
        "detected_format": None,
        "parser_confidence": 0.0,
        "date_candidates": [],
        "warnings": [],
    }

    try:
        ocr_result = run_ocr(variant_path)
    except Exception as error:
        attempt["warnings"].append(f"OCR failed: {error}")
        return attempt
    if not isinstance(ocr_result, dict):
        attempt["warnings"].append("OCR returned an invalid result.")
        return attempt

    warnings = ocr_result.get("warnings")
    if isinstance(warnings, list):
        attempt["warnings"].extend(str(warning) for warning in warnings)
    attempt["ocr_success"] = ocr_result.get("success") is True
    attempt["ocr_text"] = str(ocr_result.get("raw_text") or "")
    attempt["ocr_confidence"] = _optional_float(ocr_result.get("confidence")) or 0.0
    if not attempt["ocr_success"]:
        attempt["warnings"].append("OCR did not succeed.")
        return attempt
    if not attempt["ocr_text"].strip():
        attempt["warnings"].append("OCR completed but returned empty text.")
        return attempt

    try:
        parsed_result = parse_expiration_date(attempt["ocr_text"])
    except Exception as error:
        attempt["warnings"].append(f"Expiration-date parsing failed: {error}")
        return attempt
    if not isinstance(parsed_result, dict):
        attempt["warnings"].append("Expiration-date parsing returned an invalid result.")
        return attempt

    warnings = parsed_result.get("warnings")
    if isinstance(warnings, list):
        attempt["warnings"].extend(str(warning) for warning in warnings)
    attempt["parsed_success"] = parsed_result.get("success") is True
    attempt["parsed_date"] = _optional_string(parsed_result.get("selected_date"))
    attempt["selected_raw"] = _optional_string(parsed_result.get("selected_raw"))
    attempt["detected_format"] = _optional_string(parsed_result.get("detected_format"))
    attempt["parser_confidence"] = _optional_float(parsed_result.get("confidence")) or 0.0
    candidates = parsed_result.get("candidates")
    attempt["date_candidates"] = _json_value(candidates) if isinstance(candidates, list) else []
    if not attempt["parsed_success"]:
        attempt["warnings"].append("Expiration-date parsing did not succeed.")
    return attempt


def run_pipeline(image_path: str) -> dict[str, Any]:
    """Run detection, preprocessing, OCR, date parsing, and expiration evaluation."""
    result = _base_result(image_path)
    if not Path(image_path).is_file():
        result["warnings"].append(f"Input image does not exist: {image_path}")
        return result

    try:
        detection_result = detect_expiration_region(image_path)
    except Exception as error:
        result["warnings"].append(f"Expiration-region detection failed: {error}")
        return result

    if not _validate_stage_result(result, detection_result, "Expiration-region detection"):
        return result
    _extend_warnings(result, detection_result)
    result["detection_success"] = detection_result.get("detection_success") is True
    result["bbox"] = _optional_list(detection_result.get("bbox"))
    result["expanded_bbox"] = _optional_list(detection_result.get("expanded_bbox"))
    result["detection_confidence"] = _optional_float(detection_result.get("confidence"))
    result["roi_path"] = _optional_string(detection_result.get("roi_path"))
    if not result["detection_success"]:
        result["warnings"].append("Expiration-date region detection did not succeed.")
        return result
    if not result["roi_path"]:
        result["warnings"].append("Detection succeeded but did not return an ROI path.")
        return result

    try:
        preprocessing_result = preprocess_roi(result["roi_path"])
    except Exception as error:
        result["warnings"].append(f"ROI preprocessing failed: {error}")
        return result

    if not _validate_stage_result(result, preprocessing_result, "ROI preprocessing"):
        return result
    _extend_warnings(result, preprocessing_result)
    result["processed_roi_path"] = _optional_string(preprocessing_result.get("processed_image_path"))
    if not result["processed_roi_path"]:
        result["warnings"].append("ROI preprocessing did not return a processed image path.")
        return result

    try:
        threshold_output_path = str(
            Path(result["roi_path"]).with_name(f"{Path(result['roi_path']).stem}_thresholded.png")
        )
        threshold_result = preprocess_roi(
            result["roi_path"],
            output_path=threshold_output_path,
            threshold=True,
        )
    except Exception as error:
        result["warnings"].append(f"Thresholded ROI preprocessing failed: {error}")
        threshold_result = None

    variants = [
        ("original_roi", result["roi_path"]),
        ("default_preprocessed", result["processed_roi_path"]),
    ]
    if isinstance(threshold_result, dict):
        _extend_warnings(result, threshold_result)
        threshold_path = _optional_string(threshold_result.get("processed_image_path"))
        if threshold_path:
            variants.append(("thresholded", threshold_path))
        else:
            result["warnings"].append("Thresholded ROI preprocessing did not return an image path.")
    elif threshold_result is not None:
        result["warnings"].append("Thresholded ROI preprocessing returned an invalid result.")

    attempts = [_run_ocr_attempt(name, path) for name, path in variants]
    result["candidate_results"] = _json_value(attempts)
    selected_attempt = max(attempts, key=_ocr_attempt_rank)
    result["ocr_text"] = selected_attempt["ocr_text"]
    result["ocr_confidence"] = selected_attempt["ocr_confidence"]
    result["parsed_date"] = selected_attempt["parsed_date"]
    result["selected_raw"] = selected_attempt["selected_raw"]
    result["detected_format"] = selected_attempt["detected_format"]
    result["warnings"].extend(selected_attempt["warnings"])

    ocr_success = selected_attempt["ocr_success"]
    parsed_success = selected_attempt["parsed_success"]
    if not ocr_success:
        result["warnings"].append("No OCR variant produced readable text.")
        return result
    if not parsed_success:
        result["warnings"].append("No OCR variant produced a valid expiration date.")
        return result

    try:
        expiration_result = evaluate_expiration(result["parsed_date"])
    except Exception as error:
        result["warnings"].append(f"Expiration evaluation failed: {error}")
        return result

    if not _validate_stage_result(result, expiration_result, "Expiration evaluation"):
        return result
    _extend_warnings(result, expiration_result)
    status = expiration_result.get("status")
    if status not in {"valid", "near_expiry", "expired"}:
        result["warnings"].append("Expiration evaluation did not produce a final status.")
        return result
    result["status"] = status
    result["days_remaining"] = _optional_int(expiration_result.get("days_remaining"))
    result["success"] = (
        result["detection_success"]
        and ocr_success
        and parsed_success
        and result["status"] != "needs_review"
    )
    return result
