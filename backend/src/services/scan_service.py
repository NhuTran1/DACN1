from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from ai_engine.pipeline.pipeline import run_pipeline

UPLOAD_DIR = Path("uploads")


async def process_uploaded_image(file: UploadFile) -> dict[str, Any]:
    """Save an uploaded image and run the blocking AI pipeline off the event loop."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    extension = Path(file.filename or "").suffix.lower()
    file_path = UPLOAD_DIR / f"{uuid4().hex}{extension}"
    content = await file.read()
    if not content:
        return _error_result(file_path, "Uploaded image is empty.")

    try:
        file_path.write_bytes(content)
    except OSError as error:
        return _error_result(file_path, f"Could not save uploaded image: {error}")

    try:
        result = await run_in_threadpool(run_pipeline, str(file_path))
    except Exception as error:
        return _error_result(file_path, f"AI pipeline failed: {error}")

    if not isinstance(result, dict):
        return _error_result(file_path, "AI pipeline returned an invalid result.")
    return result


def _error_result(image_path: Path, warning: str) -> dict[str, Any]:
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
        "warnings": [warning],
    }
