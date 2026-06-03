"""Run the expiration-date pipeline over a directory of images."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
DEFAULT_LIMIT = 10
DEFAULT_MOCK_TEXT = "EXP 2027-12-31"
SUMMARY_COLUMNS = (
    "image_path",
    "success",
    "status",
    "detection_success",
    "ocr_success",
    "parsed_success",
    "parsed_date",
    "selected_raw",
    "ocr_text",
    "detection_confidence",
    "ocr_confidence",
    "roi_path",
    "processed_roi_path",
    "warnings",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch pipeline tests over an image directory.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum number of images to process (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--use-mock-ocr",
        action="store_true",
        help="Use MOCK_OCR_TEXT instead of running PaddleOCR inference.",
    )
    parser.add_argument(
        "--mock-text",
        default=None,
        help="OCR text used for mock inference. Providing this option enables mock OCR.",
    )
    return parser.parse_args()


def _list_images(image_dir: Path, limit: int) -> list[Path]:
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images[:limit]


def _result_filename(index: int, image_path: Path) -> str:
    return f"{index:04d}_{image_path.stem}.json"


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_path": result.get("image_path"),
        "success": result.get("success"),
        "status": result.get("status"),
        "detection_success": result.get("detection_success"),
        "ocr_success": bool(result.get("ocr_text")),
        "parsed_success": result.get("parsed_date") is not None,
        "parsed_date": result.get("parsed_date"),
        "selected_raw": result.get("selected_raw"),
        "ocr_text": result.get("ocr_text"),
        "detection_confidence": result.get("detection_confidence"),
        "ocr_confidence": result.get("ocr_confidence"),
        "roi_path": result.get("roi_path"),
        "processed_roi_path": result.get("processed_roi_path"),
        "warnings": " | ".join(str(warning) for warning in result.get("warnings", [])),
    }


def _write_summary(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be greater than 0")

    image_dir = args.image_dir.resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    images = _list_images(image_dir, args.limit)
    if not images:
        raise ValueError(f"No supported images found in: {image_dir}")

    output_dir = REPOSITORY_ROOT / "outputs" / "predictions" / "pipeline_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    previous_mock_text = os.environ.get("MOCK_OCR_TEXT")
    mock_text_override = args.mock_text if args.mock_text is not None else DEFAULT_MOCK_TEXT
    if args.use_mock_ocr or args.mock_text is not None:
        os.environ["MOCK_OCR_TEXT"] = mock_text_override

    print(f"MOCK_OCR_TEXT active: {bool(os.getenv('MOCK_OCR_TEXT'))}")

    from ai_engine.pipeline.pipeline import run_pipeline

    status_counts: Counter[str] = Counter()
    summary_rows: list[dict[str, Any]] = []
    try:
        for index, image_path in enumerate(images, 1):
            result = run_pipeline(str(image_path))
            status_counts[str(result.get("status", "needs_review"))] += 1
            summary_rows.append(_summary_row(result))

            result_path = output_dir / _result_filename(index, image_path)
            result_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            print(f"[{index}/{len(images)}] {image_path.name}: {result.get('status')}")
    finally:
        if args.use_mock_ocr or args.mock_text is not None:
            if previous_mock_text is None:
                os.environ.pop("MOCK_OCR_TEXT", None)
            else:
                os.environ["MOCK_OCR_TEXT"] = previous_mock_text

    summary_path = output_dir / "summary.csv"
    _write_summary(summary_path, summary_rows)

    print("Status counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print(f"Summary CSV: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
