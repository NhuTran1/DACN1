"""Convert the ExpDate Products datasets to YOLO detection format."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLASS_ID = 0
CLASS_NAME = "expiration_date"
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SEED = 42


@dataclass(frozen=True)
class DatasetSource:
    name: str
    images_dir: Path
    annotations_path: Path


@dataclass(frozen=True)
class ImageRecord:
    source: DatasetSource
    filename: str
    annotation: dict[str, Any]


@dataclass(frozen=True)
class SelectedTarget:
    annotation: dict[str, Any]
    source_class: str
    reason: str


@dataclass
class Summary:
    train_images: int = 0
    val_images: int = 0
    test_images: int = 0
    images_using_exp_boxes: int = 0
    images_using_date_near_due_fallback: int = 0
    images_using_single_date_fallback: int = 0
    images_using_weak_multiple_date_fallback: int = 0
    skipped_ambiguous_images: int = 0
    skipped_images_without_target: int = 0
    invalid_boxes: int = 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ExpDate Products-Real and Products-Synth annotations to YOLO format."
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=f"Fraction of Real/train and Synth images reserved for validation (default: {DEFAULT_VAL_RATIO}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed used for the train/val split (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="YOLO dataset output directory (default: <repository>/data/yolo).",
    )
    return parser.parse_args()


def _load_annotations(source: DatasetSource) -> list[ImageRecord]:
    with source.annotations_path.open("r", encoding="utf-8") as annotations_file:
        annotations = json.load(annotations_file)

    if not isinstance(annotations, dict):
        raise ValueError(f"Expected an object in {source.annotations_path}")

    return [
        ImageRecord(source=source, filename=filename, annotation=annotation)
        for filename, annotation in annotations.items()
    ]


def _prepare_output_dirs(output_dir: Path) -> None:
    for category in ("images", "labels"):
        for split in ("train", "val", "test"):
            split_dir = output_dir / category / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)


def _normalized_box(
    bbox: Any,
    image_width: Any,
    image_height: Any,
    *,
    image_name: str,
) -> tuple[float, float, float, float] | None:
    try:
        width = float(image_width)
        height = float(image_height)
        if width <= 0 or height <= 0 or not math.isfinite(width) or not math.isfinite(height):
            raise ValueError("image dimensions must be positive finite numbers")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("bbox must contain exactly four coordinates")

        x1, y1, x2, y2 = (float(value) for value in bbox)
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("bbox coordinates must be finite numbers")
    except (TypeError, ValueError) as error:
        warnings.warn(f"Skipping invalid exp/date target box in {image_name}: {error}")
        return None

    clipped_x1 = min(max(x1, 0.0), width)
    clipped_y1 = min(max(y1, 0.0), height)
    clipped_x2 = min(max(x2, 0.0), width)
    clipped_y2 = min(max(y2, 0.0), height)

    if (clipped_x1, clipped_y1, clipped_x2, clipped_y2) != (x1, y1, x2, y2):
        warnings.warn(f"Clipped out-of-bounds exp/date target box in {image_name}: {bbox}")

    box_width = clipped_x2 - clipped_x1
    box_height = clipped_y2 - clipped_y1
    if box_width <= 0 or box_height <= 0:
        warnings.warn(
            "Skipping invalid exp/date target box with non-positive area "
            f"in {image_name}: {bbox}"
        )
        return None

    return (
        (clipped_x1 + clipped_x2) / 2.0 / width,
        (clipped_y1 + clipped_y2) / 2.0 / height,
        box_width / width,
        box_height / height,
    )


def _destination_filename(record: ImageRecord) -> str:
    return f"{record.source.name}__{record.filename}"


def _bbox_center(annotation: dict[str, Any]) -> tuple[float, float] | None:
    bbox = annotation.get("bbox")
    try:
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = (float(value) for value in bbox)
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            return None
    except (TypeError, ValueError):
        return None
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _center_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _select_targets(annotation: dict[str, Any]) -> tuple[list[SelectedTarget], str | None]:
    annotations = [item for item in annotation.get("ann", []) if isinstance(item, dict)]
    exp_annotations = [item for item in annotations if item.get("cls") == "exp"]
    if exp_annotations:
        return [SelectedTarget(item, "exp", "exp_box") for item in exp_annotations], None

    date_annotations = [item for item in annotations if item.get("cls") == "date"]
    due_annotations = [item for item in annotations if item.get("cls") == "due"]
    prod_annotations = [item for item in annotations if item.get("cls") == "prod"]
    if not date_annotations:
        return [], "without_target"

    if due_annotations:
        date_centers = [
            (index, item, center)
            for index, item in enumerate(date_annotations)
            if (center := _bbox_center(item)) is not None
        ]
        prod_centers = [
            center for item in prod_annotations if (center := _bbox_center(item)) is not None
        ]
        selected_indexes: set[int] = set()
        for due_annotation in due_annotations:
            due_center = _bbox_center(due_annotation)
            if due_center is None or not date_centers:
                continue

            date_index, _, nearest_date_center = min(
                date_centers,
                key=lambda candidate: _center_distance(candidate[2], due_center),
            )
            due_distance = _center_distance(nearest_date_center, due_center)
            if prod_centers and min(
                _center_distance(nearest_date_center, prod_center) for prod_center in prod_centers
            ) < due_distance:
                continue
            selected_indexes.add(date_index)

        selected = [
            SelectedTarget(date_annotations[index], "date", "date_near_due")
            for index in sorted(selected_indexes)
        ]
        return selected, None if selected else "without_target"

    if len(date_annotations) == 1:
        return [SelectedTarget(date_annotations[0], "date", "single_date")], None
    if prod_annotations:
        return [], "ambiguous"
    return [
        SelectedTarget(item, "date", "weak_multiple_date") for item in date_annotations
    ], None


def _count_selection(summary: Summary, selected_reason: str) -> None:
    if selected_reason == "exp_box":
        summary.images_using_exp_boxes += 1
    elif selected_reason == "date_near_due":
        summary.images_using_date_near_due_fallback += 1
    elif selected_reason == "single_date":
        summary.images_using_single_date_fallback += 1
    else:
        summary.images_using_weak_multiple_date_fallback += 1


def _convert_record(
    record: ImageRecord,
    split: str,
    output_dir: Path,
    summary: Summary,
    debug_rows: list[dict[str, str]],
) -> bool:
    annotation = record.annotation
    if not isinstance(annotation, dict):
        warnings.warn(f"Skipping image with malformed annotation entry: {record.filename}")
        summary.skipped_images_without_target += 1
        return False

    selected_targets, skip_reason = _select_targets(annotation)
    if skip_reason == "ambiguous":
        warnings.warn(f"Skipping ambiguous image with multiple date boxes: {record.filename}")
        summary.skipped_ambiguous_images += 1
        return False
    if not selected_targets:
        warnings.warn(f"Skipping image without exp/date target annotation: {record.filename}")
        summary.skipped_images_without_target += 1
        return False

    label_lines: list[str] = []
    valid_targets: list[SelectedTarget] = []
    for selected_target in selected_targets:
        box = _normalized_box(
            selected_target.annotation.get("bbox"),
            annotation.get("width"),
            annotation.get("height"),
            image_name=record.filename,
        )
        if box is None:
            summary.invalid_boxes += 1
            continue
        x_center, y_center, width, height = box
        label_lines.append(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        valid_targets.append(selected_target)

    if not label_lines:
        warnings.warn(f"Skipping image without valid exp/date target boxes: {record.filename}")
        summary.skipped_images_without_target += 1
        return False

    source_image = record.source.images_dir / record.filename
    if not source_image.is_file():
        warnings.warn(f"Skipping missing image file: {source_image}")
        summary.skipped_images_without_target += 1
        return False

    destination_name = _destination_filename(record)
    destination_image = output_dir / "images" / split / destination_name
    destination_label = output_dir / "labels" / split / f"{Path(destination_name).stem}.txt"

    shutil.copy2(source_image, destination_image)
    destination_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    _count_selection(summary, valid_targets[0].reason)
    for selected_target in valid_targets:
        debug_rows.append(
            {
                "split": split,
                "output_image_name": destination_name,
                "source_image_name": record.filename,
                "selected_source_class": selected_target.source_class,
                "selected_reason": selected_target.reason,
                "selected_bbox": json.dumps(selected_target.annotation.get("bbox")),
                "transcription": str(selected_target.annotation.get("transcription", "")),
            }
        )
    return True


def _write_dataset_yaml(output_dir: Path) -> None:
    dataset_yaml = "\n".join(
        (
            "path: data/yolo",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            f"  {CLASS_ID}: {CLASS_NAME}",
            "",
        )
    )
    (output_dir / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")


def _convert_split(
    records: list[ImageRecord],
    split: str,
    output_dir: Path,
    summary: Summary,
    debug_rows: list[dict[str, str]],
) -> int:
    return sum(_convert_record(record, split, output_dir, summary, debug_rows) for record in records)


def _write_debug_csv(debug_csv_path: Path, debug_rows: list[dict[str, str]]) -> None:
    fieldnames = (
        "split",
        "output_image_name",
        "source_image_name",
        "selected_source_class",
        "selected_reason",
        "selected_bbox",
        "transcription",
    )
    debug_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with debug_csv_path.open("w", encoding="utf-8", newline="") as debug_csv:
        writer = csv.DictWriter(debug_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(debug_rows)


def _print_summary(summary: Summary) -> None:
    print(f"Conversion summary for exp/date targets -> {CLASS_ID}: {CLASS_NAME}")
    print(f"  train images: {summary.train_images}")
    print(f"  val images: {summary.val_images}")
    print(f"  test images: {summary.test_images}")
    print(f"  images using exp boxes: {summary.images_using_exp_boxes}")
    print(f"  images using date near due fallback: {summary.images_using_date_near_due_fallback}")
    print(f"  images using single date fallback: {summary.images_using_single_date_fallback}")
    print(
        "  images using weak multiple-date fallback: "
        f"{summary.images_using_weak_multiple_date_fallback}"
    )
    print(f"  skipped ambiguous images: {summary.skipped_ambiguous_images}")
    print(f"  skipped images without target: {summary.skipped_images_without_target}")
    print(f"  invalid boxes: {summary.invalid_boxes}")


def main() -> None:
    args = _parse_args()
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be greater than or equal to 0 and less than 1")

    repository_root = Path(__file__).resolve().parents[1]
    raw_expdate_dir = repository_root / "data" / "raw" / "expdate"
    output_dir = args.output_dir or repository_root / "data" / "yolo"

    real_train = DatasetSource(
        name="real_train",
        images_dir=raw_expdate_dir / "Products-Real" / "train" / "images",
        annotations_path=raw_expdate_dir / "Products-Real" / "train" / "annotations.json",
    )
    synth = DatasetSource(
        name="synth",
        images_dir=raw_expdate_dir / "Products-Synth" / "images",
        annotations_path=raw_expdate_dir / "Products-Synth" / "annotations.json",
    )
    real_evaluation = DatasetSource(
        name="real_evaluation",
        images_dir=raw_expdate_dir / "Products-Real" / "evaluation" / "images",
        annotations_path=raw_expdate_dir / "Products-Real" / "evaluation" / "annotations.json",
    )

    train_val_records = _load_annotations(real_train) + _load_annotations(synth)
    random.Random(args.seed).shuffle(train_val_records)
    val_count = round(len(train_val_records) * args.val_ratio)
    val_records = train_val_records[:val_count]
    train_records = train_val_records[val_count:]
    test_records = _load_annotations(real_evaluation)

    _prepare_output_dirs(output_dir)
    summary = Summary()
    debug_rows: list[dict[str, str]] = []
    summary.train_images = _convert_split(train_records, "train", output_dir, summary, debug_rows)
    summary.val_images = _convert_split(val_records, "val", output_dir, summary, debug_rows)
    summary.test_images = _convert_split(test_records, "test", output_dir, summary, debug_rows)
    _write_dataset_yaml(output_dir)
    _write_debug_csv(repository_root / "outputs" / "logs" / "dataset_conversion_summary.csv", debug_rows)
    _print_summary(summary)


if __name__ == "__main__":
    main()
