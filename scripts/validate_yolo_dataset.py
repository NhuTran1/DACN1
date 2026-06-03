"""Validate a YOLO detection dataset without modifying any files."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path


EXPECTED_CLASS_ID = 0
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass
class SplitSummary:
    images: int = 0
    labels: int = 0
    boxes: int = 0
    empty_labels: list[Path] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)
    images_without_labels: list[Path] = field(default_factory=list)
    labels_without_images: list[Path] = field(default_factory=list)
    missing_directories: list[Path] = field(default_factory=list)

    @property
    def issue_count(self) -> int:
        return (
            len(self.empty_labels)
            + len(self.invalid_labels)
            + len(self.images_without_labels)
            + len(self.labels_without_images)
            + len(self.missing_directories)
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO detection dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="YOLO dataset directory (default: <repository>/data/yolo).",
    )
    return parser.parse_args()


def _list_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _list_labels(labels_dir: Path) -> list[Path]:
    if not labels_dir.is_dir():
        return []
    return sorted(
        path for path in labels_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"
    )


def _validate_label_line(label_path: Path, line_number: int, line: str) -> str | None:
    parts = line.split()
    location = f"{label_path}:{line_number}"
    if len(parts) != 5:
        return f"{location}: expected 5 fields, found {len(parts)}"

    class_id_text, *coordinate_texts = parts
    try:
        class_id = int(class_id_text)
    except ValueError:
        return f"{location}: class_id must be an integer, found {class_id_text!r}"

    if class_id != EXPECTED_CLASS_ID:
        return f"{location}: class_id must be {EXPECTED_CLASS_ID}, found {class_id}"

    try:
        x_center, y_center, width, height = (float(value) for value in coordinate_texts)
    except ValueError:
        return f"{location}: coordinates must be numbers"

    coordinates = {
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
    }
    for name, value in coordinates.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return f"{location}: {name} must be between 0 and 1, found {value}"

    if width <= 0.0 or height <= 0.0:
        return f"{location}: width and height must be greater than 0"

    return None


def _validate_label_file(label_path: Path, summary: SplitSummary) -> None:
    lines = label_path.read_text(encoding="utf-8").splitlines()
    non_empty_lines = [(line_number, line.strip()) for line_number, line in enumerate(lines, 1) if line.strip()]
    if not non_empty_lines:
        summary.empty_labels.append(label_path)
        return

    for line_number, line in non_empty_lines:
        error = _validate_label_line(label_path, line_number, line)
        if error:
            summary.invalid_labels.append(error)
        else:
            summary.boxes += 1


def _validate_split(dataset_dir: Path, split: str) -> SplitSummary:
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split
    summary = SplitSummary()

    for directory in (images_dir, labels_dir):
        if not directory.is_dir():
            summary.missing_directories.append(directory)

    images = _list_images(images_dir)
    labels = _list_labels(labels_dir)
    summary.images = len(images)
    summary.labels = len(labels)

    images_by_stem = {image.stem: image for image in images}
    labels_by_stem = {label.stem: label for label in labels}

    summary.images_without_labels.extend(
        images_by_stem[stem] for stem in sorted(images_by_stem.keys() - labels_by_stem.keys())
    )
    summary.labels_without_images.extend(
        labels_by_stem[stem] for stem in sorted(labels_by_stem.keys() - images_by_stem.keys())
    )

    for label_path in labels:
        _validate_label_file(label_path, summary)

    return summary


def _print_paths(title: str, paths: list[Path], dataset_dir: Path) -> None:
    if not paths:
        return
    print(f"  {title}:")
    for path in paths:
        print(f"    - {path.relative_to(dataset_dir)}")


def _print_split_summary(split: str, summary: SplitSummary, dataset_dir: Path) -> None:
    print(f"{split}:")
    print(f"  images: {summary.images}")
    print(f"  labels: {summary.labels}")
    print(f"  valid boxes: {summary.boxes}")
    print(f"  empty labels: {len(summary.empty_labels)}")
    print(f"  invalid labels: {len(summary.invalid_labels)}")
    print(f"  images without labels: {len(summary.images_without_labels)}")
    print(f"  labels without images: {len(summary.labels_without_images)}")
    print(f"  missing directories: {len(summary.missing_directories)}")

    _print_paths("empty label files", summary.empty_labels, dataset_dir)
    _print_paths("images without labels", summary.images_without_labels, dataset_dir)
    _print_paths("labels without images", summary.labels_without_images, dataset_dir)
    _print_paths("missing directories", summary.missing_directories, dataset_dir)
    if summary.invalid_labels:
        print("  invalid label lines:")
        for error in summary.invalid_labels:
            print(f"    - {error}")


def main() -> int:
    args = _parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    dataset_dir = (args.dataset_dir or repository_root / "data" / "yolo").resolve()

    summaries = {split: _validate_split(dataset_dir, split) for split in SPLITS}

    print(f"YOLO dataset validation: {dataset_dir}")
    for split in SPLITS:
        _print_split_summary(split, summaries[split], dataset_dir)

    total_issues = sum(summary.issue_count for summary in summaries.values())
    print("overall:")
    print(f"  total images: {sum(summary.images for summary in summaries.values())}")
    print(f"  total labels: {sum(summary.labels for summary in summaries.values())}")
    print(f"  total valid boxes: {sum(summary.boxes for summary in summaries.values())}")
    print(f"  total issues: {total_issues}")
    print(f"  status: {'valid' if total_issues == 0 else 'invalid'}")
    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
