"""Draw YOLO labels on random dataset samples for visual inspection."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2


CLASS_NAME = "expiration_date"
DEFAULT_SPLIT = "train"
DEFAULT_NUM_SAMPLES = 10
DEFAULT_SEED = 42
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize random samples from a YOLO dataset.")
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default=DEFAULT_SPLIT,
        help=f"Dataset split to visualize (default: {DEFAULT_SPLIT}).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of random images to visualize (default: {DEFAULT_NUM_SAMPLES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed used for sample selection (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="YOLO dataset directory (default: <repository>/data/yolo).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Visualization output directory (default: <repository>/outputs/predictions/yolo_dataset_check).",
    )
    return parser.parse_args()


def _list_images(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _load_yolo_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 fields, found {len(parts)}")

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError as error:
            raise ValueError(f"{label_path}:{line_number}: invalid YOLO label") from error

        boxes.append((class_id, x_center, y_center, width, height))

    return boxes


def _draw_box(
    image,
    box: tuple[int, float, float, float, float],
) -> None:
    class_id, x_center, y_center, box_width, box_height = box
    image_height, image_width = image.shape[:2]

    x1 = max(0, round((x_center - box_width / 2.0) * image_width))
    y1 = max(0, round((y_center - box_height / 2.0) * image_height))
    x2 = min(image_width - 1, round((x_center + box_width / 2.0) * image_width))
    y2 = min(image_height - 1, round((y_center + box_height / 2.0) * image_height))

    color = (0, 255, 0)
    label = f"{class_id}: {CLASS_NAME}"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        1,
    )
    label_y1 = max(0, y1 - text_height - baseline - 6)
    label_y2 = label_y1 + text_height + baseline + 6
    cv2.rectangle(image, (x1, label_y1), (x1 + text_width + 8, label_y2), color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 4, label_y2 - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def main() -> int:
    args = _parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be greater than 0")

    repository_root = Path(__file__).resolve().parents[1]
    dataset_dir = (args.dataset_dir or repository_root / "data" / "yolo").resolve()
    output_dir = (
        args.output_dir or repository_root / "outputs" / "predictions" / "yolo_dataset_check"
    ).resolve()
    images_dir = dataset_dir / "images" / args.split
    labels_dir = dataset_dir / "labels" / args.split

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {labels_dir}")

    images = _list_images(images_dir)
    if not images:
        raise ValueError(f"No images found in {images_dir}")

    sample_count = min(args.num_samples, len(images))
    selected_images = random.Random(args.seed).sample(images, sample_count)
    split_output_dir = output_dir / args.split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in selected_images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Matching label file does not exist: {label_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        for box in _load_yolo_boxes(label_path):
            _draw_box(image, box)

        output_path = split_output_dir / image_path.name
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Unable to write visualization: {output_path}")
        print(f"Saved: {output_path.relative_to(repository_root)}")

    print(f"Visualized {sample_count} {args.split} image(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
