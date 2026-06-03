"""Run YOLO expiration-date detection on a single image."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the trained expiration-date detector.")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the image used for inference.",
    )
    return parser.parse_args()


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _draw_detection(
    image,
    bbox: tuple[int, int, int, int],
    confidence: float,
    class_name: str,
) -> None:
    x1, y1, x2, y2 = bbox
    color = (0, 255, 0)
    label = f"{class_name} {confidence:.2f}"

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
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
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def main() -> int:
    args = _parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    model_path = repository_root / "models" / "detection" / "best.pt"
    image_path = args.image.resolve()
    output_dir = repository_root / "outputs" / "predictions" / "detection_test"

    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model does not exist: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read input image: {image_path}")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model.predict(source=str(image_path), verbose=False)
    detections_found = False

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = (round(value) for value in box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = _class_name(result.names, class_id)
            bbox = (x1, y1, x2, y2)

            detections_found = True
            _draw_detection(image, bbox, confidence, class_name)
            print(f"Detected bbox={list(bbox)}, confidence={confidence:.6f}, class={class_name}")

    if not detections_found:
        print("Warning: no expiration-date detection found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_detection{image_path.suffix}"
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Unable to write detection output image: {output_path}")

    print(f"Saved output image: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
