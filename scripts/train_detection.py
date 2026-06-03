"""Train a YOLOv8 expiration-date detector with Ultralytics."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS = 100
DEFAULT_IMAGE_SIZE = 640
DEFAULT_BATCH_SIZE = 16
DEFAULT_DEVICE = "cpu"
DEFAULT_WORKERS = 0
DEFAULT_CACHE = False
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
    parser = argparse.ArgumentParser(description="Train a YOLOv8 expiration-date detector.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Pretrained YOLO model name or path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Training image size in pixels (default: {DEFAULT_IMAGE_SIZE}).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"Training device, such as cpu, 0, or 0,1 (default: {DEFAULT_DEVICE}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of data-loading workers (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CACHE,
        help="Cache dataset images during training.",
    )
    return parser.parse_args()


def _validate_positive(value: int, argument_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{argument_name} must be greater than 0")


def _directory_has_files(directory: Path, extensions: set[str]) -> bool:
    return directory.is_dir() and any(
        path.is_file() and path.suffix.lower() in extensions for path in directory.iterdir()
    )


def _validate_dataset(dataset_yaml: Path, train_images_dir: Path, train_labels_dir: Path) -> None:
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset config does not exist: {dataset_yaml}")
    if not _directory_has_files(train_images_dir, IMAGE_EXTENSIONS):
        raise ValueError(f"Training image directory is missing or empty: {train_images_dir}")
    if not _directory_has_files(train_labels_dir, {".txt"}):
        raise ValueError(f"Training label directory is missing or empty: {train_labels_dir}")


def _resolve_best_weight(training_result: Any, run_dir: Path) -> Path:
    trainer = getattr(training_result, "trainer", None)
    best_weight = getattr(trainer, "best", None)
    if best_weight:
        resolved_weight = Path(best_weight)
        if resolved_weight.is_file():
            return resolved_weight

    fallback_weight = run_dir / "weights" / "best.pt"
    if fallback_weight.is_file():
        return fallback_weight
    raise FileNotFoundError(f"Ultralytics did not produce a best.pt weight under: {run_dir}")


def main() -> int:
    args = _parse_args()
    _validate_positive(args.epochs, "--epochs")
    _validate_positive(args.imgsz, "--imgsz")
    _validate_positive(args.batch, "--batch")
    if args.workers < 0:
        raise ValueError("--workers must be greater than or equal to 0")

    repository_root = Path(__file__).resolve().parents[1]
    dataset_yaml = repository_root / "data" / "yolo" / "dataset.yaml"
    train_images_dir = repository_root / "data" / "yolo" / "images" / "train"
    train_labels_dir = repository_root / "data" / "yolo" / "labels" / "train"
    runs_dir = repository_root / "outputs" / "logs" / "detection"
    models_dir = repository_root / "models" / "detection"
    run_name = f"yolov8_expiration_date_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = runs_dir / run_name
    best_model_output = models_dir / "best.pt"

    _validate_dataset(dataset_yaml, train_images_dir, train_labels_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("YOLOv8 expiration-date detection training")
    print(f"  dataset path: {dataset_yaml}")
    print(f"  model name: {args.model}")
    print(f"  epochs: {args.epochs}")
    print(f"  image size: {args.imgsz}")
    print(f"  batch size: {args.batch}")
    print(f"  device: {args.device}")
    print(f"  workers: {args.workers}")
    print(f"  cache: {args.cache}")
    print(f"  training run directory: {run_dir}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    training_result = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        project=str(runs_dir),
        name=run_name,
        exist_ok=False,
    )

    best_weight = _resolve_best_weight(training_result, run_dir)
    shutil.copy2(best_weight, best_model_output)
    print(f"  best model output path: {best_model_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
