from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _build_output_path(image_path: str, output_path: str | None) -> Path:
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    source = Path(image_path)
    return source.with_name(f"{source.stem}_preprocessed.png")


def _enhance_contrast(gray_image: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    return clahe.apply(gray_image)


def _sharpen(image: np.ndarray, amount: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def preprocess_roi(
    image_path: str,
    *,
    output_path: str | None = None,
    enhance_contrast: bool = True,
    denoise: bool = False,
    threshold: bool = False,
    sharpen: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
    sharpen_amount: float = 0.35,
) -> dict[str, Any]:
    """Preprocess a cropped expiration-date ROI for OCR.

    The defaults are intentionally conservative so small printed and
    dot-matrix-like characters are not erased before OCR.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read ROI image: {image_path}")

    operations: list[str] = []
    original_shape = image.shape

    processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    operations.append("grayscale")

    if enhance_contrast:
        processed = _enhance_contrast(
            processed,
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size,
        )
        operations.append("contrast_enhancement")

    if denoise:
        processed = cv2.fastNlMeansDenoising(
            processed,
            None,
            h=7,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        operations.append("denoising")

    if sharpen:
        processed = _sharpen(processed, amount=sharpen_amount)
        operations.append("sharpening")

    if threshold:
        processed = cv2.adaptiveThreshold(
            processed,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=31,
            C=7,
        )
        operations.append("adaptive_thresholding")

    processed_path = _build_output_path(image_path, output_path)
    success = cv2.imwrite(str(processed_path), processed)
    if not success:
        raise OSError(f"Unable to write preprocessed ROI image: {processed_path}")

    return {
        "processed_image_path": str(processed_path),
        "source_image_path": image_path,
        "operations": operations,
        "metadata": {
            "original_shape": original_shape,
            "processed_shape": processed.shape,
            "enhance_contrast": enhance_contrast,
            "denoise": denoise,
            "threshold": threshold,
            "sharpen": sharpen,
            "clahe_clip_limit": clahe_clip_limit,
            "clahe_tile_grid_size": clahe_tile_grid_size,
            "sharpen_amount": sharpen_amount,
        },
    }
