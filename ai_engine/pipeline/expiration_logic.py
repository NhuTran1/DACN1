from datetime import date


def evaluate_expiration(expiration_date: str | None, near_expiry_days: int = 30) -> dict:
    """Evaluate an ISO expiration date relative to the current date."""
    warnings: list[str] = []

    if near_expiry_days < 0:
        raise ValueError("near_expiry_days must be greater than or equal to 0")

    if not expiration_date:
        warnings.append("Expiration date is missing.")
        return {
            "status": "needs_review",
            "days_remaining": None,
            "expiration_date": expiration_date,
            "warnings": warnings,
        }

    try:
        parsed_date = date.fromisoformat(expiration_date)
    except (TypeError, ValueError):
        warnings.append("Expiration date must use a valid YYYY-MM-DD format.")
        return {
            "status": "needs_review",
            "days_remaining": None,
            "expiration_date": expiration_date,
            "warnings": warnings,
        }

    days_remaining = (parsed_date - date.today()).days
    if days_remaining < 0:
        status = "expired"
    elif days_remaining <= near_expiry_days:
        status = "near_expiry"
    else:
        status = "valid"

    return {
        "status": status,
        "days_remaining": days_remaining,
        "expiration_date": expiration_date,
        "warnings": warnings,
    }


def build_expiration_result(image_path: str, detection: dict, ocr: dict, parsed: dict) -> dict:
    status = "needs_review"
    if parsed.get("normalized_date"):
        status = "parsed"

    return {
        "image_path": image_path,
        "bbox": detection.get("bbox"),
        "detection_confidence": detection.get("confidence"),
        "ocr_text": ocr.get("text"),
        "ocr_confidence": ocr.get("confidence"),
        "parsed_date": parsed.get("normalized_date"),
        "is_valid": parsed.get("is_valid", False),
        "status": status,
        "processed_at": str(date.today())
    }
