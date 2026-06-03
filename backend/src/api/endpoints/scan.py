from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.src.services.scan_service import process_uploaded_image

router = APIRouter()
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@router.post("/scan")
async def scan_image(file: UploadFile = File(...)) -> dict:
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image must use a .jpg, .jpeg, or .png extension.",
        )

    try:
        return await process_uploaded_image(file)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail="Image scan failed.") from error
