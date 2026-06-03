from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.src.api.endpoints.scan import router as scan_router

CROPPED_OUTPUT_DIR = Path("outputs/cropped")
CROPPED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Smart Expiration System")

app.include_router(scan_router, prefix="/api", tags=["scan"])


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/outputs/cropped", StaticFiles(directory=CROPPED_OUTPUT_DIR), name="cropped")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
