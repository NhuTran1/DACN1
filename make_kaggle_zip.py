from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path("kaggle_upload")
zip_path = Path("smart-expiration-yolo-kaggle.zip")

if zip_path.exists():
    zip_path.unlink()

with ZipFile(zip_path, "w", ZIP_DEFLATED) as z:
    for file_path in root.rglob("*"):
        if file_path.is_file():
            # Kaggle cần path dùng dấu /, không dùng dấu \
            arcname = file_path.relative_to(root).as_posix()
            z.write(file_path, arcname)

print(f"Created: {zip_path}")
