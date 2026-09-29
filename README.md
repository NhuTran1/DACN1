# Smart Expiration System

## 1. Giới thiệu

Smart Expiration System là hệ thống thông minh hỗ trợ nhận diện hạn sử dụng của hàng hóa từ ảnh sản phẩm. Hệ thống sử dụng mô hình YOLOv8 để phát hiện vùng chứa ngày hạn sử dụng, sau đó cắt vùng ảnh ROI, tiền xử lý bằng OpenCV, nhận dạng ký tự bằng PaddleOCR, trích xuất ngày bằng Regex/Date Parser và đánh giá trạng thái hạn sử dụng.

Luồng xử lý chính:

```text
Image Upload
→ YOLOv8 Detection
→ Crop Expiration ROI
→ OpenCV Preprocessing
→ PaddleOCR
→ Regex Date Parsing
→ Expiration Logic
→ FastAPI Response
```

Trạng thái hệ thống trả về gồm:

```text
valid        : Còn hạn
near_expiry  : Sắp hết hạn
expired      : Hết hạn
needs_review : Cần kiểm tra lại
```

---

## 2. Công nghệ sử dụng

- Python 3.10+
- FastAPI
- Uvicorn
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- PaddleOCR
- PaddlePaddle
- Regex Date Parser
- ReactJS/Vite frontend, nếu sử dụng frontend demo

---

## 3. Cấu trúc thư mục chính

```text
smart-expiration-system/
├── ai_engine/
│   ├── detection/
│   │   └── predict.py
│   ├── ocr/
│   │   ├── paddle_ocr.py
│   │   ├── preprocess.py
│   │   ├── parse_date.py
│   │   └── regex_patterns.py
│   └── pipeline/
│       ├── pipeline.py
│       └── expiration_logic.py
│
├── backend/
│   └── src/
│       ├── main.py
│       ├── api/
│       ├── services/
│       ├── schemas/
│       └── core/
│
├── frontend/
│   └── src/
│
├── models/
│   └── detection/
│       └── best.pt
│
├── scripts/
│   ├── test_detection.py
│   ├── test_pipeline_batch.py
│   ├── train_detection.py
│   ├── validate_yolo_dataset.py
│   └── convert_expdate_products_to_yolo.py
│
├── uploads/
├── requirements.txt
├── .env.example
└── README.md
```

---

## 4. Cài đặt môi trường

### Bước 1: Clone dự án

```bash
git clone <YOUR_REPOSITORY_URL>
cd smart-expiration-system
```

### Bước 2: Tạo môi trường ảo

Trên Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn activate script, chạy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### Bước 3: Cài thư viện

```powershell
pip install -r requirements.txt
```

Nếu thiếu thư viện upload file cho FastAPI:

```powershell
pip install python-multipart
```

---

## 5. Chuẩn bị model

Model YOLO đã train được đặt tại:

```text
models/detection/best.pt
```

Nếu chưa có file này, cần tải model đã train từ nhóm và đặt đúng đường dẫn:

```text
smart-expiration-system/models/detection/best.pt
```

Lưu ý: file model phải đúng tên `best.pt`.

---

## 6. Chạy kiểm thử detection

Test YOLO detection trên một ảnh:

```powershell
.\venv\Scripts\python.exe scripts\test_detection.py --image path\to\image.jpg
```

Ví dụ:

```powershell
.\venv\Scripts\python.exe scripts\test_detection.py --image data\yolo\images\test\real_evaluation__test_00115.jpg
```

Kết quả ảnh dự đoán sẽ được lưu tại:

```text
outputs/predictions/detection_test/
```

---

## 7. Chạy kiểm thử full AI pipeline

Pipeline gồm:

```text
YOLO → Crop ROI → Preprocess → OCR → Parse Date → Expiration Logic
```

Chạy thử với ảnh:

```powershell
.\venv\Scripts\python.exe scripts\test_pipeline_batch.py --image-dir path\to\image_folder --limit 10
```

Ví dụ:

```powershell
.\venv\Scripts\python.exe scripts\test_pipeline_batch.py --image-dir data\yolo\images\test --limit 10
```

Kết quả summary được lưu tại:

```text
outputs/predictions/pipeline_test/summary.csv
```

### Test bằng OCR giả lập

Dùng khi muốn kiểm tra parser và expiration logic mà không phụ thuộc PaddleOCR:

```powershell
.\venv\Scripts\python.exe scripts\test_pipeline_batch.py --image-dir data\yolo\images\test --limit 10 --mock-text "PRO 13.11.2020 EXP 11.07.2021"
```

Tắt mock OCR:

```powershell
Remove-Item Env:MOCK_OCR_TEXT -ErrorAction SilentlyContinue
```

---

## 8. Chạy backend FastAPI

Chạy server:

```powershell
python -m uvicorn backend.src.main:app --reload
```

Mở Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Endpoint chính:

```text
POST /api/scan
```

Chức năng:

- Upload ảnh sản phẩm
- Chạy AI pipeline
- Trả về JSON kết quả gồm:
  - ngày hết hạn
  - trạng thái hạn sử dụng
  - độ tin cậy detection
  - độ tin cậy OCR
  - văn bản OCR
  - đường dẫn ROI
  - cảnh báo nếu có

---

## 9. Chạy frontend

Nếu dùng frontend React/Vite:

```powershell
cd frontend
npm install
npm run dev
```

Sau đó mở đường dẫn Vite hiển thị trong terminal, thường là:

```text
http://localhost:5173
```

Frontend demo gồm:

- Upload ảnh sản phẩm
- Hiển thị ảnh gốc
- Hiển thị vùng ngày tháng được phát hiện
- Hiển thị OCR text
- Hiển thị ngày hết hạn
- Hiển thị trạng thái: Còn hạn / Sắp hết hạn / Hết hạn / Cần kiểm tra lại

---

## 10. Lưu ý về dataset

Dataset lớn không được đưa trực tiếp lên GitHub. Các thư mục sau không được commit:

```text
data/raw/
data/yolo/
outputs/
kaggle_upload/
```

Nếu cần train lại model, thành viên nhóm cần tải dataset riêng và đặt theo cấu trúc:

```text
data/raw/expdate/
├── Products-Real/
└── Products-Synth/
```

Sau đó convert sang YOLO format bằng script:

```powershell
.\venv\Scripts\python.exe scripts\convert_expdate_products_to_yolo.py
```

Validate dataset:

```powershell
.\venv\Scripts\python.exe scripts\validate_yolo_dataset.py
```

Train YOLO:

```powershell
.\venv\Scripts\python.exe scripts\train_detection.py --epochs 50 --imgsz 640 --batch 8
```

Khuyến nghị train trên Kaggle hoặc Google Colab GPU thay vì máy local yếu.

---

## 11. Một số lỗi thường gặp

### Lỗi thiếu `python-multipart`

Nếu chạy FastAPI bị lỗi form-data:

```text
Form data requires "python-multipart" to be installed
```

Cài thêm:

```powershell
pip install python-multipart
```

### PaddleOCR tải model lần đầu lâu

Lần đầu chạy OCR thật, PaddleOCR có thể tải model về cache. Cần có kết nối mạng và đợi hoàn tất.

### OCR đọc sai hoặc không tìm thấy ngày

Một số ảnh có chữ quá mờ, nghiêng, phản sáng hoặc định dạng ngày lạ có thể trả về:

```text
needs_review
```

Trường hợp này người dùng cần kiểm tra lại thủ công hoặc chụp lại ảnh rõ hơn.

### Thay model mới nhưng hệ thống vẫn dùng model cũ

Sau khi thay `models/detection/best.pt`, nên restart backend hoặc terminal để chắc chắn model mới được load.

---

## 12. Kết quả hiện tại

Model YOLOv8 đã được train trên Kaggle với 50 epochs, ảnh kích thước 640. Pipeline đã test thành công với ảnh ngoài dataset, ví dụ:

```text
NSX: 15.01.2024
HSD: 14.01.2026
```

Hệ thống nhận diện được:

```text
Ngày hết hạn: 2026-01-14
Văn bản OCR: HSD:14.01.2026B
Trạng thái: expired
```

---

## 13. Thành viên nhóm

- Trần Quang Như
- Lê Duy Đạt

---

## 14. Mục tiêu phát triển tiếp theo

- Cải thiện OCR cho ảnh mờ, nghiêng, phản sáng.
- Hỗ trợ thêm nhiều định dạng ngày như `14 MAY 09`, `040917`, `12 10 2022`.
- Bổ sung cơ sở dữ liệu quản lý lịch sử quét.
- Bổ sung dashboard thống kê hàng hóa còn hạn, sắp hết hạn, hết hạn.
- Tối ưu frontend demo cho cửa hàng bán lẻ.
#   P r o d u c t _ E x p i r y _ D a t e  
 