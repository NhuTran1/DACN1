const form = document.querySelector("#scan-form");
const imageInput = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const scanButton = document.querySelector("#scan-button");
const selectedFile = document.querySelector("#selected-file");
const errorMessage = document.querySelector("#error-message");
const resultPanel = document.querySelector("#result-panel");
const originalPreview = document.querySelector("#original-preview");
const roiFigure = document.querySelector("#roi-figure");
const roiPreview = document.querySelector("#roi-preview");
const statusBadge = document.querySelector("#status-badge");
const parsedDate = document.querySelector("#parsed-date");
const selectedRaw = document.querySelector("#selected-raw");
const ocrText = document.querySelector("#ocr-text");
const detectionConfidence = document.querySelector("#detection-confidence");
const ocrConfidence = document.querySelector("#ocr-confidence");
const warningsBlock = document.querySelector("#warnings-block");
const warningsList = document.querySelector("#warnings-list");

const statusLabels = {
  valid: "Còn hạn",
  near_expiry: "Sắp hết hạn",
  expired: "Hết hạn",
  needs_review: "Cần kiểm tra lại",
};

let previewUrl = null;

function displayValue(value) {
  return value || "-";
}

function formatConfidence(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "-";
}

function setSelectedFile(file) {
  if (!file) {
    selectedFile.textContent = "Chưa chọn ảnh";
    return;
  }
  selectedFile.textContent = file.name;
}

function setOriginalPreview(file) {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  previewUrl = URL.createObjectURL(file);
  originalPreview.src = previewUrl;
}

function roiUrl(path) {
  const filename = String(path).split(/[\\/]/).pop();
  return `/outputs/cropped/${encodeURIComponent(filename)}`;
}

function renderWarnings(warnings) {
  warningsList.replaceChildren();
  const items = Array.isArray(warnings) ? warnings : [];
  warningsBlock.hidden = items.length === 0;
  for (const warning of items) {
    const item = document.createElement("li");
    item.textContent = String(warning);
    warningsList.append(item);
  }
}

function renderResult(result) {
  const status = statusLabels[result.status] ? result.status : "needs_review";
  statusBadge.className = `status-badge status-${status}`;
  statusBadge.textContent = statusLabels[status];
  parsedDate.textContent = displayValue(result.parsed_date);
  selectedRaw.textContent = displayValue(result.selected_raw);
  ocrText.textContent = displayValue(result.ocr_text);
  detectionConfidence.textContent = formatConfidence(result.detection_confidence);
  ocrConfidence.textContent = formatConfidence(result.ocr_confidence);
  renderWarnings(result.warnings);

  roiFigure.hidden = !result.roi_path;
  if (result.roi_path) {
    roiPreview.src = roiUrl(result.roi_path);
  } else {
    roiPreview.removeAttribute("src");
  }
  resultPanel.hidden = false;
}

async function scanImage(file) {
  const payload = new FormData();
  payload.append("file", file);
  const response = await fetch("/api/scan", { method: "POST", body: payload });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(result.detail || "Không thể xử lý ảnh.");
  }
  return result;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const [file] = imageInput.files;
  if (!file) {
    errorMessage.textContent = "Vui lòng chọn ảnh trước khi quét.";
    errorMessage.hidden = false;
    return;
  }

  errorMessage.hidden = true;
  scanButton.disabled = true;
  scanButton.textContent = "Đang phân tích...";
  setOriginalPreview(file);
  try {
    renderResult(await scanImage(file));
  } catch (error) {
    errorMessage.textContent = error.message;
    errorMessage.hidden = false;
  } finally {
    scanButton.disabled = false;
    scanButton.textContent = "Quét hạn sử dụng";
  }
});

imageInput.addEventListener("change", () => setSelectedFile(imageInput.files[0]));

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
}

dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (!file) {
    return;
  }
  const transfer = new DataTransfer();
  transfer.items.add(file);
  imageInput.files = transfer.files;
  setSelectedFile(file);
});
