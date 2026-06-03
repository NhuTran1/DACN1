# Project Overview
This repository is an AI-powered smart expiration detection system.

## Tech Stack
- Python
- FastAPI
- OpenCV
- PyTorch
- YOLO
- PaddleOCR
- SQLite for development
- Frontend for demo only

## Repository Layout
- `ai_engine/`: detection, OCR, parsing, expiration logic, end-to-end AI pipeline
- `backend/src/`: FastAPI backend
- `frontend/`: demo UI
- `scripts/`: train, evaluate, export scripts
- `data/`: raw and processed datasets
- `outputs/`: predictions, logs, metrics
- `docs/`: report, diagrams, presentation

## Working Rules
- Do not redesign the repository unless explicitly asked
- Keep code modular and readable
- Avoid duplicate logic
- Use type hints for public functions
- Do not hardcode absolute paths
- Keep API endpoints thin; move logic into services
- Do not modify files inside `data/raw/`
- Prefer minimal changes over broad refactors

## Expected AI Flow
image
-> YOLO detection
-> crop expiration region
-> OpenCV preprocessing
-> PaddleOCR
-> regex date parsing
-> expiration logic
-> structured result

## Backend Rules
- Validate uploaded files before processing
- Save the original uploaded image
- Return structured JSON
- Do not load heavy models repeatedly inside request handlers

## Definition of Done
A task is done when:
1. imports resolve,
2. code runs,
3. logic is explained,
4. no duplicate logic is introduced,
5. a quick verification step is included.