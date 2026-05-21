FROM python:3.12-slim

WORKDIR /app

# OpenCV 執行時依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 建置時下載 YOLO 權重，避免雲端冷啟動才抓檔失敗
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY app ./app

ENV PORT=8765
ENV YOLO_ENABLED=true
EXPOSE 8765

CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --ws-ping-interval 20 --ws-ping-timeout 20"
