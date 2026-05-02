FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Railway / Render 會注入 PORT；本機預設 8765
ENV PORT=8765
EXPOSE 8765

CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"
