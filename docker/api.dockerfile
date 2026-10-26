FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY docker/requirements-api.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 -r requirements-api.txt
COPY src/ src/
COPY sql/schemas/ sql/schemas/
COPY artifacts/xgb_v2/final/ artifacts/xgb_v2/final/
RUN useradd --create-home appuser
USER appuser
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT}"]
