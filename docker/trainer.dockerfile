FROM python:3.11-slim
# XGBoost needs the OpenMP runtime, which slim images don't include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY docker/requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt
COPY src/ src/
ENTRYPOINT ["python", "-m", "src.cloud.train_job"]
