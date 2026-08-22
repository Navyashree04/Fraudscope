# FraudScope -- ML-powered transaction risk scoring service
# Serves a model trained on a PUBLIC/SYNTHETIC dataset only.

# ---- builder stage --------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- final stage -----------------------------------------------------------
FROM python:3.11-slim

WORKDIR /srv/fraudscope

COPY --from=builder /install /usr/local

COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY .env.example ./.env.example

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# NOTE: models/ is expected to already contain a trained artifact
# (fraud_model_v1.joblib + metadata.json) produced by `python -m src.train`
# on a public/synthetic dataset -- mount a volume over models/ in
# docker-compose to pick up retraining without rebuilding the image.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
