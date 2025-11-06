FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY ml-models/requirements.txt ./

RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local

COPY ml-models ./ml-models

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "ml-models.api:app", "--host", "0.0.0.0", "--port", "8000"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"
