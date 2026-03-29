FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first
RUN pip install --no-cache-dir torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY configs/settings.yaml ./configs/
COPY configs/prompts.yaml ./configs/

# Don't copy .env (passed via env_file in docker-compose)

EXPOSE 8000

ENV PYTHONPATH=/app
ENV PYTHONIOENCODING=utf-8

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
