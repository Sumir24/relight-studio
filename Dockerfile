FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    RELIGHT_HOST=0.0.0.0 \
    RELIGHT_PORT=8000 \
    RELIGHT_DEVICE=cuda \
    RELIGHT_STORAGE_DIR=/data/relight_server

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-models.txt requirements-server.txt /app/

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir torch torchvision \
    && python3 -m pip install --no-cache-dir -r requirements-models.txt -r requirements-server.txt

COPY relight_engine /app/relight_engine
COPY scripts /app/scripts

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["python3", "scripts/run_server.py"]
