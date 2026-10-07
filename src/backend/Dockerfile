FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Hindari prompt interaktif saat install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# Install Python 3.11 + tools dasar
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Pastikan `python` dan `pip` mengarah ke Python 3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip untuk Python 3.11 (PAKAI python -m pip, bukan pip langsung)
RUN python -m pip install --no-cache-dir --upgrade pip

# Install library utama (PAKAI python -m pip)
RUN python -m pip install --no-cache-dir \
    ctranslate2==4.5.0 \
    transformers==4.45.0 \
    sentencepiece \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    requests

# Working dir
WORKDIR /app

# Default command
CMD ["bash"]