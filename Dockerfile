FROM python:3.12-slim

# ffmpeg is required by transformers/librosa to decode m4a and other formats
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# RTX 50-series (Blackwell, sm_120) needs the CUDA 12.8 wheels.
# These wheels bundle the CUDA runtime, so only the host NVIDIA driver is required.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchaudio==2.8.0

COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app/transcribe.py /app/transcribe.py

ENV HF_HOME=/cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app
ENTRYPOINT ["python", "/app/transcribe.py"]
