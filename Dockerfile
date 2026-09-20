FROM python:3.12-slim

WORKDIR /app

# build-essential required to compile some chromadb / sentence-transformers
# dependencies when no prebuilt wheel exists for the target platform
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch BEFORE requirements.txt. The default PyPI torch
# wheel for Linux ships CUDA libraries (several GB) that Cloud Run can
# never use, since it runs on CPU. Once torch is present, pip treats the
# sentence-transformers dependency as already satisfied and skips it.
# Kept in its own layer, before COPY requirements.txt, so it stays cached
# when other dependencies change.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data

ENV PYTHONUNBUFFERED=1 \
    CHROMA_PERSIST_DIR=/app/data/processed/chroma \
    ANONYMIZED_TELEMETRY=False

EXPOSE 8080

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]