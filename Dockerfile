ARG PYTORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-docker.txt

COPY . .

CMD ["python", "-m", "src.train", "--help"]
