# RunPod Serverless ACE-Step 1.5 with LoRA hot-swap
# Base image: PyTorch 2.7 with CUDA 12.8 (matches ACE-Step 1.5 dependencies)
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone ACE-Step 1.5 (specific commit can be pinned for reproducibility)
RUN git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /app/ACE-Step-1.5

# Install ACE-Step dependencies (skip torch reinstall — already in base image)
RUN cd /app/ACE-Step-1.5 && \
    pip install --no-cache-dir \
        "transformers>=4.51.0,<4.58.0" \
        "diffusers" \
        "scipy>=1.10.1" \
        "soundfile>=0.13.1" \
        "loguru>=0.7.3" \
        "einops>=0.8.1" \
        "accelerate>=1.12.0" \
        "diskcache" \
        "numba>=0.63.1" \
        "vector-quantize-pytorch>=1.27.15" \
        "torchcodec>=0.9.1" \
        "torchao" \
        "toml" \
        "peft>=0.7.0" \
        "lightning>=2.0.0" \
        "modelscope" \
        "matplotlib>=3.7.5" \
        "huggingface_hub" \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.27.0"

# Install nano-vllm from the local third_parts directory
RUN cd /app/ACE-Step-1.5 && \
    pip install --no-cache-dir ./acestep/third_parts/nano-vllm || \
    echo "nano-vllm install skipped (optional)"

# RunPod SDK
RUN pip install --no-cache-dir runpod

# Install ACE-Step package itself (editable so we can patch handler.py later)
RUN cd /app/ACE-Step-1.5 && pip install --no-deps -e .

# Copy the handler
COPY handler.py /app/handler.py

# Default env vars (override at runtime)
ENV ACESTEP_VOLUME_ROOT=/runpod-volume
ENV ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo
ENV ACESTEP_LM_MODEL=acestep-5Hz-lm-0.6B
ENV PYTHONIOENCODING=utf-8

# Run handler
CMD ["python", "-u", "/app/handler.py"]
