# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ARG GSPLAT_PACKAGE=gsplat==1.5.3
ARG TORCH_CUDA_ARCH_LIST=12.0
ARG UV_VERSION=0.8.15

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_DEVICE=cuda \
    SFMAPI_GSPLAT_OUTPUT_ROOT=/sfmapi/output \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
ENV PATH="/root/.local/bin:/opt/conda/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    python -c "import sys, torch; assert sys.version_info >= (3, 10), sys.version; assert torch.__version__.startswith('2.7.1'), torch.__version__; assert torch.version.cuda and torch.version.cuda.startswith('12.8'), torch.version.cuda" \
    && uv pip install --system --no-build-isolation --no-binary gsplat "${GSPLAT_PACKAGE}" \
    && uv pip install --system . \
    && python -c "import torch, gsplat; assert torch.cuda.is_available() or True; assert hasattr(gsplat, 'rasterization')"

EXPOSE 8080
CMD ["uvicorn", "sfmapi_gsplat.server:app", "--host", "0.0.0.0", "--port", "8080"]
