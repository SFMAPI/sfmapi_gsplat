# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG GSPLAT_INDEX_URL=https://docs.gsplat.studio/whl/pt24cu124
ARG TORCH_PACKAGE=torch==2.4.1
ARG UV_VERSION=0.8.15

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_DEVICE=cuda \
    SFMAPI_GSPLAT_OUTPUT_ROOT=/sfmapi/output \
    DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl python3.10 python3.10-dev python3.10-venv cuda-cupti-12-4 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN uv venv /opt/venv --python python3.10
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install filelock fsspec jinja2 networkx sympy typing-extensions triton==3.0.0 \
    && uv pip install --index-url "${TORCH_INDEX_URL}" --no-deps "${TORCH_PACKAGE}" \
    && uv pip install --index-url "${GSPLAT_INDEX_URL}" --no-deps gsplat==1.5.3 \
    && uv pip install . \
    && python -c "import sys, torch, gsplat; assert sys.version_info[:2] == (3, 10), sys.version; assert torch.__version__.startswith('2.4.1'), torch.__version__; assert torch.version.cuda and torch.version.cuda.startswith('12.4'), torch.version.cuda; assert hasattr(gsplat, 'rasterization')"

EXPOSE 8080
CMD ["uvicorn", "sfmapi_gsplat.server:app", "--host", "0.0.0.0", "--port", "8080"]
