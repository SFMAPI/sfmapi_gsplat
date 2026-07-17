> [!IMPORTANT]
> **MOVED.** Development continued and now lives in the SceneAPI org as part of the merged radiance family: <https://github.com/SceneAPI/3DGS>. This repository holds the final sfmapi-era state for history and will be archived.

# sfmapi-gsplat

`sfmapi-gsplat` is an `sfmapi-plugin-http-v1` container-service plugin for CUDA-backed gsplat radiance-field training.

The service intentionally fails when CUDA, PyTorch, `gsplat`, or an input image dataset is missing. It does not silently fall back to pseudo training.
When `backend_options.dataset_path` points to a COLMAP dataset, training uses the sparse model and reports interval-based multi-image metrics (`eval_protocol=colmap_interval`).
The old one-image CUDA smoke path is available only when `single_image_smoke=true`.
The Docker image uses PyTorch 2.7.1 + CUDA 12.8 and builds `gsplat` from source for `TORCH_CUDA_ARCH_LIST=12.0`, which supports Blackwell GPUs such as RTX 5090.

## Endpoints

- `GET /healthz`: liveness.
- `GET /version`: protocol and runtime information.
- `POST /execute`: executes `radiance_train` for provider `gsplat`.

## Build

```powershell
docker build -t sfmapi-plugin-gsplat:main .
docker run --rm --gpus all -p 127.0.0.1:8098:8080 `
  -v C:\data\bicycle:/data/bicycle `
  -v ${PWD}\outputs:/sfmapi/output `
  sfmapi-plugin-gsplat:main
```

For Linux containers, mount Linux-style paths:

```bash
docker run --rm --gpus all -p 127.0.0.1:8098:8080 \
  -v /data/bicycle:/data/bicycle \
  -v "$PWD/outputs:/sfmapi/output" \
  sfmapi-plugin-gsplat:main
```

## Execute Payload

`sfmapi` sends this service through `/execute`. The training path expects an image source in `backend_options`:

```json
{
  "protocol": "sfmapi-plugin-http-v1",
  "task_kind": "radiance_train",
  "capability": "radiance.train",
  "provider": "gsplat",
  "inputs": {
    "project_id": "project",
    "radiance_field_id": "field",
    "dataset_id": "dataset"
  },
  "spec": {
    "method": "gsplat.train.default",
    "max_steps": 3000,
    "backend_options": {
      "dataset_path": "/data/bicycle",
      "image_root": "/data/bicycle/images_2",
      "output_path": "/sfmapi/output",
      "target_size": 384,
      "num_gaussians": 1000000,
      "test_every": 8
    }
  }
}
```

For a CUDA smoke test without a COLMAP dataset, set `"single_image_smoke": true` and either provide one target image or set `"allow_synthetic_target": true`; this remains explicit and is never the default.

## Development

```bash
uv python install 3.10
uv sync --python 3.10 --extra test
uv run ruff check .
uv run pytest
```
