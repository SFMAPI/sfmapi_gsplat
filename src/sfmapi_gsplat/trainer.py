from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sfmapi_gsplat.protocol import ExecuteRequest

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_required": True}
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
        info["cuda_available"] = False
    try:
        import gsplat

        info["gsplat"] = getattr(gsplat, "__version__", "installed")
    except Exception as exc:
        info["gsplat_error"] = f"{type(exc).__name__}: {exc}"
    return info


def train(request: ExecuteRequest) -> dict[str, Any]:
    torch, rasterization = _require_cuda_gsplat()
    spec = request.spec
    inputs = request.inputs
    options = spec.get("backend_options") if isinstance(spec.get("backend_options"), dict) else {}
    radiance_field_id = _required_str(inputs, "radiance_field_id")
    project_id = _required_str(inputs, "project_id")
    max_steps = int(spec.get("max_steps") or 3000)
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    target = _target_tensor(torch, options)
    height, width = int(target.shape[0]), int(target.shape[1])
    num_gaussians = int(options.get("num_gaussians") or 2048)
    lr = float(options.get("learning_rate") or 0.03)
    log_interval = max(1, int(options.get("log_interval") or max_steps // 10 or 1))

    device = torch.device("cuda")
    means = torch.nn.Parameter(torch.randn(num_gaussians, 3, device=device) * 0.45)
    with torch.no_grad():
        means[:, 2].add_(2.0)
    raw_scales = torch.nn.Parameter(torch.full((num_gaussians, 3), -4.0, device=device))
    raw_quats = torch.nn.Parameter(torch.zeros(num_gaussians, 4, device=device))
    with torch.no_grad():
        raw_quats[:, 0] = 1.0
    raw_opacity = torch.nn.Parameter(torch.zeros(num_gaussians, device=device))
    raw_colors = torch.nn.Parameter(torch.rand(num_gaussians, 3, device=device))
    optimizer = torch.optim.Adam(
        [means, raw_scales, raw_quats, raw_opacity, raw_colors],
        lr=lr,
    )

    viewmats = torch.eye(4, device=device, dtype=torch.float32)[None]
    focal = float(options.get("focal") or max(width, height) * 0.9)
    Ks = torch.tensor(
        [[[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )

    metrics: list[dict[str, float | int]] = []
    first_loss: float | None = None
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        quats = torch.nn.functional.normalize(raw_quats, dim=-1)
        scales = torch.nn.functional.softplus(raw_scales) + 1e-4
        opacities = torch.sigmoid(raw_opacity)
        colors = torch.sigmoid(raw_colors)
        rendered, _alphas, _meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            packed=False,
        )
        rgb = rendered[0, ..., :3].clamp(0, 1)
        loss = torch.mean((rgb - target) ** 2)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if first_loss is None:
            first_loss = loss_value
        if step == 1 or step % log_interval == 0 or step == max_steps:
            psnr = -10.0 * np.log10(max(loss_value, 1e-12))
            metrics.append({"step": step, "loss": loss_value, "psnr": float(psnr)})

    torch.cuda.synchronize()
    seq = int(options.get("snapshot_seq") or 1)
    snapshot_path = _snapshot_path(options, project_id, radiance_field_id, seq)
    snapshot_path.mkdir(parents=True, exist_ok=True)
    means_np = means.detach().cpu().numpy()
    colors_np = (torch.sigmoid(raw_colors).detach().cpu().numpy() * 255).clip(0, 255)
    _write_ply(snapshot_path / "point_cloud.ply", means_np, colors_np)
    summary = {
        "provider": "gsplat",
        "method": str(spec.get("method") or "gsplat.train.default"),
        "radiance_field_id": radiance_field_id,
        "dataset_id": inputs.get("dataset_id"),
        "max_steps": max_steps,
        "completed_steps": max_steps,
        "loss_initial": first_loss,
        "loss_final": metrics[-1]["loss"],
        "psnr_final": metrics[-1]["psnr"],
        "vertex_count": int(num_gaussians),
        "format": "ply",
        "target_size": [width, height],
    }
    _write_json(snapshot_path / "summary.json", summary)
    _write_json(snapshot_path / "metrics.json", {"samples": metrics, "max_steps": max_steps})
    _write_json(snapshot_path / "metadata.json", {"runtime": runtime_info(), **summary})
    ply_uri = str(snapshot_path / "point_cloud.ply")
    return {
        "radiance_field_id": radiance_field_id,
        "snapshot_seq": seq,
        "snapshot_path": str(snapshot_path),
        "summary": summary,
        "artifacts": [
            {
                "kind": "radiance.snapshot",
                "name": f"snapshot-{seq}",
                "uri": str(snapshot_path),
                "artifact_format": "sfmapi.radiance.snapshot.v1",
                "metadata": {"radiance_field_id": radiance_field_id, "snapshot_seq": seq},
                "summary": summary,
            },
            {
                "kind": "radiance.variant.ply",
                "name": "point_cloud.ply",
                "uri": ply_uri,
                "media_type": "application/octet-stream",
                "artifact_format": "sfmapi.radiance.variant.ply.v1",
                "metadata": {"radiance_field_id": radiance_field_id, "snapshot_seq": seq},
                "summary": {"vertex_count": int(num_gaussians)},
            },
        ],
        "variants": [
            {
                "format": "ply",
                "uri": ply_uri,
                "media_type": "application/octet-stream",
                "summary": {"vertex_count": int(num_gaussians)},
            }
        ],
    }


def _require_cuda_gsplat() -> tuple[Any, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for sfmapi-gsplat training")
    try:
        from gsplat import rasterization
    except Exception as exc:
        raise RuntimeError("Python package 'gsplat' is required for training") from exc
    return torch, rasterization


def _target_tensor(torch: Any, options: dict[str, Any]) -> Any:
    path = _target_image_path(options)
    if path is None:
        if options.get("allow_synthetic_target") is True:
            size = int(options.get("target_size") or 128)
            arr = np.zeros((size, size, 3), dtype=np.float32)
            yy, xx = np.mgrid[0:size, 0:size]
            arr[..., 0] = xx / max(size - 1, 1)
            arr[..., 1] = yy / max(size - 1, 1)
            arr[..., 2] = 0.25
            return torch.from_numpy(arr).to("cuda")
        raise RuntimeError(
            "gsplat training requires backend_options.image_path or dataset_path; "
            "set allow_synthetic_target=true only for explicit CUDA smoke tests"
        )
    size = int(options.get("target_size") or 256)
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    image = image.resize((image.width, image.height), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to("cuda")


def _target_image_path(options: dict[str, Any]) -> Path | None:
    for key in ("image_path", "target_image"):
        raw = options.get(key)
        if isinstance(raw, str) and raw:
            path = Path(raw)
            if not path.is_file():
                raise FileNotFoundError(f"{key} does not exist: {path}")
            return path
    for key in ("dataset_path", "image_root"):
        raw = options.get(key)
        if not isinstance(raw, str) or not raw:
            continue
        root = Path(raw)
        if not root.is_dir():
            raise FileNotFoundError(f"{key} does not exist: {root}")
        for item in sorted(root.rglob("*")):
            if item.suffix.lower() in IMAGE_EXTENSIONS and item.is_file():
                return item
        raise FileNotFoundError(f"{key} contains no supported images: {root}")
    return None


def _snapshot_path(
    options: dict[str, Any],
    project_id: str,
    radiance_field_id: str,
    seq: int,
) -> Path:
    root = Path(
        str(
            options.get("output_path")
            or os.environ.get("SFMAPI_GSPLAT_OUTPUT_ROOT")
            or "/sfmapi/output"
        )
    )
    return root / project_id / radiance_field_id / "snapshots" / str(seq)


def _required_str(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"inputs.{key} is required")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_ply(path: Path, means: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as fp:
        fp.write("ply\nformat ascii 1.0\n")
        fp.write(f"element vertex {len(means)}\n")
        fp.write("property float x\nproperty float y\nproperty float z\n")
        fp.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fp.write("end_header\n")
        for xyz, rgb in zip(means, colors, strict=True):
            fp.write(
                f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )

