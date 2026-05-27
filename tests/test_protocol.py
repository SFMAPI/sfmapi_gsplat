from __future__ import annotations

import subprocess

import pytest
from fastapi.testclient import TestClient

import sfmapi_gsplat.server as server
import sfmapi_gsplat.trainer as trainer


def test_health_and_version() -> None:
    client = TestClient(server.app)

    health = client.get("/healthz")
    version = client.get("/version")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert version.status_code == 200
    assert version.json()["protocol"] == "sfmapi-plugin-http-v1"
    assert version.json()["protocol_version"] == "1.0"
    assert version.json()["provider"] == "gsplat"


def test_gpu_runtime_info_reports_visible_gpu(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="NVIDIA Test GPU\n", stderr="")

    monkeypatch.setattr(trainer.subprocess, "run", fake_run)

    info = trainer._gpu_runtime_info()

    assert info["gpu_runtime_available"] is True
    assert info["gpu_device"] == "NVIDIA Test GPU"


def test_require_gpu_runtime_rejects_missing_gpu(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="no gpu")

    monkeypatch.setattr(trainer.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="GPU runtime is required"):
        trainer._require_gpu_runtime()


def test_execute_surfaces_missing_cuda_as_plugin_failure(monkeypatch) -> None:
    def fake_train(_request):
        raise RuntimeError("CUDA is required for sfmapi-gsplat training")

    monkeypatch.setattr(server, "train", fake_train)
    client = TestClient(server.app)

    response = client.post(
        "/execute",
        json={
            "protocol": "sfmapi-plugin-http-v1",
            "task_kind": "radiance_train",
            "capability": "radiance.train",
            "provider": "gsplat",
            "inputs": {"project_id": "p", "radiance_field_id": "rf"},
            "spec": {"method": "gsplat.train.default", "max_steps": 1},
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert "CUDA is required" in response.json()["error"]


def test_execute_rejects_wrong_provider() -> None:
    client = TestClient(server.app)

    response = client.post(
        "/execute",
        json={
            "protocol": "sfmapi-plugin-http-v1",
            "task_kind": "radiance_train",
            "capability": "radiance.train",
            "provider": "other",
            "inputs": {},
            "spec": {},
        },
    )

    assert response.status_code == 422


def test_execute_dispatches_radiance_eval(monkeypatch) -> None:
    def fake_evaluate(request):
        return {
            "radiance_field_id": request.inputs["radiance_field_id"],
            "evaluation_id": request.inputs["evaluation_id"],
            "snapshot_seq": 1,
            "metrics": {
                "psnr_db": 30.0,
                "ssim": 1.0,
                "lpips": 0.0,
                "num_images": 1,
                "duration_s": 0.0,
                "render_time_s_total": 0.0,
                "render_time_s_mean": 0.0,
            },
            "artifacts": [],
        }

    monkeypatch.setattr(server, "evaluate", fake_evaluate)
    client = TestClient(server.app)

    response = client.post(
        "/execute",
        json={
            "protocol": "sfmapi-plugin-http-v1",
            "task_kind": "radiance_eval",
            "capability": "radiance.evaluate",
            "provider": "gsplat",
            "inputs": {
                "project_id": "p",
                "radiance_field_id": "rf",
                "evaluation_id": "ev",
            },
            "spec": {"snapshot_seq": 1},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["outputs"]["evaluation_id"] == "ev"
    assert body["outputs"]["metrics"]["psnr_db"] == 30.0


def test_normalized_metrics_keep_single_image_smoke_scope() -> None:
    metrics = trainer._normalize_metrics_payload(
        {"psnr_db": 30.0, "ssim": 0.9, "lpips": 0.1},
        duration_s=0.25,
    )

    assert metrics["num_images"] == 1
    assert metrics["eval_protocol"] == "single_image_smoke"
