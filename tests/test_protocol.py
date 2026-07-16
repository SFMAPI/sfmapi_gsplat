from __future__ import annotations

import subprocess

import pytest
from fastapi.testclient import TestClient
from sfmapi.plugin_service import PROTOCOL, PROTOCOL_VERSION

import sfmapi_gsplat.server as server
import sfmapi_gsplat.trainer as trainer
from sfmapi_gsplat.plugin import MANIFEST


def test_health_and_version() -> None:
    client = TestClient(server.app)

    health = client.get("/healthz")
    version = client.get("/version")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert version.status_code == 200
    assert version.json()["protocol"] == PROTOCOL
    assert version.json()["protocol_version"] == PROTOCOL_VERSION == "1.1"
    assert version.json()["plugin_id"] == "gsplat"
    assert version.json()["runtime"]["provider"] == "gsplat"
    assert MANIFEST["runtime_modes"]["container_service"]["protocol_version"] == PROTOCOL_VERSION


def test_capabilities_serves_the_manifest_capability_set() -> None:
    features = TestClient(server.app).get("/capabilities").json()["features"]

    assert features == sorted(MANIFEST["capabilities"])


def test_execute_rejects_wrong_protocol() -> None:
    response = TestClient(server.app).post(
        "/execute", json={"protocol": "nope", "task_kind": "x"}
    )

    assert response.status_code == 400
    assert response.json()["error"] == "protocol_mismatch"


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
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
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
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "task_kind": "radiance_train",
            "capability": "radiance.train",
            "provider": "other",
            "inputs": {},
            "spec": {},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "request.provider must be 'gsplat'" in body["error"]


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
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
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
