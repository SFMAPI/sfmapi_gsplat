from __future__ import annotations

from fastapi.testclient import TestClient

import sfmapi_gsplat.server as server


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

