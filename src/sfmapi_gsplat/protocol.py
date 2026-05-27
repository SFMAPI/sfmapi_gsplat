from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL = "sfmapi-plugin-http-v1"
PROTOCOL_VERSION = "1.0"


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol: Literal["sfmapi-plugin-http-v1"]
    task_kind: Literal["radiance_train", "radiance_eval"]
    capability: Literal["radiance.train", "radiance.evaluate"]
    tenant_id: str | None = None
    job_id: str | None = None
    task_id: str | None = None
    provider: Literal["gsplat"]
    inputs: dict[str, Any] = Field(default_factory=dict)
    spec: dict[str, Any] = Field(default_factory=dict)


class ExecuteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["succeeded", "failed"]
    outputs: dict[str, Any] | None = None
    error: str | None = None
