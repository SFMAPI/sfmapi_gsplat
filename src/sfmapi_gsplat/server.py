from __future__ import annotations

import argparse
from typing import Any

from sfmapi.plugin_service import ManifestBackend, build_plugin_server

from sfmapi_gsplat import __version__
from sfmapi_gsplat.plugin import MANIFEST
from sfmapi_gsplat.trainer import PROVIDER, ExecuteRequest, evaluate, runtime_info, train


def execute_task(
    *,
    task_kind: str,
    capability: str,
    inputs: dict[str, Any],
    spec: dict[str, Any],
    tenant_id: str,
    job_id: str,
    task_id: str,
    provider: str,
) -> dict[str, Any]:
    """Kit executor: dispatch one task to the trainer, mapping trainer errors
    onto the ``status: failed`` result the sfmapi worker expects."""
    request = ExecuteRequest(
        task_kind=task_kind,
        capability=capability,
        inputs=inputs,
        spec=spec,
        tenant_id=tenant_id,
        job_id=job_id,
        task_id=task_id,
        provider=provider,
    )
    try:
        if provider != PROVIDER:
            raise ValueError(f"request.provider must be {PROVIDER!r}")
        outputs = evaluate(request) if task_kind == "radiance_eval" else train(request)
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "succeeded", "outputs": outputs}


app = build_plugin_server(
    ManifestBackend(MANIFEST, version=__version__),
    plugin_id=MANIFEST["plugin_id"],
    package_version=__version__,
    executor=execute_task,
    runtime_info=runtime_info,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
