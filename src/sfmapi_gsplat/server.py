from __future__ import annotations

import argparse
from typing import Any

from fastapi import FastAPI

from sfmapi_gsplat import __version__
from sfmapi_gsplat.protocol import PROTOCOL, PROTOCOL_VERSION, ExecuteRequest, ExecuteResponse
from sfmapi_gsplat.trainer import runtime_info, train

app = FastAPI(title="sfmapi-gsplat", version=__version__)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "provider": "gsplat",
        "plugin": "sfmapi-gsplat",
        "version": __version__,
        "runtime": runtime_info(),
    }


@app.post("/execute", response_model=ExecuteResponse)
def execute(request: ExecuteRequest) -> ExecuteResponse:
    try:
        outputs = train(request)
    except Exception as exc:
        return ExecuteResponse(status="failed", error=f"{type(exc).__name__}: {exc}")
    return ExecuteResponse(status="succeeded", outputs=outputs)


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

