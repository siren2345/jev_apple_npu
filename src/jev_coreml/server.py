"""Local HTTP server exposing the Jev System One request shape."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .api import LocalJev
from .semantic import load


class SystemOneRequest(BaseModel):
    state: Any
    questions: dict[str, dict[str, Any]]


def create_app(compute_units: str = "cpu_and_ne") -> FastAPI:
    app = FastAPI(title="jev-coreml", version="0.1.0")
    model = LocalJev(load(compute_units=compute_units))

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "model": "jev-coreml-minilm-v0"}

    @app.post("/v1/systemone")
    def system_one(request: SystemOneRequest) -> dict[str, Any]:
        try:
            return model.system_one(request.state, request.questions)
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app
