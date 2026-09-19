"""Run the local Jev-compatible Core ML HTTP API."""

import os

import uvicorn

from jev_coreml.server import create_app


if __name__ == "__main__":
    uvicorn.run(
        create_app(os.environ.get("JEV_COREML_COMPUTE_UNITS", "cpu_and_ne")),
        host=os.environ.get("JEV_COREML_HOST", "127.0.0.1"),
        port=int(os.environ.get("JEV_COREML_PORT", "8787")),
    )
