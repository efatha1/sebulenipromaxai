"""U11 FastAPI application factory."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.dependencies import AppServices, ServiceError
from api.routes import router
from training.contracts import StructuredErrorDetail


def create_app(
    *,
    config_path: str | Path,
    active_model_manifest_path: str | Path | None = None,
    current_time=None,
) -> FastAPI:
    """Create the U11 FastAPI application."""
    app = FastAPI(title="Sebuleni Pro Max AI")
    app.include_router(router)
    app.state.services = AppServices(
        config_path=Path(config_path),
        active_model_manifest_path=Path(active_model_manifest_path) if active_model_manifest_path is not None else None,
        current_time=current_time,
    )

    @app.exception_handler(ServiceError)
    @app.exception_handler(ValueError)
    def handle_service_error(_request: Request, exc: Exception) -> JSONResponse:
        detail = StructuredErrorDetail(
            location=("body",),
            message=str(exc),
            input_value=None,
        )
        return JSONResponse(status_code=400, content={"errors": [detail.model_dump(mode="json")]})

    @app.exception_handler(RequestValidationError)
    def handle_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            StructuredErrorDetail(
                location=tuple(str(item) for item in error["loc"]),
                message=str(error["msg"]),
                input_value=error.get("input"),
            ).model_dump(mode="json")
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"errors": errors})

    return app


def create_app_from_env() -> FastAPI:
    """Create an app using environment-provided configuration paths."""
    config_path = os.environ.get("SEBULENI_CONFIG_PATH")
    if not config_path:
        raise ServiceError("SEBULENI_CONFIG_PATH must be set for API startup.")
    active_model_manifest_path = os.environ.get("SEBULENI_ACTIVE_MODEL_MANIFEST_PATH")
    return create_app(
        config_path=config_path,
        active_model_manifest_path=active_model_manifest_path,
    )


if os.environ.get("SEBULENI_CONFIG_PATH"):
    app = create_app_from_env()
else:  # pragma: no cover - default import path outside configured runtime
    app = FastAPI(title="Sebuleni Pro Max AI")
    app.include_router(router)
    app.state.services = None
