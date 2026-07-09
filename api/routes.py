"""U11 REST API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.dependencies import AppServices, get_services
from api.schemas import HealthResponse, PredictRequest, PredictResponse

router = APIRouter()


class GenerateReportRequest(BaseModel):
    """Batch report generation request."""

    prediction_responses: tuple[PredictResponse, ...] = Field(min_length=1)
    report_name: str | None = None


class RetrainingRequestBody(BaseModel):
    """Retraining request creation payload."""

    reason: str = Field(min_length=1)
    candidate_model_id: str | None = None


@router.get("/health", response_model=HealthResponse)
def health(services: AppServices = Depends(get_services)) -> HealthResponse:
    """Return service health and model availability."""
    return HealthResponse.model_validate(services.health().model_dump(mode="json"))


@router.post("/predict", response_model=PredictResponse | list[PredictResponse])
def predict_route(
    request: PredictRequest,
    services: AppServices = Depends(get_services),
) -> PredictResponse | list[PredictResponse]:
    """Run prediction through the shared U10 inference contract."""
    responses = services.predict_request(request)
    payload = [PredictResponse.model_validate(item.model_dump(mode="json")) for item in responses]
    if len(payload) == 1:
        return payload[0]
    return payload


@router.get("/models/current")
def current_model(services: AppServices = Depends(get_services)) -> dict[str, Any]:
    """Return active-model metadata for inspection."""
    return services.inspect_current_model()


@router.post("/reports/generate")
def generate_report(
    request: GenerateReportRequest,
    services: AppServices = Depends(get_services),
) -> dict[str, Any]:
    """Generate a batch report from approved prediction outputs only."""
    artifact = services.generate_prediction_report(
        request.prediction_responses,
        report_name=request.report_name,
    )
    return {
        "report_id": artifact.report_id,
        "report_type": artifact.report_type,
        "output_path": str(artifact.output_path),
        "generated_at": artifact.generated_at.isoformat(),
        "summary": artifact.summary,
    }


@router.post("/retraining/request")
def request_retraining(
    request: RetrainingRequestBody,
    services: AppServices = Depends(get_services),
) -> dict[str, str]:
    """Create a retraining request without entering U12 review logic."""
    return services.create_retraining_request(
        reason=request.reason,
        candidate_model_id=request.candidate_model_id,
    )
