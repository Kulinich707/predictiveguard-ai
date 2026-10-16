from fastapi import APIRouter, HTTPException, Response, status
from starlette.concurrency import run_in_threadpool

from predictiveguard.db import get_postgres_health
from predictiveguard.model_service import model_service
from predictiveguard.schemas import (
    HealthResponse,
    LivenessResponse,
    ProcessRequest,
    ProcessResponse,
    VersionResponse,
)
from predictiveguard.version import get_app_version

router = APIRouter()


@router.get("/healthz", response_model=LivenessResponse)
async def healthz() -> LivenessResponse:
    return LivenessResponse(status="ok")


@router.get("/api/v1/version", response_model=VersionResponse)
async def app_version() -> VersionResponse:
    return VersionResponse(version=get_app_version())


@router.get("/api/v1/health", response_model=HealthResponse)
async def health(response: Response) -> HealthResponse:
    postgres = await get_postgres_health()
    if postgres.status == "unavailable":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="degraded", components=[postgres])
    return HealthResponse(status="ok", components=[postgres])


@router.post("/process", response_model=ProcessResponse)
async def process(payload: ProcessRequest) -> ProcessResponse:
    if not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model is not loaded",
        )
    return await run_in_threadpool(model_service.predict, payload)
