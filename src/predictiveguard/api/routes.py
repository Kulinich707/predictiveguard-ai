from fastapi import APIRouter, Response, status

from predictiveguard.db import get_postgres_health
from predictiveguard.schemas import HealthResponse, LivenessResponse, VersionResponse
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
