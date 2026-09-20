from fastapi import APIRouter, HTTPException

from predictiveguard.db import get_postgres_health
from predictiveguard.version import get_app_version

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/api/v1/version")
async def app_version() -> dict[str, str]:
    return {"version": get_app_version()}


@router.get("/api/v1/health")
async def health() -> dict[str, str | list[dict[str, str | float]]]:
    try:
        postgres = await get_postgres_health()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable") from exc

    return {
        "status": "ok",
        "components": [postgres],
    }
