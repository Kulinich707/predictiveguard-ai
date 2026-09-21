import asyncio
import logging
from time import perf_counter
from typing import Any

from predictiveguard.config import settings
from predictiveguard.schemas import ComponentHealth

logger = logging.getLogger(__name__)


async def _connect() -> Any:
    import asyncpg

    return await asyncpg.connect(
        settings.postgres_dsn,
        timeout=settings.postgres_connect_timeout_seconds,
        command_timeout=settings.postgres_command_timeout_seconds,
    )


async def get_postgres_health() -> ComponentHealth:
    started = perf_counter()
    try:
        async with asyncio.timeout(settings.postgres_health_timeout_seconds):
            connection = await _connect()
            try:
                version = await connection.fetchval("SHOW server_version")
            finally:
                await connection.close()
    except Exception as exc:
        elapsed_ms = (perf_counter() - started) * 1000
        logger.warning(
            "Dependency health check failed",
            extra={"component": "postgresql", "duration_ms": round(elapsed_ms, 3)},
            exc_info=True,
        )
        return ComponentHealth(
            name="postgresql",
            status="unavailable",
            version=None,
            response_time_ms=round(elapsed_ms, 3),
            detail=type(exc).__name__,
        )

    elapsed_ms = (perf_counter() - started) * 1000
    return ComponentHealth(
        name="postgresql",
        status="healthy",
        version=str(version),
        response_time_ms=round(elapsed_ms, 3),
    )
