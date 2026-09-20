from time import perf_counter
from typing import Any

from predictiveguard.config import settings


async def _connect() -> Any:
    import asyncpg

    return await asyncpg.connect(settings.postgres_dsn)


async def get_postgres_health() -> dict[str, str | float]:
    started = perf_counter()
    connection = await _connect()
    try:
        version = await connection.fetchval("SHOW server_version")
    finally:
        await connection.close()

    elapsed_ms = (perf_counter() - started) * 1000
    return {
        "name": "postgresql",
        "version": str(version),
        "response_time_ms": round(elapsed_ms, 3),
    }
