from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from predictiveguard.main import app
from predictiveguard.schemas import ComponentHealth


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_healthz(client: AsyncClient) -> None:
    response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"]
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


@pytest.mark.asyncio
async def test_healthz_preserves_request_id(client: AsyncClient) -> None:
    response = await client.get("/healthz", headers={"X-Request-ID": "test-request"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request"


@pytest.mark.asyncio
async def test_version(client: AsyncClient) -> None:
    response = await client.get("/api/v1/version")

    assert response.status_code == 200
    assert response.json()["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_health(client: AsyncClient) -> None:
    component = ComponentHealth(
        name="postgresql",
        status="healthy",
        version="17.6",
        response_time_ms=1.234,
    )

    with patch(
        "predictiveguard.api.routes.get_postgres_health",
        new=AsyncMock(return_value=component),
    ):
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "components": [component.model_dump()],
    }


@pytest.mark.asyncio
async def test_health_when_postgres_is_unavailable(client: AsyncClient) -> None:
    component = ComponentHealth(
        name="postgresql",
        status="unavailable",
        version=None,
        response_time_ms=3000.0,
        detail="TimeoutError",
    )

    with patch(
        "predictiveguard.api.routes.get_postgres_health",
        new=AsyncMock(return_value=component),
    ):
        response = await client.get("/api/v1/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "components": [component.model_dump()],
    }
