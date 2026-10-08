from importlib.metadata import version
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from predictiveguard.main import app
from predictiveguard.model_service import model_service
from predictiveguard.schemas import ComponentHealth, ModelMetadata, ProcessResponse


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
    assert response.json()["version"] == version("predictiveguard-ai")


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


@pytest.mark.asyncio
async def test_process(client: AsyncClient) -> None:
    prediction = ProcessResponse(
        prediction="failure",
        failure_probability=0.82,
        model=ModelMetadata(
            name="PredictiveGuardFailureModel",
            alias="champion",
            version="3",
            run_id="run-123",
        ),
    )
    payload = {
        "type": "L",
        "air_temperature_k": 302.1,
        "process_temperature_k": 314.2,
        "rotational_speed_rpm": 1100,
        "torque_nm": 64.3,
        "tool_wear_min": 220,
    }
    with (
        patch.object(type(model_service), "is_ready", new_callable=PropertyMock, return_value=True),
        patch.object(model_service, "predict", return_value=prediction),
    ):
        response = await client.post("/process", json=payload)

    assert response.status_code == 200
    assert response.json() == prediction.model_dump()


@pytest.mark.asyncio
async def test_process_when_model_is_not_loaded(client: AsyncClient) -> None:
    payload = {
        "type": "L",
        "air_temperature_k": 300,
        "process_temperature_k": 310,
        "rotational_speed_rpm": 1500,
        "torque_nm": 40,
        "tool_wear_min": 100,
    }
    with patch.object(
        type(model_service), "is_ready", new_callable=PropertyMock, return_value=False
    ):
        response = await client.post("/process", json=payload)

    assert response.status_code == 503
    assert response.json() == {"detail": "ML model is not loaded"}


@pytest.mark.asyncio
async def test_process_validates_sensor_ranges(client: AsyncClient) -> None:
    response = await client.post(
        "/process",
        json={
            "type": "L",
            "air_temperature_k": 999,
            "process_temperature_k": 310,
            "rotational_speed_rpm": 1500,
            "torque_nm": 40,
            "tool_wear_min": 100,
        },
    )

    assert response.status_code == 422
