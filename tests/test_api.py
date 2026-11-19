from contextlib import asynccontextmanager
from importlib.metadata import version
from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from predictiveguard.main import app
from predictiveguard.schemas import ComponentHealth
from predictiveguard.telemetry import trace_headers


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


async def test_confirmation_retry_uses_stored_task_trace(client, config, task, monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("predictiveguard.telemetry.trace.get_tracer", provider.get_tracer)
    row = {"status": "SUBMITTING", "trace_context": None}
    published = []

    @asynccontextmanager
    async def locked(task_id):
        yield None

    async def ensure(conn, task, status, *, trace_context):
        if row["trace_context"] is None:
            row["trace_context"] = trace_context
        return row

    async def state(conn, task_id, status):
        row["status"] = status

    async def publish(*args):
        published.append(trace_headers())
        if len(published) == 1:
            raise TimeoutError("AMQP confirm timeout")

    monkeypatch.setattr("predictiveguard.api.tasks.publish", publish)
    monkeypatch.setattr(
        app.state, "store", Mock(locked=locked, ensure=ensure, state=state), raising=False
    )
    monkeypatch.setattr(app.state, "broker", Mock(), raising=False)
    monkeypatch.setattr(app.state, "config", config, raising=False)
    kwargs = {"headers": {"Idempotency-Key": str(task.task_id)}, "json": task.features.model_dump()}
    try:
        first = await client.post("/api/v1/tasks", **kwargs)
        second = await client.post("/api/v1/tasks", **kwargs)
        third = await client.post("/api/v1/tasks", **kwargs)
        assert (first.status_code, second.status_code, third.status_code) == (503, 202, 202)
        assert (
            first.headers["X-Trace-ID"]
            == second.headers["X-Trace-ID"]
            == third.headers["X-Trace-ID"]
        )
        assert second.headers["X-Request-Trace-ID"] != second.headers["X-Trace-ID"]
        assert published == [row["trace_context"]] * 2
        first_span, second_span, third_span = exporter.get_finished_spans()
        for retry in (second_span, third_span):
            assert retry.links[0].context.trace_id == first_span.context.trace_id
    finally:
        provider.shutdown()
