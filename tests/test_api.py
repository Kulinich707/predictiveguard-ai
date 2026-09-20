from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from predictiveguard.main import app

client = TestClient(app)


def test_healthz() -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version() -> None:
    response = client.get("/api/v1/version")

    assert response.status_code == 200
    assert response.json()["version"] == "0.1.0"


def test_health() -> None:
    component = {
        "name": "postgresql",
        "version": "17.6",
        "response_time_ms": 1.234,
    }

    with patch(
        "predictiveguard.api.routes.get_postgres_health",
        new=AsyncMock(return_value=component),
    ):
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "components": [component],
    }


def test_health_when_postgres_is_unavailable() -> None:
    with patch(
        "predictiveguard.api.routes.get_postgres_health",
        new=AsyncMock(side_effect=RuntimeError("connection failed")),
    ):
        response = client.get("/api/v1/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "PostgreSQL is unavailable"}
