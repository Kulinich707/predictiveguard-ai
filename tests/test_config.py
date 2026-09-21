from predictiveguard.config import Settings


def test_postgres_dsn() -> None:
    settings = Settings(
        postgres_host="db",
        postgres_port=5432,
        postgres_db="predictiveguard",
        postgres_user="user",
        postgres_password="pass",
    )

    assert settings.postgres_dsn == "postgresql://user:pass@db:5432/predictiveguard"


def test_operational_defaults() -> None:
    settings = Settings()

    assert settings.log_level == "INFO"
    assert settings.postgres_connect_timeout_seconds == 3.0
    assert settings.postgres_command_timeout_seconds == 3.0
    assert settings.postgres_health_timeout_seconds == 5.0
