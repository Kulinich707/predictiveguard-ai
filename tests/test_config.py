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
