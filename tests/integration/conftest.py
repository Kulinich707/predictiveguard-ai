import pytest
from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.rabbitmq import RabbitMqContainer


@pytest.fixture(scope="session")
def infrastructure():
    with (
        RabbitMqContainer("rabbitmq:4.1-management") as rabbit,
        PostgresContainer("postgres:17-alpine") as postgres,
    ):
        yield rabbit, postgres


@pytest.fixture
def real_config(config, infrastructure):
    rabbit, pg = infrastructure
    return config.model_copy(
        update={
            "rabbit_url": f"amqp://guest:guest@{rabbit.get_container_host_ip()}:"
            f"{rabbit.get_exposed_port(5672)}/",
            "postgres_host": pg.get_container_host_ip(),
            "postgres_port": int(pg.get_exposed_port(5432)),
            "postgres_db": pg.dbname,
            "postgres_user": pg.username,
            "postgres_password": pg.password,
        }
    )
