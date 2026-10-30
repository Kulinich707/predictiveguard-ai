import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def test_compose_forwards_runtime_settings_and_local_database_port(tmp_path):
    env_file = tmp_path / "empty.env"
    env_file.write_text("")
    environment = dict(
        os.environ,
        TASK_QUEUE="test.config.changed",
        SHUTDOWN_TIMEOUT_SECONDS="90",
        WORKER_STOP_GRACE_PERIOD="105s",
        WORKER_HEALTH_PORT="18001",
        RETRY_DELAY_SECONDS="0.25",
        POSTGRES_LOCAL_PORT="55432",
    )

    def config(local=False):
        command = [
            "docker",
            "compose",
            "--env-file",
            str(env_file),
            "-f",
            str(ROOT / "docker-compose.yml"),
        ]
        if local:
            command.extend(["-f", str(ROOT / "docker-compose.local.yml")])
        command.extend(["--profile", "batch", "config", "--format", "json"])
        result = subprocess.run(
            command, env=environment, check=True, capture_output=True, text=True
        )
        return json.loads(result.stdout)["services"]

    services = config()
    for service in ("app", "worker", "source"):
        assert services[service]["environment"]["TASK_QUEUE"] == "test.config.changed"
        assert float(services[service]["environment"]["RETRY_DELAY_SECONDS"]) == 0.25
    worker = services["worker"]
    assert float(worker["environment"]["SHUTDOWN_TIMEOUT_SECONDS"]) == 90
    assert worker["environment"]["WORKER_HEALTH_PORT"] == "18001"
    assert worker["stop_grace_period"] in ("105s", "1m45s")
    assert "WORKER_HEALTH_PORT" in " ".join(worker["healthcheck"]["test"])
    assert not services["app-db"].get("ports")
    ports = config(local=True)["app-db"]["ports"]
    assert len(ports) == 1
    assert ports[0]["host_ip"] == "127.0.0.1"
    assert int(ports[0]["published"]) == 55432
    assert int(ports[0]["target"]) == 5432
