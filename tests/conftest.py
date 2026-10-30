from uuid import uuid4

import pytest

from predictiveguard.config import Settings
from predictiveguard.schemas import ModelMetadata, ProcessRequest, ProcessResponse
from predictiveguard.tasks import TaskMessage


@pytest.fixture
def task():
    return TaskMessage(
        features=ProcessRequest(
            type="L",
            air_temperature_k=300,
            process_temperature_k=310,
            rotational_speed_rpm=1500,
            torque_nm=40,
            tool_wear_min=100,
        )
    )


@pytest.fixture
def prediction():
    return ProcessResponse(
        prediction="no_failure",
        failure_probability=0.05,
        model=ModelMetadata(name="test", alias="champion", version="1", run_id="test"),
    )


@pytest.fixture
def config():
    return Settings(
        task_queue="test." + uuid4().hex,
        inference_delay_seconds=0,
        retry_delay_seconds=0.001,
        queue_max_length=2,
    )
