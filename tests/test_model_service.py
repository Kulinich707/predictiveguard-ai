from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from predictiveguard.config import Settings
from predictiveguard.model_service import ModelService
from predictiveguard.schemas import ProcessRequest


def test_load_model_from_registry_alias() -> None:
    service = ModelService()
    config = Settings(
        mlflow_tracking_uri="http://mlflow:5000",
        model_name="test-model",
        model_alias="champion",
    )
    model = Mock()
    version = SimpleNamespace(version="7", run_id="run-7")

    with (
        patch("predictiveguard.model_service.mlflow.set_tracking_uri") as set_uri,
        patch("predictiveguard.model_service.MlflowClient") as client,
        patch("predictiveguard.model_service.load_model", return_value=model) as load,
    ):
        client.return_value.get_model_version_by_alias.return_value = version
        service.load(config)

    set_uri.assert_called_once_with("http://mlflow:5000")
    client.return_value.get_model_version_by_alias.assert_called_once_with("test-model", "champion")
    load.assert_called_once_with("models:/test-model/7")
    assert service.is_ready


def test_predict_returns_probability_and_model_metadata() -> None:
    service = ModelService()
    model = Mock()
    model.predict.return_value = pd.DataFrame({"failure_probability": [0.82], "prediction": [1]})
    config = Settings(model_name="test-model", model_alias="champion")
    version = SimpleNamespace(version="2", run_id="run-2")

    with (
        patch("predictiveguard.model_service.MlflowClient") as client,
        patch("predictiveguard.model_service.load_model", return_value=model),
    ):
        client.return_value.get_model_version_by_alias.return_value = version
        service.load(config)

    result = service.predict(
        ProcessRequest(
            type="L",
            air_temperature_k=302,
            process_temperature_k=314,
            rotational_speed_rpm=1100,
            torque_nm=64,
            tool_wear_min=220,
        )
    )

    assert result.prediction == "failure"
    assert result.failure_probability == 0.82
    assert result.model.version == "2"
    assert model.predict.call_count == 1


def test_predict_requires_loaded_model() -> None:
    service = ModelService()
    request = ProcessRequest(
        type="L",
        air_temperature_k=300,
        process_temperature_k=310,
        rotational_speed_rpm=1500,
        torque_nm=40,
        tool_wear_min=100,
    )

    try:
        service.predict(request)
    except RuntimeError as error:
        assert str(error) == "Model is not loaded"
    else:
        raise AssertionError("RuntimeError was not raised")
