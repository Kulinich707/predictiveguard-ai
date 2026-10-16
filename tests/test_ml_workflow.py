from types import SimpleNamespace
from unittest.mock import patch

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from fastapi.testclient import TestClient
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression

from ml.train import (
    DATA_PATH,
    SOURCE_SHA256,
    build_pipeline,
    choose_threshold,
    evaluate,
    lineage,
    load_data,
    save_bundle,
    select_candidates,
    split_data,
    validation_gate,
)
from predictiveguard.config import Settings
from predictiveguard.inference import FEATURES, FailurePredictor
from predictiveguard.main import app
from predictiveguard.model_service import ModelService, model_service
from predictiveguard.schemas import ProcessRequest


def test_exact_source_no_leakage_and_disjoint_splits(tmp_path):
    raw, frame = load_data()
    assert len(raw) == 10000
    assert raw["Machine failure"].sum() == 339
    assert list(frame.columns) == FEATURES + ["failure"]
    splits = split_data(frame)
    assert [len(part) for part in splits.values()] == [6000, 2000, 2000]
    ids = [set(part.index) for part in splits.values()]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert set.union(*ids) == set(raw.UDI)
    assert lineage(splits, DATA_PATH)["source_sha256"] == SOURCE_SHA256
    assert split_data(frame)["training"].equals(splits["training"])
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("wrong data")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_data(wrong)
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "missing.csv")


def test_threshold_gate_and_selection_ignore_test():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.12, 0.2, 0.3])
    threshold = choose_threshold(y, p)
    assert threshold == 0.2
    metrics = evaluate(y, p, threshold)
    assert metrics["f1"] == 1
    assert validation_gate(metrics, 0.5, True)
    assert not validation_gate(metrics, 0.5, False)
    assert not validation_gate(metrics, 1, True)
    assert not validation_gate({**metrics, "recall": 0.1}, 0.5, True)
    assert not validation_gate({**metrics, "precision": 0.1}, 0.5, True)
    results = [
        {"name": "A", "average_precision": 0.9, "test_ap": 0, "passed": True},
        {"name": "B", "average_precision": 0.8, "test_ap": 1, "passed": True},
        {"name": "C", "average_precision": 1, "passed": False},
    ]
    assert [item["name"] for item in select_candidates(results)] == ["A", "B"]
    with pytest.raises(RuntimeError, match="no alias changed"):
        select_candidates(results[:1])


@pytest.fixture
def registry(tmp_path):
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/tracking.db")
    mlflow.set_experiment("real-roundtrip")
    raw, frame = load_data()
    # Both classes and all Type categories; keep the integration test small.
    sample = pd.concat([frame[frame.failure == 0].head(90), frame[frame.failure == 1].head(30)])
    x, y = sample[FEATURES], sample.failure
    estimators = [
        ("sklearn", build_pipeline(LogisticRegression(max_iter=500))),
        (
            "catboost",
            CatBoostClassifier(
                iterations=15,
                depth=3,
                cat_features=["type"],
                verbose=False,
                thread_count=1,
                allow_writing_files=False,
            ),
        ),
    ]
    registered = []
    for flavor, estimator in estimators:
        estimator.fit(x, y)
        bundle = tmp_path / flavor
        # Low threshold proves inference uses the serialized threshold, not 0.5.
        save_bundle(estimator, flavor, 0.1, bundle)
        expected = FailurePredictor(estimator, 0.1).predict(x)
        with mlflow.start_run():
            info = mlflow.pyfunc.log_model(
                name="model",
                loader_module="predictiveguard.inference",
                data_path=str(bundle),
                signature=infer_signature(x, expected),
                input_example=x.head(2),
                pip_requirements=[],
            )
            loaded = mlflow.pyfunc.load_model(info.model_uri)
            pd.testing.assert_frame_equal(loaded.predict(x), expected)
            registered.append(mlflow.register_model(info.model_uri, "roundtrip"))
    client = MlflowClient()
    client.set_registered_model_alias("roundtrip", "champion", registered[0].version)
    yield client, registered, x
    mlflow.set_tracking_uri(previous)


def test_real_registry_alias_switch_and_lifespan(registry):
    client, versions, x = registry
    config = Settings(mlflow_tracking_uri=mlflow.get_tracking_uri(), model_name="roundtrip")
    service = ModelService()
    service.load(config)
    request = ProcessRequest(**x.iloc[0].to_dict())
    first = service.predict(request)
    assert first.model.version == str(versions[0].version)
    client.set_registered_model_alias("roundtrip", "champion", versions[1].version)
    assert service.predict(request) == first
    service.load(config)
    assert service.predict(request).model.version == str(versions[1].version)
    # TestClient enters the actual FastAPI lifespan, unlike ASGITransport-only mocks.
    previous = model_service._loaded
    try:
        with patch("predictiveguard.main.settings", config):
            with TestClient(app) as http:
                for _ in range(2):
                    response = http.post("/process", json=request.model_dump())
                    assert response.status_code == 200
                    result = response.json()
                    assert result["model"]["version"] == str(versions[1].version)
                    expected = service.predict(request)
                    assert result["failure_probability"] == expected.failure_probability
                    assert result["prediction"] == expected.prediction
                assert (
                    http.post("/process", json={**request.model_dump(), "type": "X"}).status_code
                    == 422
                )
                assert (
                    http.post("/process", json={**request.model_dump(), "TWF": 1}).status_code
                    == 422
                )
    finally:
        model_service._loaded = previous


def test_alias_resolved_only_once_even_if_it_changes_during_load():
    service = ModelService()
    config = Settings(model_name="racing")
    with (
        patch("predictiveguard.model_service.MlflowClient") as client,
        patch("predictiveguard.model_service.load_model") as load,
    ):
        client.return_value.get_model_version_by_alias.return_value = SimpleNamespace(
            version="1", run_id="first"
        )
        service.load(config)
        client.return_value.get_model_version_by_alias.assert_called_once()
        load.assert_called_once_with("models:/racing/1")
        assert service._loaded.metadata.run_id == "first"
