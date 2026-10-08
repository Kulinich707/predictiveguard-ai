from dataclasses import dataclass

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.pyfunc import load_model

from predictiveguard.config import Settings
from predictiveguard.schemas import ModelMetadata, ProcessRequest, ProcessResponse


@dataclass
class LoadedModel:
    model: object
    metadata: ModelMetadata


class ModelService:
    def __init__(self) -> None:
        self._loaded: LoadedModel | None = None

    @property
    def is_ready(self) -> bool:
        return self._loaded is not None

    def load(self, config: Settings) -> None:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        version = MlflowClient().get_model_version_by_alias(config.model_name, config.model_alias)
        self._loaded = LoadedModel(
            model=load_model(f"models:/{config.model_name}/{version.version}"),
            metadata=ModelMetadata(
                name=config.model_name,
                alias=config.model_alias,
                version=str(version.version),
                run_id=version.run_id,
            ),
        )

    def predict(self, request: ProcessRequest) -> ProcessResponse:
        if self._loaded is None:
            raise RuntimeError("Model is not loaded")

        frame = pd.DataFrame([request.model_dump()])
        result = self._loaded.model.predict(frame).iloc[0]
        probability = float(result["failure_probability"])
        return ProcessResponse(
            prediction="failure" if int(result["prediction"]) == 1 else "no_failure",
            failure_probability=round(probability, 6),
            model=self._loaded.metadata,
        )


model_service = ModelService()
