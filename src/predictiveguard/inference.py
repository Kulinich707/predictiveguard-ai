import json
from pathlib import Path

import mlflow.catboost
import mlflow.sklearn
import pandas as pd

NUMERIC_FEATURES = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
FEATURES = ["type", *NUMERIC_FEATURES]


class FailurePredictor:
    def __init__(self, estimator: object, threshold: float) -> None:
        self.estimator = estimator
        self.threshold = threshold

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.estimator.predict_proba(frame[FEATURES])[:, 1]
        return pd.DataFrame(
            {
                "failure_probability": probabilities,
                "prediction": (probabilities >= self.threshold).astype(int),
            },
            index=frame.index,
        )


def _load_pyfunc(path: str) -> FailurePredictor:
    bundle = Path(path)
    config = json.loads((bundle / "decision.json").read_text())
    loader = (
        mlflow.catboost.load_model if config["flavor"] == "catboost" else mlflow.sklearn.load_model
    )
    return FailurePredictor(loader(str(bundle / "estimator")), config["threshold"])
