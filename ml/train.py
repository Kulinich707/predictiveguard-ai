import hashlib
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.catboost
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictiveguard.inference import FEATURES, NUMERIC_FEATURES, FailurePredictor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT / "data/ai4i2020.csv"))
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "predictiveguard-ai4i")
MODEL_NAME = os.getenv("MODEL_NAME", "PredictiveGuardFailureModel")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
SOURCE_SHA256 = "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"
TARGET = "failure"
SEED = 42
RENAME = {
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": TARGET,
}


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Expected the supplied, unchanged ai4i2020.csv (SHA256 mismatch)")
    raw = pd.read_csv(path)
    frame = raw.rename(columns=RENAME).set_index("UDI")[FEATURES + [TARGET]].copy()
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(float)
    frame[TARGET] = frame[TARGET].astype(int)
    return raw, frame


def split_data(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    development, test = train_test_split(
        frame, test_size=0.2, stratify=frame[TARGET], random_state=SEED
    )
    train, validation = train_test_split(
        development, test_size=0.25, stratify=development[TARGET], random_state=SEED
    )
    return {"training": train, "validation": validation, "test": test}


def dataset(frame: pd.DataFrame, name: str, path: Path):
    return mlflow.data.from_pandas(
        frame, source=path.resolve().as_uri(), targets=TARGET, name=f"ai4i2020-{name}"
    )


def lineage(splits: dict[str, pd.DataFrame], path: Path) -> dict:
    return {
        "source": path.resolve().as_uri(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "random_state": SEED,
        "split_fractions": {"training": 0.6, "validation": 0.2, "test": 0.2},
        "row_identifiers": {key: frame.index.tolist() for key, frame in splits.items()},
        "features": FEATURES,
        "column_mapping": RENAME,
        "excluded": ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"],
        "preprocessing": "numeric float; sklearn train-only median imputation, scaling, "
        "one-hot Type; CatBoost native categorical Type, no scaling",
        "selection": "validation Average Precision; threshold maximizes validation F1; "
        "test evaluated once after freezing selection",
    }


def log_eda(splits: dict[str, pd.DataFrame], path: Path = DATA_PATH) -> str:
    frame = splits["training"]
    with mlflow.start_run(run_name="00-eda-training") as run:
        mlflow.set_tags({"stage": "eda", "source_sha256": SOURCE_SHA256})
        mlflow.log_input(dataset(frame, "training", path), context="eda")
        mlflow.log_metrics(
            {
                "rows": len(frame),
                "failure_rate": frame[TARGET].mean(),
                "missing_values": frame.isna().sum().sum(),
                "duplicate_feature_rows": frame.duplicated().sum(),
            }
        )
        mlflow.log_dict(lineage(splits, path), "lineage/split_manifest.json")
        mlflow.log_artifact(str(path), "source")
        mlflow.log_text(
            "AI4I — синтетические данные датчиков оборудования.\n"
            "Отказы редки, поэтому высокая accuracy сама по себе не означает полезную модель. "
            "Основная метрика — Average Precision; дополнительно оцениваются precision, "
            "recall и PR-кривая.\n"
            "UDI и Product ID исключены как идентификаторы, а колонки типов отказов — "
            "во избежание утечки целевой метки. Type используется как категориальный признак.\n"
            "Масштабирование применяется для логистической регрессии. "
            "EDA выполняется на train; test используется только для итоговой оценки.\n"
            "Случайное стратифицированное разбиение не проверяет перенос модели "
            "на другие периоды или оборудование.",
            "eda/conclusions.txt",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            frame.describe(include="all").T.to_csv(output / "summary_statistics.csv")
            frame.isna().sum().to_csv(output / "missing_values.csv")
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            sns.countplot(data=frame, x=TARGET, ax=axes[0])
            sns.barplot(data=frame, x="type", y=TARGET, errorbar=None, ax=axes[1])
            axes[0].set_title("Rare failures (training only)")
            axes[1].set_title("Failure rate by Type")
            fig.tight_layout()
            fig.savefig(output / "target_and_type.png", dpi=100)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(frame.corr(numeric_only=True), annot=True, fmt=".2f", ax=ax)
            fig.tight_layout()
            fig.savefig(output / "correlations.png", dpi=100)
            plt.close(fig)
            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            for column, ax in zip(NUMERIC_FEATURES, axes.flat, strict=False):
                sns.histplot(data=frame, x=column, hue=TARGET, bins=30, ax=ax)
            axes.flat[-1].axis("off")
            fig.tight_layout()
            fig.savefig(output / "distributions.png", dpi=100)
            plt.close(fig)
            mlflow.log_artifacts(str(output), "eda")
        return run.info.run_id


def build_pipeline(estimator: object) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUMERIC_FEATURES,
            ),
            ("type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["type"]),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", estimator)])


def choose_threshold(y, probabilities) -> float:
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(scores)])


def evaluate(y, probabilities, threshold: float) -> dict[str, float]:
    predicted = probabilities >= threshold
    return {
        "average_precision": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y, predicted)),
    }


def validation_gate(metrics: dict, baseline_ap: float, reload_ok: bool) -> bool:
    return bool(
        metrics["average_precision"] > baseline_ap
        and metrics["recall"] >= 0.5
        and metrics["precision"] >= 0.2
        and reload_ok
    )


def log_diagnostics(y, probabilities, threshold: float, directory: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ConfusionMatrixDisplay.from_predictions(y, probabilities >= threshold, ax=axes[0])
    precision, recall, _ = precision_recall_curve(y, probabilities)
    axes[1].plot(recall, precision)
    axes[1].axhline(float(y.mean()), linestyle="--", label="prior")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR curve")
    axes[1].legend()
    fpr, tpr, _ = roc_curve(y, probabilities)
    axes[2].plot(fpr, tpr)
    axes[2].plot([0, 1], [0, 1], "--")
    axes[2].set(xlabel="FPR", ylabel="TPR", title="ROC curve")
    fig.tight_layout()
    fig.savefig(directory / "diagnostics.png", dpi=100)
    plt.close(fig)


def save_bundle(estimator, flavor: str, threshold: float, destination: Path) -> None:
    destination.mkdir()
    if flavor == "catboost":
        mlflow.catboost.save_model(estimator, str(destination / "estimator"))
    else:
        mlflow.sklearn.save_model(
            estimator,
            str(destination / "estimator"),
            serialization_format="skops",
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )
    (destination / "decision.json").write_text(
        json.dumps(
            {
                "flavor": flavor,
                "threshold": threshold,
                "features": FEATURES,
            }
        )
    )


def train_candidates(splits: dict[str, pd.DataFrame], path: Path = DATA_PATH) -> list[dict]:
    train, val = splits["training"], splits["validation"]
    candidates = {
        "01-dummy-prior": build_pipeline(DummyClassifier(strategy="prior", random_state=SEED)),
        "02-logistic-balanced": build_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=1000, C=1, random_state=SEED)
        ),
        "03-random-forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=250,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=2,
                random_state=SEED,
            )
        ),
        "04-catboost": CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            cat_features=["type"],
            random_seed=SEED,
            thread_count=2,
            verbose=False,
            allow_writing_files=False,
        ),
    }
    results = []
    baseline_ap = float(val[TARGET].mean())
    for name, estimator in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            flavor = "catboost" if isinstance(estimator, CatBoostClassifier) else "sklearn"
            estimator.fit(train[FEATURES], train[TARGET])
            probabilities = estimator.predict_proba(val[FEATURES])[:, 1]
            threshold = 0.5 if "dummy" in name else choose_threshold(val[TARGET], probabilities)
            metrics = evaluate(val[TARGET], probabilities, threshold)
            for context in ("training", "validation"):
                mlflow.log_input(dataset(splits[context], context, path), context=context)
            mlflow.log_dict(lineage(splits, path), "lineage/split_manifest.json")
            mlflow.log_params(
                {
                    "algorithm": name,
                    "seed": SEED,
                    "threshold": threshold,
                    "train_rows": len(train),
                    "validation_rows": len(val),
                    "threshold_selection": "validation_f1",
                    "flavor": flavor,
                    **(
                        estimator.get_params()
                        if flavor == "catboost"
                        else estimator.named_steps["classifier"].get_params()
                    ),
                }
            )
            mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                save_bundle(estimator, flavor, threshold, output / "bundle")
                predictor = FailurePredictor(estimator, threshold)
                expected = predictor.predict(val[FEATURES])
                info = mlflow.pyfunc.log_model(
                    name="model",
                    loader_module="predictiveguard.inference",
                    data_path=str(output / "bundle"),
                    code_paths=[str(ROOT / "src/predictiveguard")],
                    signature=infer_signature(val[FEATURES], expected),
                    input_example=val[FEATURES].head(3),
                    pip_requirements=[
                        f"{package}=={version(package)}"
                        for package in (
                            "mlflow",
                            "pandas",
                            "numpy",
                            "scikit-learn",
                            "catboost",
                            "skops",
                        )
                    ],
                )
                reloaded = mlflow.pyfunc.load_model(info.model_uri)
                actual = reloaded.predict(val[FEATURES])
                reload_ok = bool(
                    np.allclose(actual["failure_probability"], expected["failure_probability"])
                    and np.array_equal(actual["prediction"], expected["prediction"])
                )
                passed = validation_gate(metrics, baseline_ap, reload_ok)
                mlflow.log_dict(
                    {
                        "ap_above_prior": metrics["average_precision"] > baseline_ap,
                        "min_recall": 0.5,
                        "min_precision": 0.2,
                        "reload_equal": reload_ok,
                        "passed": passed,
                    },
                    "validation/gate.json",
                )
                log_diagnostics(val[TARGET], probabilities, threshold, output)
                mlflow.log_artifact(str(output / "diagnostics.png"), "validation")
                mlflow.set_tags(
                    {
                        "stage": "model-development",
                        "primary_metric": "validation_average_precision",
                        "source_sha256": SOURCE_SHA256,
                        "validation_status": "passed" if passed else "failed",
                        "reload_equal": str(reload_ok),
                    }
                )
            results.append(
                {
                    "name": name,
                    "run_id": run.info.run_id,
                    "model_uri": info.model_uri,
                    "threshold": threshold,
                    "flavor": flavor,
                    "passed": passed,
                    **metrics,
                }
            )
    return results


def select_candidates(results: list[dict]) -> list[dict]:
    eligible = sorted(
        [item for item in results if item["passed"]],
        key=lambda item: item["average_precision"],
        reverse=True,
    )
    if len(eligible) < 2:
        raise RuntimeError("Need two models passing the validation gate; no alias changed")
    return eligible[:2]


def evaluate_final(best: dict, splits: dict[str, pd.DataFrame], path: Path = DATA_PATH) -> dict:
    test = splits["test"]
    predictor = mlflow.pyfunc.load_model(best["model_uri"])
    probabilities = predictor.predict(test[FEATURES])["failure_probability"].to_numpy()
    metrics = evaluate(test[TARGET], probabilities, best["threshold"])
    with mlflow.start_run(run_name="05-frozen-champion-test"):
        mlflow.set_tags({"stage": "final-test", "selected_run_id": best["run_id"]})
        mlflow.log_input(dataset(test, "test", path), context="test")
        mlflow.log_params(
            {"frozen_model": best["model_uri"], "frozen_threshold": best["threshold"]}
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        mlflow.log_dict(lineage(splits, path), "lineage/split_manifest.json")
        with tempfile.TemporaryDirectory() as directory:
            log_diagnostics(test[TARGET], probabilities, best["threshold"], Path(directory))
            mlflow.log_artifacts(directory, "test")
        mlflow.log_dict({"selected": best, "test_metrics": metrics}, "selection.json")
    return metrics


def register_candidates(selected: list[dict], model_name=MODEL_NAME, alias=MODEL_ALIAS) -> list:
    client = MlflowClient()
    versions = []
    # Runner-up first, champion last. Alias changes only after both versions exist.
    for candidate in reversed(selected):
        registered = mlflow.register_model(candidate["model_uri"], model_name)
        for key, value in {
            "validation_status": "passed",
            "primary_metric": "validation_average_precision",
            "validation_average_precision": candidate["average_precision"],
            "candidate": candidate["name"],
            "threshold": candidate["threshold"],
            "flavor": candidate["flavor"],
            "source_sha256": SOURCE_SHA256,
        }.items():
            client.set_model_version_tag(model_name, registered.version, key, str(value))
        versions.append(registered)
    client.set_registered_model_alias(model_name, alias, versions[-1].version)
    return versions


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    _, frame = load_data()
    splits = split_data(frame)
    log_eda(splits)
    results = train_candidates(splits)
    selected = select_candidates(results)
    print(pd.DataFrame(results).drop(columns=["model_uri"]).to_string(index=False))
    print("Final untouched test:", evaluate_final(selected[0], splits))
    versions = register_candidates(selected)
    print(f"Champion: {selected[0]['name']}, version {versions[-1].version}")


if __name__ == "__main__":
    main()
