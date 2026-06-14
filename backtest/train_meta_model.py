"""Train a meta-label model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.meta_features import feature_names
from src.meta_model import LightGBMMetaModel, LogisticMetaModel


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    means = train_x.mean(axis=0)
    scales = train_x.std(axis=0)
    scales = np.where(scales < 1e-9, 1.0, scales)
    return (train_x - means) / scales, (test_x - means) / scales, means.tolist(), scales.tolist()


def _fit_logistic(x: np.ndarray, y: np.ndarray, epochs: int, lr: float, l2: float) -> tuple[np.ndarray, float]:
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    for _ in range(epochs):
        z = x @ weights + bias
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
        error = pred - y
        grad_w = (x.T @ error) / len(y) + l2 * weights
        grad_b = float(error.mean())
        weights -= lr * grad_w
        bias -= lr * grad_b
    return weights, bias


def _metrics(probs: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    pred = (probs >= threshold).astype(int)
    accuracy = float((pred == y).mean()) if len(y) else 0.0
    selected = pred == 1
    selected_win_rate = float(y[selected].mean()) if selected.any() else 0.0
    coverage = float(selected.mean()) if len(y) else 0.0
    baseline = float(y.mean()) if len(y) else 0.0
    return {
        "rows": int(len(y)),
        "baseline_win_rate": round(baseline, 4),
        "accuracy": round(accuracy, 4),
        "selected_win_rate": round(selected_win_rate, 4),
        "coverage": round(coverage, 4),
    }


def _train_logistic(train_x: np.ndarray, test_x: np.ndarray, train_y: np.ndarray, test_y: np.ndarray, names: list[str], threshold: float, model_out: str | Path) -> tuple[np.ndarray, dict]:
    train_x, test_x, means, scales = _standardize(train_x, test_x)
    weights, bias = _fit_logistic(train_x, train_y, epochs=1200, lr=0.05, l2=0.001)
    model = LogisticMetaModel(names, weights.tolist(), float(bias), threshold, means=means, scales=scales)
    model.save(model_out)
    probs = 1.0 / (1.0 + np.exp(-np.clip(test_x @ weights + bias, -35, 35)))
    return probs, {"model_type": "logistic"}


def _train_lightgbm(train_x: np.ndarray, test_x: np.ndarray, train_y: np.ndarray, names: list[str], threshold: float, model_out: str | Path) -> tuple[np.ndarray, dict]:
    positives = float(train_y.sum())
    negatives = float(len(train_y) - positives)
    scale_pos_weight = negatives / positives if positives > 0 else 1.0
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.035,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
        "seed": 42,
        "feature_pre_filter": False,
    }
    train_set = lgb.Dataset(train_x, label=train_y, feature_name=names, free_raw_data=False)
    booster = lgb.train(params, train_set, num_boost_round=220)
    model = LightGBMMetaModel(names, booster.model_to_string(), threshold)
    model.save(model_out)
    probs = booster.predict(test_x)
    importance = booster.feature_importance(importance_type="gain")
    ranked = sorted(
        ({"feature": name, "gain": round(float(gain), 4)} for name, gain in zip(names, importance)),
        key=lambda item: item["gain"],
        reverse=True,
    )
    return probs, {"model_type": "lightgbm", "scale_pos_weight": round(scale_pos_weight, 4), "feature_importance_gain": ranked[:10]}


def train(
    dataset_path: str | Path,
    model_out: str | Path,
    metrics_out: str | Path | None = None,
    threshold: float = 0.50,
    model_type: str = "lightgbm",
    target_column: str = "label",
) -> dict:
    """Train model using chronological 80/20 validation split."""
    df = pd.read_csv(dataset_path)
    names = feature_names()
    if df.empty:
        raise ValueError("dataset is empty")
    missing = [name for name in names + [target_column] if name not in df.columns]
    if missing:
        raise ValueError(f"dataset missing columns: {missing}")

    df = df.sort_values("ts").reset_index(drop=True)
    split = max(1, int(len(df) * 0.8))
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    train_x = train_df[names].fillna(0.0).to_numpy(dtype=float)
    test_x = test_df[names].fillna(0.0).to_numpy(dtype=float)
    train_y = train_df[target_column].to_numpy(dtype=float)
    test_y = test_df[target_column].to_numpy(dtype=int)

    if model_type == "logistic":
        probs, extra = _train_logistic(train_x, test_x, train_y, test_y, names, threshold, model_out)
    elif model_type == "lightgbm":
        probs, extra = _train_lightgbm(train_x, test_x, train_y, names, threshold, model_out)
    else:
        raise ValueError(f"unsupported model_type: {model_type}")

    metrics = _metrics(probs, test_y, threshold)
    metrics.update({"model_out": str(model_out), "dataset": str(dataset_path), "threshold": threshold, "target_column": target_column, **extra})
    if metrics_out:
        out = Path(metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train meta-label model")
    parser.add_argument("--dataset", default="backtest/datasets/meta_signals_2y.csv")
    parser.add_argument("--model-out", default="backtest/models/meta_model_lgbm_2y.json")
    parser.add_argument("--metrics-out", default="backtest/reports/meta_model_2y_metrics.json")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--model-type", choices=["lightgbm", "logistic"], default="lightgbm")
    parser.add_argument("--target-column", default="label", help="Target column, e.g. label or quality_label")
    args = parser.parse_args()
    train(args.dataset, args.model_out, metrics_out=args.metrics_out, threshold=args.threshold, model_type=args.model_type, target_column=args.target_column)


if __name__ == "__main__":
    main()
