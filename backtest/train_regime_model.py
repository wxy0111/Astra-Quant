"""Train a LightGBM multiclass model for future market regimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.regime_features import regime_feature_names
from src.regime_model import LightGBMRegimeModel


def train(dataset_path: str | Path, model_out: str | Path, metrics_out: str | Path | None = None, threshold: float = 0.35) -> dict:
    """Train model with chronological validation split."""
    df = pd.read_csv(dataset_path).sort_values("ts").reset_index(drop=True)
    names = regime_feature_names()
    labels = ["trend_up", "trend_down", "chop", "high_vol", "low_opportunity"]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    df = df[df["label"].isin(label_to_id)].copy()
    if df.empty:
        raise ValueError("regime dataset is empty")

    split = max(1, int(len(df) * 0.8))
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    train_x = train_df[names].fillna(0.0).to_numpy(dtype=float)
    test_x = test_df[names].fillna(0.0).to_numpy(dtype=float)
    train_y = train_df["label"].map(label_to_id).to_numpy(dtype=int)
    test_y = test_df["label"].map(label_to_id).to_numpy(dtype=int)

    params = {
        "objective": "multiclass",
        "num_class": len(labels),
        "metric": "multi_logloss",
        "learning_rate": 0.04,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
        "feature_pre_filter": False,
    }
    train_set = lgb.Dataset(train_x, label=train_y, feature_name=names, free_raw_data=False)
    booster = lgb.train(params, train_set, num_boost_round=180)
    model = LightGBMRegimeModel(names, labels, booster.model_to_string(), threshold=threshold)
    model.save(model_out)

    probs = booster.predict(test_x)
    pred = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    confident = confidence >= threshold
    accuracy = float((pred == test_y).mean()) if len(test_y) else 0.0
    confident_accuracy = float((pred[confident] == test_y[confident]).mean()) if confident.any() else 0.0
    label_counts = {label: int((df["label"] == label).sum()) for label in labels}
    importance = booster.feature_importance(importance_type="gain")
    metrics = {
        "rows": int(len(df)),
        "validation_rows": int(len(test_y)),
        "accuracy": round(accuracy, 4),
        "confident_accuracy": round(confident_accuracy, 4),
        "coverage": round(float(confident.mean()), 4) if len(confident) else 0.0,
        "threshold": threshold,
        "labels": labels,
        "label_counts": label_counts,
        "feature_importance_gain": sorted(
            [{"feature": name, "gain": round(float(gain), 4)} for name, gain in zip(names, importance)],
            key=lambda item: item["gain"],
            reverse=True,
        )[:10],
        "model_out": str(model_out),
        "dataset": str(dataset_path),
    }
    if metrics_out:
        out = Path(metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train future-regime LightGBM model")
    parser.add_argument("--dataset", default="backtest/datasets/regime_2y_until_20260415.csv")
    parser.add_argument("--model-out", default="backtest/models/regime_model_lgbm_2y_until_20260415.json")
    parser.add_argument("--metrics-out", default="backtest/reports/regime_model_lgbm_2y_until_20260415_metrics.json")
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()
    train(args.dataset, args.model_out, metrics_out=args.metrics_out, threshold=args.threshold)


if __name__ == "__main__":
    main()
