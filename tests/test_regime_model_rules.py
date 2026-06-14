"""Focused checks for market-regime model plumbing.

Run with:
    python tests/test_regime_model_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regime_features import regime_feature_names, regime_features_from_window
from src.regime_label import label_future_regime
from src.regime_model import LightGBMRegimeModel, load_regime_model


def _ohlcv(closes: list[float]) -> pd.DataFrame:
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "vol": np.linspace(100.0, 200.0, len(close)),
        }
    )


def test_regime_features_include_structure_factors() -> None:
    df = _ohlcv([100 + i * 0.2 for i in range(160)])
    features = regime_features_from_window(df, bars_per_day=4)
    assert list(features) == regime_feature_names()
    assert features["er_120d"] > 0
    assert features["adx_60d"] >= 0
    assert features["natr_7d"] > 0


def test_future_regime_labels_trending_and_choppy_windows() -> None:
    up = label_future_regime(_ohlcv([100 + i for i in range(80)]), bars_per_day=4)
    chop = label_future_regime(_ohlcv([100, 103, 97, 102, 98, 101, 99, 100] * 10), bars_per_day=4)
    assert up.label == "trend_up"
    assert chop.label in ("chop", "high_vol", "low_opportunity")


def test_regime_model_blocks_wrong_direction(tmp_path: Path | None = None) -> None:
    train = lgb.Dataset(
        np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]),
        label=np.array([0, 0, 1, 1, 2, 2]),
        feature_name=["x"],
    )
    booster = lgb.train(
        {"objective": "multiclass", "num_class": 3, "verbosity": -1, "num_leaves": 3, "min_data_in_leaf": 1},
        train,
        num_boost_round=3,
    )
    model = LightGBMRegimeModel(["x"], ["trend_up", "trend_down", "chop"], booster.model_to_string(), threshold=0.0)
    path = (tmp_path or Path("backtest/reports")) / "tmp_regime_model.json"
    model.save(path)
    loaded = load_regime_model(path)
    assert loaded is not None
    decision = loaded.decision({"x": 0.0}, direction="short")
    assert decision.label in ("trend_up", "trend_down", "chop")
    if decision.label == "trend_up" and decision.probability >= 0.65:
        assert not decision.allowed
        assert decision.reason == "regime_direction_mismatch"
    else:
        assert decision.reason in ("regime_not_blocked", "regime_direction_aligned", "regime_probability_low")


def test_regime_model_returns_risk_multiplier() -> None:
    model = LightGBMRegimeModel(["x"], ["trend_up", "trend_down", "chop"], "", threshold=0.0)
    aligned = model.multiplier_for_label("trend_up", 0.7, direction="long")
    chop = model.multiplier_for_label("chop", 0.7, direction="long")
    blocked = model.multiplier_for_label("high_vol", 0.8, direction="long")
    assert aligned > 1.0
    assert 0.0 < chop < 1.0
    assert blocked == 0.0


if __name__ == "__main__":
    test_regime_features_include_structure_factors()
    test_future_regime_labels_trending_and_choppy_windows()
    test_regime_model_blocks_wrong_direction()
    test_regime_model_returns_risk_multiplier()
    print("Regime model rule tests passed")
