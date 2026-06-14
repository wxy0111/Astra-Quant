"""Focused checks for meta-label dataset and model plumbing.

Run with:
    python tests/test_meta_model_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.meta_features import feature_names, row_from_context
from src.meta_label import label_plan_outcome
from src.meta_model import LightGBMMetaModel, LogisticMetaModel, load_meta_model
from src.trade_plan import TradePlan


def _plan() -> TradePlan:
    return TradePlan(
        direction="long",
        style="trend_pullback",
        entry=100.0,
        stop=96.0,
        take_profit=112.0,
        partial_take_profit=106.0,
        trailing_distance=5.0,
        qty=1.0,
        margin=20.0,
        risk_usdt=4.0,
        risk_fraction=0.01,
        reward_risk=3.0,
        confidence=0.82,
        partial_exit_fraction=0.3,
        breakeven_arm_r=1.0,
        hold_for_trend=True,
        leverage=5,
        reasons=("ema_pullback",),
    )


def test_triple_barrier_like_label_marks_first_take_profit() -> None:
    future = pd.DataFrame(
        [
            {"high": 103.0, "low": 99.0, "close": 101.0},
            {"high": 113.0, "low": 100.0, "close": 112.0},
        ]
    )
    label = label_plan_outcome(_plan(), future)
    assert label.label == 1
    assert label.reason == "take_profit_first"


def test_feature_row_has_stable_numeric_columns() -> None:
    row = pd.Series(
        {
            "adx": 28.0,
            "atr_pct": 0.02,
            "plus_di": 30.0,
            "minus_di": 10.0,
            "ema_fast": 105.0,
            "ema_slow": 101.0,
            "ema_trend": 99.0,
            "close": 106.0,
            "boll_width_pct": 0.05,
            "vol": 123.0,
        }
    )
    features = row_from_context(row, _plan(), higher_aligned=True, market_breadth=0.6, strength_rank=0.8)
    assert list(features) == feature_names()
    assert features["higher_aligned"] == 1.0
    assert features["direction_long"] == 1.0


def test_feature_row_marks_agent_identity() -> None:
    row = pd.Series(
        {
            "adx": 28.0,
            "atr_pct": 0.02,
            "plus_di": 30.0,
            "minus_di": 10.0,
            "ema_fast": 105.0,
            "ema_slow": 101.0,
            "ema_trend": 99.0,
            "close": 106.0,
            "boll_width_pct": 0.05,
            "vol": 123.0,
        }
    )
    features = row_from_context(row, _plan(), higher_aligned=True, agent="trend_breakout_agent")
    assert features["agent_trend_pullback"] == 0.0
    assert features["agent_trend_breakout"] == 1.0
    assert features["agent_range_reversion"] == 0.0
    assert features["agent_volatility_breakout"] == 0.0


def test_feature_row_adds_window_momentum_and_volume_features() -> None:
    window = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 106.0],
            "vol": [10.0, 11.0, 12.0, 14.0, 18.0, 30.0],
            "atr_pct": [0.01, 0.011, 0.012, 0.013, 0.014, 0.015],
        }
    )
    row = pd.Series(
        {
            "adx": 28.0,
            "atr_pct": 0.015,
            "plus_di": 30.0,
            "minus_di": 10.0,
            "ema_fast": 105.0,
            "ema_slow": 101.0,
            "ema_trend": 99.0,
            "close": 106.0,
            "boll_width_pct": 0.05,
            "vol": 30.0,
        }
    )
    features = row_from_context(row, _plan(), higher_aligned=True, window=window)
    assert "ret_4" in features
    assert "realized_vol_4" in features
    assert "volume_z_20" in features
    assert features["ret_4"] > 0
    assert features["volume_z_20"] > 0


def test_logistic_meta_model_predicts_probability() -> None:
    model = LogisticMetaModel(feature_names=["x"], weights=[2.0], bias=-1.0, threshold=0.5)
    prob = model.predict_proba({"x": 1.0})
    assert 0.5 < prob < 1.0
    assert model.allows({"x": 1.0})


def test_meta_model_probability_maps_to_risk_multiplier() -> None:
    high = LogisticMetaModel(feature_names=["x"], weights=[3.0], bias=0.0, threshold=0.5)
    low = LogisticMetaModel(feature_names=["x"], weights=[-3.0], bias=0.0, threshold=0.5)
    assert high.risk_multiplier({"x": 1.0}) >= 1.0
    assert low.risk_multiplier({"x": 1.0}) == 0.0


def test_label_records_mfe_mae_and_partial_barrier() -> None:
    future = pd.DataFrame(
        [
            {"high": 105.0, "low": 98.0, "close": 103.0},
            {"high": 107.0, "low": 99.0, "close": 104.0},
            {"high": 108.0, "low": 95.0, "close": 96.0},
        ]
    )
    label = label_plan_outcome(_plan(), future)
    assert label.label == 0
    assert label.reached_partial
    assert label.mfe_r > 1.0
    assert label.mae_r <= -1.0
    assert label.outcome_class == "partial_then_stop"
    assert label.quality_label == 1
    assert label.outcome_score > -1.0


def test_lightgbm_meta_model_round_trips(tmp_path: Path | None = None) -> None:
    train = lgb.Dataset(np.array([[0.0], [1.0], [2.0], [3.0]]), label=np.array([0, 0, 1, 1]), feature_name=["x"])
    booster = lgb.train({"objective": "binary", "verbosity": -1, "num_leaves": 3, "min_data_in_leaf": 1}, train, num_boost_round=3)
    model = LightGBMMetaModel(["x"], booster.model_to_string(), threshold=0.4)
    path = (tmp_path or Path("backtest/reports")) / "tmp_lightgbm_meta_model.json"
    model.save(path)
    loaded = load_meta_model(path)
    assert loaded is not None
    prob = loaded.predict_proba({"x": 3.0})
    assert 0.0 <= prob <= 1.0


if __name__ == "__main__":
    test_triple_barrier_like_label_marks_first_take_profit()
    test_feature_row_has_stable_numeric_columns()
    test_feature_row_marks_agent_identity()
    test_feature_row_adds_window_momentum_and_volume_features()
    test_logistic_meta_model_predicts_probability()
    test_meta_model_probability_maps_to_risk_multiplier()
    test_label_records_mfe_mae_and_partial_barrier()
    test_lightgbm_meta_model_round_trips()
    print("Meta model rule tests passed")
