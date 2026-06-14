"""Feature extraction for meta-labeling candidate signals."""

from __future__ import annotations

import math
from collections import OrderedDict

import pandas as pd

from src.trade_plan import TradePlan


def feature_names() -> list[str]:
    """Stable feature order used by the lightweight meta model."""
    return [
        "adx",
        "atr_pct",
        "di_spread",
        "ema_fast_gap",
        "ema_slow_gap",
        "ema_trend_gap",
        "boll_width_pct",
        "volume_log",
        "plan_confidence",
        "reward_risk",
        "risk_fraction",
        "leverage",
        "higher_aligned",
        "market_breadth",
        "strength_rank",
        "direction_long",
        "style_pullback",
        "style_breakout",
        "agent_trend_pullback",
        "agent_trend_breakout",
        "agent_range_reversion",
        "agent_volatility_breakout",
        "ret_4",
        "ret_16",
        "ret_48",
        "realized_vol_4",
        "realized_vol_16",
        "realized_vol_48",
        "atr_pct_rank_100",
        "volume_z_20",
        "adx_delta_16",
        "pullback_depth_atr",
        "distance_to_stop_atr",
    ]


def _safe(value: float) -> float:
    if value is None:
        return 0.0
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def row_from_context(
    row: pd.Series,
    plan: TradePlan,
    higher_aligned: bool,
    market_breadth: float = 0.0,
    strength_rank: float = 0.0,
    window: pd.DataFrame | None = None,
    agent: str | None = None,
) -> OrderedDict[str, float]:
    """Build one numeric feature row from the latest indicator row and plan."""
    close = max(_safe(row.get("close", 0.0)), 1e-12)
    features: OrderedDict[str, float] = OrderedDict()
    features["adx"] = _safe(row.get("adx", 0.0))
    features["atr_pct"] = _safe(row.get("atr_pct", 0.0))
    features["di_spread"] = _safe(row.get("plus_di", 0.0)) - _safe(row.get("minus_di", 0.0))
    features["ema_fast_gap"] = (_safe(row.get("ema_fast", close)) / close) - 1.0
    features["ema_slow_gap"] = (_safe(row.get("ema_slow", close)) / close) - 1.0
    features["ema_trend_gap"] = (_safe(row.get("ema_trend", close)) / close) - 1.0
    features["boll_width_pct"] = _safe(row.get("boll_width_pct", 0.0))
    features["volume_log"] = math.log1p(max(_safe(row.get("vol", 0.0)), 0.0))
    features["plan_confidence"] = _safe(plan.confidence)
    features["reward_risk"] = _safe(plan.reward_risk)
    features["risk_fraction"] = _safe(plan.risk_fraction)
    features["leverage"] = _safe(plan.leverage)
    features["higher_aligned"] = 1.0 if higher_aligned else 0.0
    features["market_breadth"] = _safe(market_breadth)
    features["strength_rank"] = _safe(strength_rank)
    features["direction_long"] = 1.0 if plan.direction == "long" else 0.0
    features["style_pullback"] = 1.0 if plan.style == "trend_pullback" else 0.0
    features["style_breakout"] = 1.0 if plan.style == "trend_breakout" else 0.0
    features["agent_trend_pullback"] = 1.0 if agent == "trend_pullback_agent" else 0.0
    features["agent_trend_breakout"] = 1.0 if agent == "trend_breakout_agent" else 0.0
    features["agent_range_reversion"] = 1.0 if agent == "range_reversion_agent" else 0.0
    features["agent_volatility_breakout"] = 1.0 if agent == "volatility_breakout_agent" else 0.0
    features.update(_window_features(row, plan, window))
    return features


def _return(window: pd.DataFrame, bars: int) -> float:
    if window is None or len(window) <= bars:
        return 0.0
    start = _safe(window["close"].iloc[-bars - 1])
    end = _safe(window["close"].iloc[-1])
    if start <= 0:
        return 0.0
    return end / start - 1.0


def _realized_vol(window: pd.DataFrame, bars: int) -> float:
    if window is None or len(window) <= bars:
        return 0.0
    returns = window["close"].pct_change().tail(bars).dropna()
    if returns.empty:
        return 0.0
    return _safe(returns.std())


def _rank_latest(series: pd.Series, bars: int) -> float:
    if len(series) < 2:
        return 0.0
    recent = series.tail(bars).astype(float)
    if recent.empty:
        return 0.0
    return float(recent.rank(pct=True).iloc[-1])


def _z_latest(series: pd.Series, bars: int) -> float:
    recent = series.tail(bars).astype(float)
    if len(recent) < 3:
        return 0.0
    std = float(recent.std())
    if std <= 1e-12:
        return 0.0
    return (float(recent.iloc[-1]) - float(recent.mean())) / std


def _window_features(row: pd.Series, plan: TradePlan, window: pd.DataFrame | None) -> OrderedDict[str, float]:
    features: OrderedDict[str, float] = OrderedDict()
    features["ret_4"] = _return(window, 4) if window is not None else 0.0
    features["ret_16"] = _return(window, 16) if window is not None else 0.0
    features["ret_48"] = _return(window, 48) if window is not None else 0.0
    features["realized_vol_4"] = _realized_vol(window, 4) if window is not None else 0.0
    features["realized_vol_16"] = _realized_vol(window, 16) if window is not None else 0.0
    features["realized_vol_48"] = _realized_vol(window, 48) if window is not None else 0.0
    features["atr_pct_rank_100"] = _rank_latest(window["atr_pct"], 100) if window is not None and "atr_pct" in window else 0.0
    features["volume_z_20"] = _z_latest(window["vol"], 20) if window is not None and "vol" in window else 0.0
    if window is not None and "adx" in window and len(window) > 16:
        features["adx_delta_16"] = _safe(window["adx"].iloc[-1]) - _safe(window["adx"].iloc[-17])
    else:
        features["adx_delta_16"] = 0.0
    atr = max(_safe(row.get("atr", 0.0)), 1e-12)
    close = _safe(row.get("close", plan.entry))
    ema_fast = _safe(row.get("ema_fast", close))
    features["pullback_depth_atr"] = abs(close - ema_fast) / atr
    features["distance_to_stop_atr"] = abs(plan.entry - plan.stop) / atr
    return features
