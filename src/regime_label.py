"""Future market-shape labels for regime modeling."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.regime_features import regime_features_from_window


@dataclass(frozen=True)
class RegimeLabel:
    """Future shape label and supporting metrics."""

    label: str
    future_return: float
    efficiency: float
    realized_vol: float
    reason: str


def label_future_regime(future: pd.DataFrame, bars_per_day: int = 96) -> RegimeLabel:
    """Classify the future window into a coarse tradability regime."""
    if len(future) < 4:
        return RegimeLabel("low_opportunity", 0.0, 0.0, 0.0, "too_short")
    close = future["close"].astype(float)
    start = float(close.iloc[0])
    end = float(close.iloc[-1])
    future_return = end / start - 1.0 if start > 0 else 0.0
    returns = close.pct_change().dropna()
    realized_vol = float(returns.std()) if not returns.empty else 0.0
    features = regime_features_from_window(future, bars_per_day=bars_per_day)
    efficiency = features["er_120d"]
    chop = features["chop_30d"]
    natr = features["natr_7d"]

    if realized_vol > 0.026 or natr > 0.28:
        return RegimeLabel("high_vol", future_return, efficiency, realized_vol, "future_high_vol")
    if future_return >= 0.006 and efficiency >= 0.18:
        return RegimeLabel("trend_up", future_return, efficiency, realized_vol, "future_uptrend")
    if future_return <= -0.006 and efficiency >= 0.18:
        return RegimeLabel("trend_down", future_return, efficiency, realized_vol, "future_downtrend")
    if chop >= 55.0 or efficiency < 0.20:
        return RegimeLabel("chop", future_return, efficiency, realized_vol, "future_chop")
    return RegimeLabel("low_opportunity", future_return, efficiency, realized_vol, "future_low_opportunity")
