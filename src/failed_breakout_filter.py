"""Detect breakout candles with high failure/reversal risk."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config
from src.trade_plan import TradeSetup


@dataclass(frozen=True)
class BreakoutFailureRisk:
    """Whether a breakout candle looks vulnerable to immediate reversal."""

    risky: bool
    reason: str
    wick_ratio: float


def breakout_failure_risk(setup: TradeSetup, window: pd.DataFrame) -> BreakoutFailureRisk:
    """Return risk when a breakout closes far away from the breakout extreme."""
    if setup.style != "trend_breakout" or window.empty:
        return BreakoutFailureRisk(False, "not_breakout", 0.0)

    row = window.iloc[-1]
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    open_ = float(row["open"])
    candle_range = max(high - low, 0.0)
    if candle_range <= 0:
        return BreakoutFailureRisk(False, "zero_range", 0.0)

    body_mid = (open_ + close) / 2
    if setup.direction == "short":
        lower_wick = min(open_, close) - low
        wick_ratio = lower_wick / candle_range
        close_reclaim = (close - low) / candle_range
        risky = (
            wick_ratio >= config.FAILED_BREAKOUT_WICK_RATIO
            or close_reclaim >= config.FAILED_BREAKOUT_BODY_RECLAIM_RATIO
            or body_mid > low + candle_range * 0.55
        )
        return BreakoutFailureRisk(risky, "short_reclaimed_low" if risky else "clean_short_breakout", wick_ratio)

    upper_wick = high - max(open_, close)
    wick_ratio = upper_wick / candle_range
    close_reclaim = (high - close) / candle_range
    risky = (
        wick_ratio >= config.FAILED_BREAKOUT_WICK_RATIO
        or close_reclaim >= config.FAILED_BREAKOUT_BODY_RECLAIM_RATIO
        or body_mid < high - candle_range * 0.55
    )
    return BreakoutFailureRisk(risky, "long_rejected_high" if risky else "clean_long_breakout", wick_ratio)
