"""Market-regime detection for the V2 strategy."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config


@dataclass(frozen=True)
class MarketState:
    """A compact explanation of the current market regime."""

    regime: str
    direction: str
    confidence: float
    risk_score: float
    reasons: tuple[str, ...]


def _slope_pct(series: pd.Series, bars: int = 6) -> float:
    if len(series) <= bars:
        return 0.0
    start = float(series.iloc[-bars - 1])
    end = float(series.iloc[-1])
    if start <= 0:
        return 0.0
    return (end - start) / start


def classify_market(window: pd.DataFrame) -> MarketState:
    """Classify the latest row of a candle window."""
    row = window.iloc[-1]
    reasons: list[str] = []

    atr_pct = float(row["atr_pct"])
    atr_rank = float(row["atr_rank"]) if "atr_rank" in row and pd.notna(row["atr_rank"]) else float(window["atr_pct"].rank(pct=True).iloc[-1])
    adx = float(row["adx"])
    fast = float(row["ema_fast"])
    slow = float(row["ema_slow"])
    trend = float(row["ema_trend"])
    close = float(row["close"])
    ema_slope = _slope_pct(window["ema_slow"], bars=6)

    bull_stack = close > fast > slow > trend
    bear_stack = close < fast < slow < trend
    plus_di = float(row["plus_di"])
    minus_di = float(row["minus_di"])

    risk_score = 0.0
    if atr_rank >= config.ATR_HIGH_VOL_PERCENTILE:
        risk_score += 0.25
        reasons.append("high_atr_rank")
    if float(row["tr"]) >= float(row["atr"]) * config.CRASH_ATR_MULT:
        risk_score += 0.35
        reasons.append("range_shock")
    if adx < config.ADX_TREND_THRESHOLD and atr_rank >= config.ATR_HIGH_VOL_PERCENTILE:
        risk_score += 0.25
        reasons.append("volatile_chop")

    if bull_stack and plus_di > minus_di and adx >= config.ADX_TREND_THRESHOLD and ema_slope > 0:
        confidence = min(0.95, 0.55 + adx / 100 + min(abs(ema_slope) * 20, 0.15))
        reasons.append("bull_ema_stack")
        return MarketState("trend", "long", confidence, min(risk_score, 1.0), tuple(reasons))

    if bear_stack and minus_di > plus_di and adx >= config.ADX_TREND_THRESHOLD and ema_slope < 0:
        confidence = min(0.95, 0.55 + adx / 100 + min(abs(ema_slope) * 20, 0.15))
        reasons.append("bear_ema_stack")
        return MarketState("trend", "short", confidence, min(risk_score, 1.0), tuple(reasons))

    if adx < config.ADX_TREND_THRESHOLD and atr_rank <= config.ATR_LOW_VOL_PERCENTILE:
        reasons.append("low_vol_range")
        return MarketState("range", "none", 0.60, min(risk_score, 1.0), tuple(reasons))

    if risk_score >= 0.45:
        return MarketState("high_vol_chop", "none", 0.50, min(risk_score, 1.0), tuple(reasons))

    reasons.append("transition")
    return MarketState("transition", "none", 0.50, min(risk_score, 1.0), tuple(reasons))
