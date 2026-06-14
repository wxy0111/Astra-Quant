"""Signal generation for trend, pullback, and range regimes."""

from __future__ import annotations

import pandas as pd

from src import config
from src.market_state import MarketState
from src.trade_plan import TradeSetup


def generate_setup(window: pd.DataFrame, state: MarketState) -> TradeSetup | None:
    """Return a setup when the latest candle gives a tradable event."""
    row = window.iloc[-1]
    prev = window.iloc[-2]
    close = float(row["close"])
    atr = float(row["atr"])

    if state.regime == "trend" and state.direction == "long":
        breakout = close > float(row["donchian_high"]) + config.BREAKOUT_BUFFER_ATR * atr
        pullback = (
            float(row["low"]) <= float(row["ema_fast"]) + config.PULLBACK_ATR_BAND * atr
            and close > float(row["ema_fast"])
            and float(prev["close"]) <= float(prev["ema_fast"]) + config.PULLBACK_ATR_BAND * float(prev["atr"])
        )
        if breakout:
            return TradeSetup("long", "trend_breakout", close, float(row["donchian_low"]), state.confidence, state.reasons + ("breakout_high",))
        if pullback:
            return TradeSetup("long", "trend_pullback", close, min(float(row["low"]), float(row["ema_slow"])), state.confidence, state.reasons + ("ema_pullback",))

    if state.regime == "trend" and state.direction == "short":
        breakout = close < float(row["donchian_low"]) - config.BREAKOUT_BUFFER_ATR * atr
        pullback = (
            float(row["high"]) >= float(row["ema_fast"]) - config.PULLBACK_ATR_BAND * atr
            and close < float(row["ema_fast"])
            and float(prev["close"]) >= float(prev["ema_fast"]) - config.PULLBACK_ATR_BAND * float(prev["atr"])
        )
        if breakout:
            return TradeSetup("short", "trend_breakout", close, float(row["donchian_high"]), state.confidence, state.reasons + ("breakout_low",))
        if pullback:
            return TradeSetup("short", "trend_pullback", close, max(float(row["high"]), float(row["ema_slow"])), state.confidence, state.reasons + ("ema_pullback",))

    if state.regime == "range":
        if close < float(row["boll_lower"]):
            return TradeSetup("long", "range_reversion", close, close - config.STOP_ATR_MULT * atr, 0.55, state.reasons + ("below_boll",))
        if close > float(row["boll_upper"]):
            return TradeSetup("short", "range_reversion", close, close + config.STOP_ATR_MULT * atr, 0.55, state.reasons + ("above_boll",))

    return None
