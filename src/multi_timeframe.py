"""Higher-timeframe context for V3 signal filtering."""

from __future__ import annotations

import pandas as pd

from src import config
from src.data import add_indicators
from src.market_state import MarketState, classify_market


def build_higher_timeframe_df(base_df: pd.DataFrame, timeframe: str = config.HIGHER_TIMEFRAME) -> pd.DataFrame:
    """Resample a base OHLCV DataFrame and add indicators."""
    source = base_df.copy()
    source = source.set_index("ts")
    higher = (
        source[["open", "high", "low", "close", "vol"]]
        .resample(timeframe)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "vol": "sum"})
        .dropna()
        .reset_index()
    )
    if len(higher) < max(config.EMA_TREND, config.BOLL_PERIOD, config.DONCHIAN_PERIOD) + 10:
        return pd.DataFrame()
    return add_indicators(
        higher,
        config.EMA_FAST,
        config.EMA_SLOW,
        config.EMA_TREND,
        config.ATR_PERIOD,
        config.ADX_PERIOD,
        config.DONCHIAN_PERIOD,
        config.BOLL_PERIOD,
        config.BOLL_STD,
    )


def higher_state_at(higher_df: pd.DataFrame, ts) -> MarketState | None:
    """Return the latest higher-timeframe state available at `ts`."""
    if higher_df.empty:
        return None
    window = higher_df[higher_df["ts"] <= pd.Timestamp(ts)]
    if len(window) < 20:
        return None
    return classify_market(window)


def build_higher_state_cache(base_df: pd.DataFrame, higher_df: pd.DataFrame) -> list[MarketState | None]:
    """Precompute latest higher-timeframe state for each base candle timestamp."""
    if higher_df.empty or base_df.empty:
        return [None] * len(base_df)
    states: list[MarketState | None] = []
    higher_states: list[MarketState | None] = []
    for idx in range(len(higher_df)):
        if idx < 20:
            higher_states.append(None)
        else:
            higher_states.append(classify_market(higher_df.iloc[: idx + 1]))
    cursor = -1
    higher_ts = list(higher_df["ts"])
    for ts in base_df["ts"]:
        current_ts = pd.Timestamp(ts)
        while cursor + 1 < len(higher_ts) and pd.Timestamp(higher_ts[cursor + 1]) <= current_ts:
            cursor += 1
        states.append(higher_states[cursor] if cursor >= 0 else None)
    return states
