"""Candle loading and indicator helpers."""

from __future__ import annotations

import pandas as pd


OKX_COLUMNS = [
    "ts",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "volCcy",
    "volCcyQuote",
    "confirm",
]


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """Load an OKX candle CSV and return a typed chronological DataFrame."""
    df = pd.read_csv(path)
    missing = {"ts", "open", "high", "low", "close"}.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    df = df.copy()
    ts_values = df["ts"].astype("int64")
    median_ts = int(ts_values.median())
    if median_ts < 100_000_000_000:
        unit = "s"
    elif median_ts < 100_000_000_000_000:
        unit = "ms"
    else:
        unit = "ns"
    df["ts"] = pd.to_datetime(ts_values, unit=unit)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    if "vol" in df.columns:
        df["vol"] = df["vol"].astype(float)
    else:
        df["vol"] = 0.0
    return df.sort_values("ts").reset_index(drop=True)


def ema(series: pd.Series, span: int) -> pd.Series:
    """Return an exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def add_indicators(
    df: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
    ema_trend: int,
    atr_period: int,
    adx_period: int,
    donchian_period: int,
    boll_period: int,
    boll_std: float,
) -> pd.DataFrame:
    """Add trend, volatility, channel, and Bollinger features."""
    out = df.copy()
    out["ema_fast"] = ema(out["close"], ema_fast)
    out["ema_slow"] = ema(out["close"], ema_slow)
    out["ema_trend"] = ema(out["close"], ema_trend)

    prev_close = out["close"].shift(1)
    tr_parts = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    out["tr"] = tr_parts.max(axis=1)
    out["atr"] = out["tr"].ewm(alpha=1 / atr_period, adjust=False).mean()
    out["atr_pct"] = out["atr"] / out["close"]
    out["atr_rank"] = out["atr_pct"].rolling(200, min_periods=20).rank(pct=True).fillna(0.5)

    up_move = out["high"].diff()
    down_move = -out["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di = 100 * plus_dm.ewm(alpha=1 / adx_period, adjust=False).mean() / out["atr"]
    minus_di = 100 * minus_dm.ewm(alpha=1 / adx_period, adjust=False).mean() / out["atr"]
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    out["plus_di"] = plus_di.fillna(0.0)
    out["minus_di"] = minus_di.fillna(0.0)
    out["adx"] = dx.ewm(alpha=1 / adx_period, adjust=False).mean().fillna(0.0)

    out["donchian_high"] = out["high"].rolling(donchian_period).max().shift(1)
    out["donchian_low"] = out["low"].rolling(donchian_period).min().shift(1)

    out["boll_mid"] = out["close"].rolling(boll_period).mean()
    boll_dev = out["close"].rolling(boll_period).std(ddof=0)
    out["boll_upper"] = out["boll_mid"] + boll_std * boll_dev
    out["boll_lower"] = out["boll_mid"] - boll_std * boll_dev
    out["boll_width_pct"] = (out["boll_upper"] - out["boll_lower"]) / out["close"]

    lookback = max(ema_trend, donchian_period, boll_period, atr_period, adx_period) + 5
    return out.iloc[lookback:].reset_index(drop=True)
