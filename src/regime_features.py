"""Market-structure features for future-regime prediction."""

from __future__ import annotations

import math
from collections import OrderedDict

import pandas as pd


def regime_feature_names() -> list[str]:
    """Stable feature order for the regime model."""
    return [
        "er_120d",
        "chop_30d",
        "ret_ac_60d",
        "vol_regime_30_90",
        "adx_60d",
        "natr_7d",
        "ret_7d",
        "ret_30d",
        "realized_vol_7d",
        "realized_vol_30d",
        "range_pos_30d",
        "volume_z_30d",
    ]


def _safe(value: float) -> float:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def _tail(window: pd.DataFrame, bars: int) -> pd.DataFrame:
    return window.tail(min(len(window), max(2, bars))).copy()


def _return(close: pd.Series, bars: int) -> float:
    if len(close) <= bars:
        bars = len(close) - 1
    if bars <= 0:
        return 0.0
    start = float(close.iloc[-bars - 1])
    end = float(close.iloc[-1])
    if start <= 0:
        return 0.0
    return end / start - 1.0


def _realized_vol(close: pd.Series, bars: int) -> float:
    recent = close.pct_change().tail(max(2, bars)).dropna()
    return _safe(recent.std()) if not recent.empty else 0.0


def _efficiency_ratio(close: pd.Series, bars: int) -> float:
    recent = close.tail(min(len(close), bars + 1))
    if len(recent) < 3:
        return 0.0
    net = abs(float(recent.iloc[-1]) - float(recent.iloc[0]))
    path = float(recent.diff().abs().sum())
    return net / path if path > 0 else 0.0


def _chop_index(df: pd.DataFrame, bars: int) -> float:
    recent = _tail(df, bars)
    high_low = float(recent["high"].max() - recent["low"].min())
    if high_low <= 0 or len(recent) < 3:
        return 0.0
    tr_sum = float((recent["high"] - recent["low"]).abs().sum())
    if tr_sum <= 0:
        return 0.0
    return 100.0 * math.log10(tr_sum / high_low) / math.log10(len(recent))


def _ret_autocorr(close: pd.Series, bars: int) -> float:
    returns = close.pct_change().tail(max(4, bars)).dropna()
    if len(returns) < 4:
        return 0.0
    value = returns.autocorr(lag=1)
    return _safe(value)


def _adx_like(df: pd.DataFrame, bars: int) -> float:
    recent = _tail(df, bars)
    up = recent["high"].diff()
    down = -recent["low"].diff()
    plus = up.where((up > down) & (up > 0), 0.0).sum()
    minus = down.where((down > up) & (down > 0), 0.0).sum()
    denom = plus + minus
    if denom <= 0:
        return 0.0
    return abs(float(plus - minus)) / float(denom) * 100.0


def _range_pos(df: pd.DataFrame, bars: int) -> float:
    recent = _tail(df, bars)
    low = float(recent["low"].min())
    high = float(recent["high"].max())
    close = float(recent["close"].iloc[-1])
    if high <= low:
        return 0.5
    return (close - low) / (high - low)


def _volume_z(df: pd.DataFrame, bars: int) -> float:
    if "vol" not in df:
        return 0.0
    recent = df["vol"].tail(min(len(df), bars)).astype(float)
    if len(recent) < 3:
        return 0.0
    std = float(recent.std())
    if std <= 1e-12:
        return 0.0
    return (float(recent.iloc[-1]) - float(recent.mean())) / std


def regime_features_from_window(window: pd.DataFrame, bars_per_day: int = 96) -> OrderedDict[str, float]:
    """Build market-structure factors from a chronological OHLCV window."""
    close = window["close"].astype(float)
    d7 = 7 * bars_per_day
    d30 = 30 * bars_per_day
    d60 = 60 * bars_per_day
    d90 = 90 * bars_per_day
    d120 = 120 * bars_per_day
    vol_30 = _realized_vol(close, d30)
    vol_90 = _realized_vol(close, d90)
    out: OrderedDict[str, float] = OrderedDict()
    out["er_120d"] = _safe(_efficiency_ratio(close, d120))
    out["chop_30d"] = _safe(_chop_index(window, d30))
    out["ret_ac_60d"] = _safe(_ret_autocorr(close, d60))
    out["vol_regime_30_90"] = _safe(vol_30 / vol_90) if vol_90 > 0 else 0.0
    out["adx_60d"] = _safe(_adx_like(window, d60))
    out["natr_7d"] = _safe((window["high"].tail(d7).max() - window["low"].tail(d7).min()) / max(close.iloc[-1], 1e-12))
    out["ret_7d"] = _safe(_return(close, d7))
    out["ret_30d"] = _safe(_return(close, d30))
    out["realized_vol_7d"] = _safe(_realized_vol(close, d7))
    out["realized_vol_30d"] = _safe(vol_30)
    out["range_pos_30d"] = _safe(_range_pos(window, d30))
    out["volume_z_30d"] = _safe(_volume_z(window, d30))
    return out
