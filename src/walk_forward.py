"""Walk-forward universe selection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config
from src.symbol_score import SymbolScore


@dataclass(frozen=True)
class MarketGate:
    """Whether the portfolio-level regime allows new trades."""

    open: bool
    breadth: float
    average_momentum_pct: float
    reason: str


def split_train_test(df: pd.DataFrame, train_fraction: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a chronological frame into train and test sections."""
    fraction = train_fraction if train_fraction is not None else config.WALK_FORWARD_TRAIN_FRACTION
    if not 0 < fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    split_at = int(len(df) * fraction)
    if split_at <= 0 or split_at >= len(df):
        raise ValueError("not enough rows to split train/test")
    return df.iloc[:split_at].reset_index(drop=True), df.iloc[split_at:].reset_index(drop=True)


def momentum_return_pct(df: pd.DataFrame) -> float:
    """Return simple close-to-close momentum for a training window."""
    if df.empty:
        return 0.0
    first = float(df["close"].iloc[0])
    last = float(df["close"].iloc[-1])
    if first <= 0:
        return 0.0
    return (last / first - 1.0) * 100.0


def market_gate(items: list[dict]) -> MarketGate:
    """Open only when enough symbols have positive training-window momentum."""
    if not items:
        return MarketGate(False, 0.0, 0.0, "no_symbols")
    tradable = [item for item in items if item.get("momentum_return_pct", 0.0) > 0]
    breadth = len(tradable) / len(items)
    average_momentum = sum(float(item.get("momentum_return_pct", 0.0)) for item in items) / len(items)
    if breadth < config.MIN_MARKET_BREADTH:
        return MarketGate(False, round(breadth, 4), round(average_momentum, 4), "weak_market_breadth")
    if average_momentum < config.MIN_MARKET_AVG_MOMENTUM_PCT:
        return MarketGate(False, round(breadth, 4), round(average_momentum, 4), "weak_market_momentum")
    return MarketGate(True, round(breadth, 4), round(average_momentum, 4), "market_open")


def _recent_momentum_at(df: pd.DataFrame, ts: object, lookback_bars: int) -> float | None:
    if df.empty or "ts" not in df or "close" not in df:
        return None
    recent = df.loc[df["ts"] <= ts].tail(lookback_bars + 1)
    if len(recent) < max(3, min(lookback_bars + 1, 8)):
        return None
    first = float(recent["close"].iloc[0])
    last = float(recent["close"].iloc[-1])
    if first <= 0:
        return None
    return (last / first - 1.0) * 100.0


def rolling_market_gates_by_timestamp(
    frames: dict[str, pd.DataFrame],
    lookback_bars: int | None = None,
    min_breadth: float | None = None,
    min_average_momentum_pct: float | None = None,
    min_average_abs_momentum_pct: float | None = None,
) -> dict[object, MarketGate]:
    """Build test-period portfolio gates from recent cross-asset momentum."""
    lookback = lookback_bars if lookback_bars is not None else config.ROLLING_MARKET_LOOKBACK_BARS
    breadth_threshold = min_breadth if min_breadth is not None else config.ROLLING_MIN_MARKET_BREADTH
    avg_threshold = (
        min_average_momentum_pct
        if min_average_momentum_pct is not None
        else config.ROLLING_MIN_MARKET_AVG_MOMENTUM_PCT
    )
    abs_threshold = (
        min_average_abs_momentum_pct
        if min_average_abs_momentum_pct is not None
        else config.ROLLING_MIN_MARKET_AVG_ABS_MOMENTUM_PCT
    )
    timestamps = sorted({ts for frame in frames.values() for ts in frame.get("ts", pd.Series(dtype=object)).tolist()})
    gates: dict[object, MarketGate] = {}
    for ts in timestamps:
        momentums = [
            momentum
            for frame in frames.values()
            if (momentum := _recent_momentum_at(frame, ts, lookback)) is not None
        ]
        if not momentums:
            gates[ts] = MarketGate(True, 1.0, 0.0, "rolling_insufficient_history")
            continue
        breadth = sum(1 for momentum in momentums if momentum > 0) / len(momentums)
        average_momentum = sum(momentums) / len(momentums)
        average_abs_momentum = sum(abs(momentum) for momentum in momentums) / len(momentums)
        if breadth < breadth_threshold:
            gates[ts] = MarketGate(False, round(breadth, 4), round(average_momentum, 4), "rolling_weak_market_breadth")
        elif average_momentum < avg_threshold:
            gates[ts] = MarketGate(False, round(breadth, 4), round(average_momentum, 4), "rolling_weak_market_momentum")
        elif average_abs_momentum < abs_threshold:
            gates[ts] = MarketGate(False, round(breadth, 4), round(average_momentum, 4), "rolling_market_chop")
        else:
            gates[ts] = MarketGate(True, round(breadth, 4), round(average_momentum, 4), "rolling_market_open")
    return gates


def rolling_gate_allows_direction(gate: MarketGate | None, direction: str) -> bool:
    """Return whether the rolling portfolio gate allows a setup direction."""
    if gate is None or gate.open:
        return True
    if gate.reason == "rolling_market_chop":
        return False
    if gate.reason in ("rolling_weak_market_breadth", "rolling_weak_market_momentum"):
        if direction == "short" and gate.average_momentum_pct <= 0:
            return True
        if direction == "long" and gate.average_momentum_pct >= 0:
            return True
    return False


def assign_strength_ranks(items: list[dict], min_rank: float | None = None) -> list[dict]:
    """Add relative-strength rank and walk-forward state to symbol records."""
    threshold = min_rank if min_rank is not None else config.MIN_RELATIVE_STRENGTH_RANK
    ordered = sorted(items, key=lambda item: float(item.get("momentum_return_pct", 0.0)), reverse=True)
    total = len(ordered)
    if total == 0:
        return []

    ranked = []
    for index, item in enumerate(ordered):
        out = dict(item)
        rank = (total - index) / total
        score = out.get("score", SymbolScore(0.0, "paused", 0.0, "missing_score"))
        momentum = float(out.get("momentum_return_pct", 0.0))
        train_result = out.get("train_result", {})
        train_return_pct = float(train_result.get("return_pct", 0.0)) if isinstance(train_result, dict) else 0.0
        out["strength_rank"] = round(rank, 4)
        if momentum <= 0:
            out["walk_forward_state"] = "paused"
            out["walk_forward_weight"] = 0.0
            out["walk_forward_reason"] = "negative_momentum"
        elif rank < threshold:
            out["walk_forward_state"] = "paused"
            out["walk_forward_weight"] = 0.0
            out["walk_forward_reason"] = "relative_strength_too_low"
        elif score.weight <= 0:
            if (
                rank >= config.MIN_MOMENTUM_PROBE_RANK
                and momentum >= config.MIN_MOMENTUM_PROBE_RETURN_PCT
                and train_return_pct >= config.MIN_MOMENTUM_PROBE_TRAIN_RETURN_PCT
            ):
                out["walk_forward_state"] = "reduced"
                out["walk_forward_weight"] = config.MOMENTUM_PROBE_WEIGHT
                out["walk_forward_reason"] = "relative_strength_probe"
            else:
                out["walk_forward_state"] = "paused"
                out["walk_forward_weight"] = 0.0
                out["walk_forward_reason"] = "train_loss_too_high" if train_return_pct < config.MIN_MOMENTUM_PROBE_TRAIN_RETURN_PCT else score.reason
        else:
            if score.state == "reduced" and train_return_pct < 0:
                out["walk_forward_state"] = "paused"
                out["walk_forward_weight"] = 0.0
                out["walk_forward_reason"] = "reduced_score_negative_train_return"
            else:
                out["walk_forward_state"] = score.state
                out["walk_forward_weight"] = score.weight
                out["walk_forward_reason"] = "walk_forward_selected"
        ranked.append(out)
    return sorted(ranked, key=lambda item: item.get("inst_id", ""))
