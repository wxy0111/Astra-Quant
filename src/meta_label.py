"""Outcome labels for meta-labeling candidate trade plans."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.trade_plan import TradePlan


@dataclass(frozen=True)
class MetaLabel:
    """Label and realized result for a candidate signal."""

    label: int
    reason: str
    bars_held: int
    exit_price: float
    pnl_r: float
    mfe_r: float = 0.0
    mae_r: float = 0.0
    reached_partial: bool = False
    outcome_class: str = "unknown"
    quality_label: int = 0
    outcome_score: float = 0.0


def _risk_per_unit(plan: TradePlan) -> float:
    if plan.direction == "long":
        return max(plan.entry - plan.stop, 0.0)
    return max(plan.stop - plan.entry, 0.0)


def _quality_label(label: int, reached_partial: bool, mfe_r: float, mae_r: float, pnl_r: float) -> int:
    if label == 1:
        return 1
    if pnl_r > 0:
        return 1
    if reached_partial and mfe_r >= 1.25 and mae_r > -1.35:
        return 1
    return 0


def _score(pnl_r: float, mfe_r: float, mae_r: float) -> float:
    return round(0.55 * pnl_r + 0.30 * mfe_r + 0.15 * mae_r, 4)


def label_plan_outcome(plan: TradePlan, future: pd.DataFrame, max_bars: int = 96) -> MetaLabel:
    """Label a plan by which barrier is hit first in the future window."""
    risk = _risk_per_unit(plan)
    if risk <= 0 or future.empty:
        return MetaLabel(0, "invalid_or_empty", 0, plan.entry, 0.0, 0.0, 0.0, False, "invalid", 0, 0.0)

    limited = future.head(max_bars).reset_index(drop=True)
    mfe_r = 0.0
    mae_r = 0.0
    reached_partial = False
    for idx, row in limited.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        if plan.direction == "long":
            mfe_r = max(mfe_r, (high - plan.entry) / risk)
            mae_r = min(mae_r, (low - plan.entry) / risk)
            reached_partial = reached_partial or high >= plan.partial_take_profit
            hit_stop = low <= plan.stop
            hit_tp = high >= plan.take_profit
            if hit_stop and hit_tp:
                quality = _quality_label(0, reached_partial, mfe_r, mae_r, -1.0)
                return MetaLabel(0, "ambiguous_stop_first", idx + 1, plan.stop, -1.0, mfe_r, mae_r, reached_partial, "ambiguous", quality, _score(-1.0, mfe_r, mae_r))
            if hit_stop:
                outcome = "partial_then_stop" if reached_partial else "direct_stop"
                quality = _quality_label(0, reached_partial, mfe_r, mae_r, -1.0)
                return MetaLabel(0, "stop_first", idx + 1, plan.stop, -1.0, mfe_r, mae_r, reached_partial, outcome, quality, _score(-1.0, mfe_r, mae_r))
            if hit_tp:
                pnl_r = (plan.take_profit - plan.entry) / risk
                return MetaLabel(1, "take_profit_first", idx + 1, plan.take_profit, pnl_r, mfe_r, mae_r, reached_partial, "take_profit", 1, _score(pnl_r, mfe_r, mae_r))
        else:
            mfe_r = max(mfe_r, (plan.entry - low) / risk)
            mae_r = min(mae_r, (plan.entry - high) / risk)
            reached_partial = reached_partial or low <= plan.partial_take_profit
            hit_stop = high >= plan.stop
            hit_tp = low <= plan.take_profit
            if hit_stop and hit_tp:
                quality = _quality_label(0, reached_partial, mfe_r, mae_r, -1.0)
                return MetaLabel(0, "ambiguous_stop_first", idx + 1, plan.stop, -1.0, mfe_r, mae_r, reached_partial, "ambiguous", quality, _score(-1.0, mfe_r, mae_r))
            if hit_stop:
                outcome = "partial_then_stop" if reached_partial else "direct_stop"
                quality = _quality_label(0, reached_partial, mfe_r, mae_r, -1.0)
                return MetaLabel(0, "stop_first", idx + 1, plan.stop, -1.0, mfe_r, mae_r, reached_partial, outcome, quality, _score(-1.0, mfe_r, mae_r))
            if hit_tp:
                pnl_r = (plan.entry - plan.take_profit) / risk
                return MetaLabel(1, "take_profit_first", idx + 1, plan.take_profit, pnl_r, mfe_r, mae_r, reached_partial, "take_profit", 1, _score(pnl_r, mfe_r, mae_r))

    last_close = float(limited["close"].iloc[-1])
    pnl_r = (last_close - plan.entry) / risk if plan.direction == "long" else (plan.entry - last_close) / risk
    outcome = "timeout_positive" if pnl_r > 0 else "timeout_negative"
    label = 1 if pnl_r > 0 else 0
    quality = _quality_label(label, reached_partial, mfe_r, mae_r, pnl_r)
    return MetaLabel(label, outcome, len(limited), last_close, pnl_r, mfe_r, mae_r, reached_partial, outcome, quality, _score(pnl_r, mfe_r, mae_r))
