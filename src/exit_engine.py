"""V3 position exit logic."""

from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.trade_plan import TradePlan


@dataclass(frozen=True)
class ExitDecision:
    """Exit action for the current bar."""

    partial_price: float | None
    final_price: float | None
    stop_price: float
    reason: str


def _risk_per_unit(plan: TradePlan) -> float:
    if plan.direction == "long":
        return max(plan.entry - plan.stop, 0.0)
    return max(plan.stop - plan.entry, 0.0)


def evaluate_exit(
    plan: TradePlan,
    high: float,
    low: float,
    best_price: float,
    partial_taken: bool,
) -> ExitDecision:
    """Evaluate partial, stop, and trend-hold exits for one candle."""
    risk = _risk_per_unit(plan)
    if risk <= 0:
        return ExitDecision(None, plan.stop, plan.stop, "invalid_risk")

    partial_price = None
    if plan.direction == "long":
        armed = best_price >= plan.entry + plan.breakeven_arm_r * risk
        base_stop = max(plan.stop, plan.entry) if armed else plan.stop
        trailing_stop = best_price - plan.trailing_distance
        stop = max(base_stop, trailing_stop)
        hit_partial = high >= plan.partial_take_profit
        hit_stop = low <= stop
        hit_tp = (not plan.hold_for_trend) and high >= plan.take_profit
        if hit_partial and not partial_taken:
            partial_price = plan.partial_take_profit
        final_price = stop if hit_stop else plan.take_profit if hit_tp else None
    else:
        armed = best_price <= plan.entry - plan.breakeven_arm_r * risk
        base_stop = min(plan.stop, plan.entry) if armed else plan.stop
        trailing_stop = best_price + plan.trailing_distance
        stop = min(base_stop, trailing_stop)
        hit_partial = low <= plan.partial_take_profit
        hit_stop = high >= stop
        hit_tp = (not plan.hold_for_trend) and low <= plan.take_profit
        if hit_partial and not partial_taken:
            partial_price = plan.partial_take_profit
        final_price = stop if hit_stop else plan.take_profit if hit_tp else None

    reason = "final_exit" if final_price is not None else "partial_or_hold" if partial_price is not None else "hold"
    return ExitDecision(partial_price, final_price, round(stop, 2), reason)
