"""V4 trend-position add-on logic."""

from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.market_state import MarketState
from src.trade_plan import TradePlan


@dataclass(frozen=True)
class PyramidDecision:
    """Add-on verdict for an existing profitable position."""

    allowed: bool
    reason: str
    qty: float
    price: float


def _risk_per_unit(plan: TradePlan) -> float:
    if plan.direction == "long":
        return max(plan.entry - plan.stop, 0.0)
    return max(plan.stop - plan.entry, 0.0)


def _profit_r(plan: TradePlan, mark_price: float) -> float:
    risk = _risk_per_unit(plan)
    if risk <= 0:
        return 0.0
    if plan.direction == "long":
        return (mark_price - plan.entry) / risk
    return (plan.entry - mark_price) / risk


def evaluate_pyramid_add(
    plan: TradePlan,
    mark_price: float,
    equity: float,
    current_qty: float,
    add_count: int,
    higher_state: MarketState | None,
) -> PyramidDecision:
    """Allow a smaller add-on when a trend trade moves in favor."""
    if not config.PYRAMID_ENABLED:
        return PyramidDecision(False, "disabled", 0.0, mark_price)
    if not plan.hold_for_trend or plan.style != "trend_breakout":
        return PyramidDecision(False, "not_trend_hold", 0.0, mark_price)
    if add_count >= config.PYRAMID_MAX_ADDS:
        return PyramidDecision(False, "max_adds", 0.0, mark_price)
    if higher_state is None or higher_state.regime != "trend" or higher_state.direction != plan.direction:
        return PyramidDecision(False, "higher_timeframe_not_aligned", 0.0, mark_price)
    if higher_state.confidence < config.PYRAMID_MIN_HIGHER_CONFIDENCE:
        return PyramidDecision(False, "higher_confidence_low", 0.0, mark_price)

    required_r = config.PYRAMID_TRIGGER_R + add_count * config.PYRAMID_STEP_R
    if _profit_r(plan, mark_price) < required_r:
        return PyramidDecision(False, "profit_not_enough", 0.0, mark_price)

    max_notional = equity * config.PYRAMID_MAX_TOTAL_MARGIN_RATIO * plan.leverage
    current_notional = current_qty * mark_price
    remaining_notional = max(max_notional - current_notional, 0.0)
    target_add_qty = current_qty * config.PYRAMID_ADD_FRACTION
    max_add_qty = remaining_notional / mark_price if mark_price > 0 else 0.0
    qty = max(0.0, min(target_add_qty, max_add_qty))
    if qty <= 0:
        return PyramidDecision(False, "margin_cap", 0.0, mark_price)
    return PyramidDecision(True, "pyramid_add", round(qty, 6), mark_price)
