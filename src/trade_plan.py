"""Trade-plan objects and sizing helpers."""

from __future__ import annotations

from dataclasses import dataclass

from src import config


@dataclass(frozen=True)
class TradeSetup:
    """A raw signal before risk sizing."""

    direction: str
    style: str
    entry: float
    invalidation: float
    confidence: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SizingProfile:
    """Risk and exit profile for one accepted setup."""

    risk_fraction: float
    max_margin_ratio: float
    take_profit_r: float
    partial_take_profit_r: float
    partial_exit_fraction: float
    trail_atr_mult: float
    hold_for_trend: bool
    leverage: int


@dataclass(frozen=True)
class TradePlan:
    """A complete trade plan ready for backtesting or execution."""

    direction: str
    style: str
    entry: float
    stop: float
    take_profit: float
    partial_take_profit: float
    trailing_distance: float
    qty: float
    margin: float
    risk_usdt: float
    risk_fraction: float
    reward_risk: float
    confidence: float
    partial_exit_fraction: float
    breakeven_arm_r: float
    hold_for_trend: bool
    leverage: int
    reasons: tuple[str, ...]


def _fallback_sizing() -> SizingProfile:
    return SizingProfile(
        risk_fraction=config.RISK_PER_TRADE,
        max_margin_ratio=config.MAX_POSITION_MARGIN_RATIO,
        take_profit_r=config.TAKE_PROFIT_R,
        partial_take_profit_r=config.PARTIAL_TAKE_PROFIT_R,
        partial_exit_fraction=config.PARTIAL_EXIT_FRACTION,
        trail_atr_mult=config.TRAIL_ATR_MULT,
        hold_for_trend=False,
        leverage=config.LEVERAGE,
    )


def round_price(value: float) -> float:
    """Keep enough precision for low-priced contracts without over-formatting BTC-like prices."""
    magnitude = abs(value)
    if magnitude >= 100:
        digits = 2
    elif magnitude >= 1:
        digits = 4
    elif magnitude >= 0.01:
        digits = 6
    else:
        digits = 8
    return round(value, digits)


def build_trade_plan(
    setup: TradeSetup,
    atr: float,
    equity: float,
    sizing: SizingProfile | None = None,
) -> TradePlan | None:
    """Convert a setup into a sized plan with R-based exits."""
    sizing = sizing or _fallback_sizing()
    if setup.entry <= 0 or atr <= 0 or equity <= 0:
        return None

    raw_stop = setup.invalidation
    if setup.direction == "long":
        atr_stop = setup.entry - config.STOP_ATR_MULT * atr
        stop = max(raw_stop, atr_stop)
        if stop >= setup.entry:
            stop = atr_stop
        risk_per_unit = setup.entry - stop
        take_profit = setup.entry + sizing.take_profit_r * risk_per_unit
        partial_take_profit = setup.entry + sizing.partial_take_profit_r * risk_per_unit
    else:
        atr_stop = setup.entry + config.STOP_ATR_MULT * atr
        stop = min(raw_stop, atr_stop)
        if stop <= setup.entry:
            stop = atr_stop
        risk_per_unit = stop - setup.entry
        take_profit = setup.entry - sizing.take_profit_r * risk_per_unit
        partial_take_profit = setup.entry - sizing.partial_take_profit_r * risk_per_unit

    if risk_per_unit <= 0:
        return None

    risk_budget = equity * sizing.risk_fraction
    qty_by_risk = risk_budget / risk_per_unit
    max_notional = equity * sizing.max_margin_ratio * sizing.leverage
    qty_by_margin = max_notional / setup.entry
    qty = max(0.0, min(qty_by_risk, qty_by_margin))
    if qty <= 0:
        return None

    notional = qty * setup.entry
    margin = notional / sizing.leverage
    risk_usdt = qty * risk_per_unit
    reward_risk = abs(take_profit - setup.entry) / risk_per_unit
    return TradePlan(
        direction=setup.direction,
        style=setup.style,
        entry=round_price(setup.entry),
        stop=round_price(stop),
        take_profit=round_price(take_profit),
        partial_take_profit=round_price(partial_take_profit),
        trailing_distance=round_price(sizing.trail_atr_mult * atr),
        qty=round(qty, 6),
        margin=round(margin, 4),
        risk_usdt=round(risk_usdt, 4),
        risk_fraction=sizing.risk_fraction,
        reward_risk=round(reward_risk, 3),
        confidence=setup.confidence,
        partial_exit_fraction=sizing.partial_exit_fraction,
        breakeven_arm_r=config.BREAKEVEN_ARM_R,
        hold_for_trend=sizing.hold_for_trend,
        leverage=sizing.leverage,
        reasons=setup.reasons,
    )
