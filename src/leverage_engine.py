"""Dynamic leverage selection for accepted V4 setups."""

from __future__ import annotations

from src import config
from src.market_state import MarketState
from src.trade_plan import TradeSetup


def clamp_leverage(value: int | float) -> int:
    """Clamp leverage to configured research bounds."""
    return int(max(config.MIN_LEVERAGE, min(config.MAX_LEVERAGE, round(value))))


def choose_leverage(
    setup: TradeSetup,
    state: MarketState,
    higher_state: MarketState | None,
    *,
    reason: str,
    shock: bool,
    failed_breakout: bool,
) -> int:
    """Select leverage from setup quality and market risk.

    This is research sizing only. Live OKX integration must still validate
    exchange limits and liquidation distance before placing orders.
    """
    if reason in ("failed_breakout_probe", "shock_breakout_probe") or failed_breakout:
        return clamp_leverage(config.PROBE_LEVERAGE)
    if setup.style == "range_reversion":
        return clamp_leverage(config.RANGE_LEVERAGE)

    aligned = (
        higher_state is not None
        and higher_state.regime == "trend"
        and higher_state.direction == setup.direction
    )
    if setup.style == "trend_breakout" and aligned and setup.confidence >= 0.80 and not shock:
        return clamp_leverage(config.HIGH_CONVICTION_LEVERAGE)
    if setup.style in ("trend_breakout", "trend_pullback") and state.regime == "trend":
        lev = config.NORMAL_TREND_LEVERAGE
        if state.risk_score >= 0.35 or shock:
            lev -= 2
        if aligned and setup.confidence >= 0.75:
            lev += 1
        return clamp_leverage(lev)
    return clamp_leverage(config.LEVERAGE)
