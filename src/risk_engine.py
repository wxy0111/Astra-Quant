"""Risk overlay for generated trade plans."""

from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.market_state import MarketState
from src.trade_plan import TradePlan


@dataclass(frozen=True)
class RiskDecision:
    """The risk engine verdict for a plan."""

    allowed: bool
    reason: str
    plan: TradePlan | None


def evaluate_plan(plan: TradePlan | None, state: MarketState) -> RiskDecision:
    """Reject plans that do not meet confidence, regime, or R/R requirements."""
    if plan is None:
        return RiskDecision(False, "no_plan", None)
    if state.risk_score >= 0.65:
        return RiskDecision(False, "market_risk_too_high", None)
    if plan.confidence < config.MIN_CONFIDENCE:
        return RiskDecision(False, "confidence_too_low", None)
    if plan.reward_risk < config.MIN_REWARD_RISK:
        return RiskDecision(False, "reward_risk_too_low", None)
    if plan.margin <= 0 or plan.risk_usdt <= 0:
        return RiskDecision(False, "invalid_size", None)
    return RiskDecision(True, "allowed", plan)
