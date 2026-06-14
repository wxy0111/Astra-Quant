"""Entry safety-margin scoring for model-sized trades."""

from __future__ import annotations

from dataclasses import dataclass

from src.trade_plan import TradePlan


@dataclass(frozen=True)
class SafetyMarginDecision:
    """Position-size multiplier derived from conservative entry edge."""

    multiplier: float
    score: float
    expected_edge_r: float
    probability: float
    reason: str


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safety_margin_multiplier(
    plan: TradePlan,
    meta_probability: float | None,
    regime_multiplier: float = 1.0,
    higher_aligned: bool = False,
    market_breadth: float = 0.5,
    strength_rank: float = 0.5,
    cost_buffer_r: float = 0.18,
    min_expected_edge_r: float = 0.15,
) -> SafetyMarginDecision:
    """Map a trade's conservative edge to a final sizing multiplier.

    The probability comes from the meta model when available; otherwise the
    setup confidence is used as a weaker fallback. Expected edge is measured in
    R and reduced by a fixed cost/slippage/model-error buffer.
    """
    probability = _clamp(float(meta_probability) if meta_probability is not None else float(plan.confidence), 0.0, 1.0)
    expected_edge_r = probability * plan.reward_risk - (1.0 - probability)
    edge_after_cost = expected_edge_r - cost_buffer_r
    if probability < 0.50 or edge_after_cost < min_expected_edge_r or regime_multiplier <= 0:
        return SafetyMarginDecision(
            multiplier=0.0,
            score=0.0,
            expected_edge_r=round(expected_edge_r, 4),
            probability=round(probability, 4),
            reason="safety_margin_too_low",
        )

    edge_score = _clamp(edge_after_cost / 2.0, 0.0, 1.0)
    regime_score = _clamp(regime_multiplier / 1.15, 0.0, 1.05)
    breadth_score = _clamp(market_breadth, 0.0, 1.0)
    strength_score = _clamp(strength_rank, 0.0, 1.0)
    score = (
        0.45 * probability
        + 0.35 * edge_score
        + 0.12 * regime_score
        + (0.08 if higher_aligned else 0.0)
        + 0.15 * (breadth_score - 0.5)
        + 0.20 * (strength_score - 0.5)
    )
    score = _clamp(score, 0.0, 1.0)

    if score < 0.42:
        multiplier = 0.0
        reason = "safety_margin_too_low"
    elif score < 0.60:
        multiplier = 0.50
        reason = "safety_margin_probe"
    elif score < 0.72:
        multiplier = 0.85
        reason = "safety_margin_normal"
    elif score < 0.82:
        multiplier = 1.10
        reason = "safety_margin_strong"
    else:
        multiplier = 1.35
        reason = "safety_margin_high"

    return SafetyMarginDecision(
        multiplier=multiplier,
        score=round(score, 4),
        expected_edge_r=round(expected_edge_r, 4),
        probability=round(probability, 4),
        reason=reason,
    )
