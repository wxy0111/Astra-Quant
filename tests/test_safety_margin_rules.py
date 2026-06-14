"""Focused checks for entry safety-margin scoring.

Run with:
    python tests/test_safety_margin_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.safety_margin import safety_margin_multiplier
from src.trade_plan import TradePlan


def _plan(reward_risk: float = 3.0, confidence: float = 0.82) -> TradePlan:
    return TradePlan(
        direction="long",
        style="trend_pullback",
        entry=100.0,
        stop=96.0,
        take_profit=112.0,
        partial_take_profit=106.0,
        trailing_distance=5.0,
        qty=1.0,
        margin=20.0,
        risk_usdt=4.0,
        risk_fraction=0.01,
        reward_risk=reward_risk,
        confidence=confidence,
        partial_exit_fraction=0.3,
        breakeven_arm_r=1.0,
        hold_for_trend=True,
        leverage=5,
        reasons=("ema_pullback",),
    )


def test_low_expected_edge_is_rejected_after_cost_buffer() -> None:
    decision = safety_margin_multiplier(
        _plan(reward_risk=2.0),
        meta_probability=0.45,
        regime_multiplier=1.0,
        higher_aligned=False,
        market_breadth=0.5,
        strength_rank=0.5,
    )
    assert decision.multiplier == 0.0
    assert decision.reason == "safety_margin_too_low"


def test_mid_quality_signal_gets_probe_multiplier() -> None:
    decision = safety_margin_multiplier(
        _plan(reward_risk=2.5),
        meta_probability=0.55,
        regime_multiplier=0.65,
        higher_aligned=False,
        market_breadth=0.5,
        strength_rank=0.5,
    )
    assert 0.0 < decision.multiplier <= 0.75
    assert decision.reason == "safety_margin_probe"


def test_high_confidence_aligned_signal_gets_boost_but_is_capped() -> None:
    decision = safety_margin_multiplier(
        _plan(reward_risk=3.0),
        meta_probability=0.78,
        regime_multiplier=1.15,
        higher_aligned=True,
        market_breadth=0.8,
        strength_rank=0.9,
    )
    assert decision.multiplier > 1.0
    assert decision.multiplier <= 1.35
    assert decision.reason == "safety_margin_high"


if __name__ == "__main__":
    test_low_expected_edge_is_rejected_after_cost_buffer()
    test_mid_quality_signal_gets_probe_multiplier()
    test_high_confidence_aligned_signal_gets_boost_but_is_capped()
    print("Safety margin rule tests passed")
