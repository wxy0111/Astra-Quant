"""Focused checks for multi-asset universe behavior.

Run with:
    python tests/test_universe_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.universe_engine import instrument_profile, listed_universe, profile_risk_multiplier
from src.trade_plan import SizingProfile, TradeSetup, build_trade_plan
from src.symbol_score import score_symbol_result


def test_hype_is_liquid_alt() -> None:
    assert instrument_profile("HYPE-USDT-SWAP") == "liquid_alt"


def test_universe_contains_hype_once() -> None:
    universe = listed_universe()
    assert universe.count("HYPE-USDT-SWAP") == 1


def test_alt_risk_is_below_core() -> None:
    assert profile_risk_multiplier("liquid_alt") < profile_risk_multiplier("core")
    assert profile_risk_multiplier("watch") < profile_risk_multiplier("liquid_alt")


def test_low_price_contracts_keep_sub_cent_precision() -> None:
    setup = TradeSetup(
        direction="long",
        style="trend_pullback",
        entry=0.092345,
        invalidation=0.089111,
        confidence=0.9,
        reasons=("precision_check",),
    )
    sizing = SizingProfile(
        risk_fraction=0.01,
        max_margin_ratio=0.25,
        take_profit_r=2.0,
        partial_take_profit_r=1.0,
        partial_exit_fraction=0.3,
        trail_atr_mult=3.2,
        hold_for_trend=False,
        leverage=5,
    )

    plan = build_trade_plan(setup, atr=0.0012, equity=1000.0, sizing=sizing)

    assert plan is not None
    assert plan.entry == 0.092345
    assert plan.stop != round(plan.stop, 2)
    assert plan.trailing_distance > 0


def test_symbol_score_pauses_bad_recent_fit() -> None:
    score = score_symbol_result(
        {
            "trades": 75,
            "return_pct": -14.72,
            "profit_factor": 0.612,
            "win_rate": 0.413,
            "max_drawdown_pct": 20.87,
        }
    )
    assert score.state == "paused"
    assert score.weight == 0.0


def test_symbol_score_reduces_borderline_recent_fit() -> None:
    score = score_symbol_result(
        {
            "trades": 62,
            "return_pct": -3.32,
            "profit_factor": 0.837,
            "win_rate": 0.371,
            "max_drawdown_pct": 12.36,
        }
    )
    assert score.state == "reduced"
    assert 0.0 < score.weight < 1.0


def test_symbol_score_keeps_good_recent_fit_active() -> None:
    score = score_symbol_result(
        {
            "trades": 87,
            "return_pct": 9.64,
            "profit_factor": 1.266,
            "win_rate": 0.471,
            "max_drawdown_pct": 13.21,
        }
    )
    assert score.state == "active"
    assert score.weight == 1.0


if __name__ == "__main__":
    test_hype_is_liquid_alt()
    test_universe_contains_hype_once()
    test_alt_risk_is_below_core()
    test_low_price_contracts_keep_sub_cent_precision()
    test_symbol_score_pauses_bad_recent_fit()
    test_symbol_score_reduces_borderline_recent_fit()
    test_symbol_score_keeps_good_recent_fit_active()
    print("Universe rule tests passed")
