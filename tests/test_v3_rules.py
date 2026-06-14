"""Focused checks for V3 rule behavior.

Run with:
    python tests/test_v3_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.market_state import MarketState
from src.failed_breakout_filter import breakout_failure_risk
from src.meta_filter import evaluate_setup_quality
from src.pyramiding_engine import evaluate_pyramid_add
from src.trade_plan import SizingProfile, TradeSetup, build_trade_plan


def _shock_window() -> pd.DataFrame:
    rows = []
    for i in range(12):
        rows.append(
            {
                "ts": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=15 * i),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "tr": 2.0,
                "atr": 2.0,
                "atr_pct": 0.02,
                "adx": 25.0,
                "ema_fast": 100.0,
                "ema_slow": 100.0,
                "ema_trend": 100.0,
                "plus_di": 20.0,
                "minus_di": 30.0,
            }
        )
    rows[-2]["tr"] = config.SHOCK_ATR_MULT * rows[-2]["atr"] + 0.1
    return pd.DataFrame(rows)


def _calm_window() -> pd.DataFrame:
    rows = []
    for i in range(12):
        rows.append(
            {
                "ts": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=15 * i),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "tr": 2.0,
                "atr": 2.0,
                "atr_pct": 0.02,
                "adx": 15.0,
                "ema_fast": 100.0,
                "ema_slow": 100.0,
                "ema_trend": 100.0,
                "plus_di": 20.0,
                "minus_di": 20.0,
            }
        )
    return pd.DataFrame(rows)


def _failed_short_breakout_window() -> pd.DataFrame:
    rows = []
    for i in range(12):
        rows.append(
            {
                "ts": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=15 * i),
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 99.0,
                "tr": 4.0,
                "atr": 3.0,
                "atr_pct": 0.03,
                "adx": 30.0,
                "ema_fast": 101.0,
                "ema_slow": 102.0,
                "ema_trend": 103.0,
                "plus_di": 15.0,
                "minus_di": 35.0,
            }
        )
    rows[-1].update({"open": 100.0, "high": 101.0, "low": 90.0, "close": 98.8, "tr": 11.0, "atr": 3.0})
    return pd.DataFrame(rows)


def _clean_short_breakout_window() -> pd.DataFrame:
    rows = []
    for i in range(12):
        rows.append(
            {
                "ts": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=15 * i),
                "open": 102.0,
                "high": 103.0,
                "low": 99.0,
                "close": 100.0,
                "tr": 4.0,
                "atr": 3.0,
                "atr_pct": 0.03,
                "adx": 30.0,
                "ema_fast": 101.0,
                "ema_slow": 102.0,
                "ema_trend": 103.0,
                "plus_di": 15.0,
                "minus_di": 35.0,
            }
        )
    rows[-1].update({"open": 100.0, "high": 100.5, "low": 94.5, "close": 95.0, "tr": 6.0, "atr": 3.0})
    return pd.DataFrame(rows)


def test_shock_cooldown_blocks_trend_pullback() -> None:
    setup = TradeSetup("short", "trend_pullback", 100.0, 105.0, 0.8, ("ema_pullback",))
    state = MarketState("trend", "short", 0.8, 0.2, ("bear_ema_stack",))
    decision = evaluate_setup_quality(setup, state, _shock_window(), higher_state=state)
    assert not decision.allowed
    assert decision.reason == "shock_cooldown_pullback"


def test_aligned_breakout_is_disabled_in_production_profile() -> None:
    setup = TradeSetup("short", "trend_breakout", 100.0, 110.0, 0.88, ("breakout_low",))
    state = MarketState("trend", "short", 0.88, 0.1, ("bear_ema_stack",))
    decision = evaluate_setup_quality(setup, state, _clean_short_breakout_window(), higher_state=state)
    assert not decision.allowed
    assert decision.reason == "profile_style_disabled"


def test_agent_market_can_bypass_profile_style_disable_for_breakout() -> None:
    setup = TradeSetup("short", "trend_breakout", 95.0, 105.0, 0.88, ("breakout_low",))
    state = MarketState("trend", "short", 0.88, 0.1, ("bear_ema_stack",))
    decision = evaluate_setup_quality(
        setup,
        state,
        _clean_short_breakout_window(),
        higher_state=state,
        ignore_profile_disabled=True,
    )
    assert decision.allowed
    assert decision.reason == "high_conviction_breakout"


def test_core_range_reversion_is_disabled() -> None:
    setup = TradeSetup("long", "range_reversion", 100.0, 96.0, 0.6, ("below_boll",))
    state = MarketState("range", "none", 0.6, 0.1, ("low_vol_range",))
    decision = evaluate_setup_quality(setup, state, _calm_window(), higher_state=None)
    assert not decision.allowed
    assert decision.reason == "profile_style_disabled"


def test_agent_market_can_bypass_profile_style_disable_for_range() -> None:
    setup = TradeSetup("long", "range_reversion", 100.0, 96.0, 0.6, ("below_boll",))
    state = MarketState("range", "none", 0.6, 0.1, ("low_vol_range",))
    decision = evaluate_setup_quality(
        setup,
        state,
        _calm_window(),
        higher_state=None,
        ignore_profile_disabled=True,
    )
    assert decision.allowed
    assert decision.reason == "range_allowed"


def test_liquid_alt_range_reversion_is_disabled() -> None:
    setup = TradeSetup("long", "range_reversion", 100.0, 96.0, 0.6, ("below_boll",))
    state = MarketState("range", "none", 0.6, 0.1, ("low_vol_range",))
    decision = evaluate_setup_quality(setup, state, _calm_window(), higher_state=None, profile="liquid_alt")
    assert not decision.allowed
    assert decision.reason == "profile_style_disabled"


def test_unaligned_breakout_is_blocked() -> None:
    setup = TradeSetup("short", "trend_breakout", 100.0, 110.0, 0.82, ("breakout_low",))
    state = MarketState("trend", "short", 0.82, 0.1, ("bear_ema_stack",))
    higher_state = MarketState("range", "none", 0.6, 0.1, ("low_vol_range",))
    decision = evaluate_setup_quality(setup, state, _clean_short_breakout_window(), higher_state=higher_state)
    assert not decision.allowed
    assert decision.reason == "profile_style_disabled"


def test_failed_breakout_is_downgraded_to_probe() -> None:
    setup = TradeSetup("short", "trend_breakout", 98.8, 104.0, 0.82, ("breakout_low",))
    state = MarketState("trend", "short", 0.82, 0.2, ("bear_ema_stack",))
    window = _failed_short_breakout_window()
    assert breakout_failure_risk(setup, window).risky
    decision = evaluate_setup_quality(setup, state, window, higher_state=state)
    assert not decision.allowed
    assert decision.reason == "profile_style_disabled"


def test_profitable_trend_position_can_pyramid() -> None:
    setup = TradeSetup("long", "trend_breakout", 100.0, 96.0, 0.9, ("breakout_high",))
    state = MarketState("trend", "long", 0.9, 0.1, ("bull_ema_stack",))
    sizing = SizingProfile(
        risk_fraction=config.NORMAL_TREND_RISK,
        max_margin_ratio=config.NORMAL_MARGIN_RATIO,
        take_profit_r=config.TAKE_PROFIT_R,
        partial_take_profit_r=config.PARTIAL_TAKE_PROFIT_R,
        partial_exit_fraction=config.PARTIAL_EXIT_FRACTION,
        trail_atr_mult=config.TRAIL_ATR_MULT,
        hold_for_trend=True,
        leverage=config.NORMAL_TREND_LEVERAGE,
    )
    plan = build_trade_plan(setup, atr=2.0, equity=1000.0, sizing=sizing)
    assert plan is not None
    add = evaluate_pyramid_add(plan, mark_price=110.0, equity=1000.0, current_qty=plan.qty, add_count=0, higher_state=state)
    assert add.allowed
    assert add.qty > 0


if __name__ == "__main__":
    test_shock_cooldown_blocks_trend_pullback()
    test_aligned_breakout_is_disabled_in_production_profile()
    test_agent_market_can_bypass_profile_style_disable_for_breakout()
    test_core_range_reversion_is_disabled()
    test_agent_market_can_bypass_profile_style_disable_for_range()
    test_liquid_alt_range_reversion_is_disabled()
    test_unaligned_breakout_is_blocked()
    test_failed_breakout_is_downgraded_to_probe()
    test_profitable_trend_position_can_pyramid()
    print("V3 rule tests passed")
