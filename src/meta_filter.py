"""V3 setup-quality filter and dynamic risk sizing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config
from src.failed_breakout_filter import breakout_failure_risk
from src.leverage_engine import choose_leverage
from src.market_state import MarketState
from src.trade_plan import SizingProfile, TradeSetup


@dataclass(frozen=True)
class SetupQualityDecision:
    """Quality-filter verdict before building a trade plan."""

    allowed: bool
    reason: str
    sizing: SizingProfile
    confidence: float

    @property
    def risk_fraction(self) -> float:
        """Risk fraction chosen for the setup."""
        return self.sizing.risk_fraction


def _default_sizing() -> SizingProfile:
    return SizingProfile(
        risk_fraction=config.PROBE_RISK,
        max_margin_ratio=config.RANGE_MARGIN_RATIO,
        take_profit_r=config.RANGE_TAKE_PROFIT_R,
        partial_take_profit_r=config.RANGE_PARTIAL_TAKE_PROFIT_R,
        partial_exit_fraction=config.PARTIAL_EXIT_FRACTION,
        trail_atr_mult=config.TRAIL_ATR_MULT,
        hold_for_trend=False,
        leverage=config.PROBE_LEVERAGE,
    )


def _recent_shock(window: pd.DataFrame) -> bool:
    recent = window.tail(config.SHOCK_LOOKBACK_BARS)
    if recent.empty:
        return False
    return bool((recent["tr"] >= recent["atr"] * config.SHOCK_ATR_MULT).any())


def _higher_timeframe_aligned(setup: TradeSetup, higher_state: MarketState | None) -> bool:
    if higher_state is None:
        return False
    return higher_state.regime == "trend" and higher_state.direction == setup.direction


def _higher_timeframe_conflicts(setup: TradeSetup, higher_state: MarketState | None) -> bool:
    if higher_state is None or higher_state.direction == "none":
        return False
    return higher_state.regime == "trend" and higher_state.direction != setup.direction


def evaluate_setup_quality(
    setup: TradeSetup,
    state: MarketState,
    window: pd.DataFrame,
    higher_state: MarketState | None,
    profile: str = "core",
    ignore_profile_disabled: bool = False,
) -> SetupQualityDecision:
    """Return whether a raw setup is worth trading and at what risk."""
    fallback = _default_sizing()
    disabled_styles = config.DISABLED_STYLES_BY_PROFILE.get(profile, config.DISABLED_STYLES_BY_PROFILE["unknown"])
    if not ignore_profile_disabled and setup.style in disabled_styles:
        return SetupQualityDecision(False, "profile_style_disabled", fallback, setup.confidence)
    if setup.confidence < config.MIN_CONFIDENCE:
        return SetupQualityDecision(False, "confidence_too_low", fallback, setup.confidence)
    if state.risk_score >= 0.75:
        return SetupQualityDecision(False, "market_risk_too_high", fallback, setup.confidence)
    if _higher_timeframe_conflicts(setup, higher_state):
        return SetupQualityDecision(False, "higher_timeframe_conflict", fallback, setup.confidence)

    shock = _recent_shock(window)
    if shock and setup.style == "trend_pullback":
        return SetupQualityDecision(False, "shock_cooldown_pullback", fallback, setup.confidence)

    aligned = _higher_timeframe_aligned(setup, higher_state)
    confidence = setup.confidence + (0.06 if aligned else 0.0) - (0.08 if shock else 0.0)

    if setup.style == "trend_breakout":
        if config.REQUIRE_ALIGNED_BREAKOUT and not aligned:
            return SetupQualityDecision(False, "breakout_not_aligned", fallback, confidence)
        if confidence < config.MIN_BREAKOUT_CONFIDENCE:
            return SetupQualityDecision(False, "breakout_confidence_too_low", fallback, confidence)

    failure = breakout_failure_risk(setup, window)
    if failure.risky:
        reason = "failed_breakout_probe"
        sizing = SizingProfile(
            risk_fraction=config.PROBE_RISK,
            max_margin_ratio=config.RANGE_MARGIN_RATIO,
            take_profit_r=config.TAKE_PROFIT_R,
            partial_take_profit_r=config.PARTIAL_TAKE_PROFIT_R,
            partial_exit_fraction=config.PARTIAL_EXIT_FRACTION,
            trail_atr_mult=config.TRAIL_ATR_MULT,
            hold_for_trend=True,
            leverage=choose_leverage(setup, state, higher_state, reason=reason, shock=shock, failed_breakout=True),
        )
        return SetupQualityDecision(True, reason, sizing, max(confidence, config.MIN_CONFIDENCE))

    if shock and setup.style == "trend_breakout" and not aligned:
        reason = "shock_breakout_probe"
        sizing = SizingProfile(
            risk_fraction=config.PROBE_RISK,
            max_margin_ratio=config.RANGE_MARGIN_RATIO,
            take_profit_r=config.TAKE_PROFIT_R,
            partial_take_profit_r=config.PARTIAL_TAKE_PROFIT_R,
            partial_exit_fraction=config.PARTIAL_EXIT_FRACTION,
            trail_atr_mult=config.TRAIL_ATR_MULT,
            hold_for_trend=True,
            leverage=choose_leverage(setup, state, higher_state, reason=reason, shock=shock, failed_breakout=False),
        )
        return SetupQualityDecision(True, reason, sizing, max(confidence, config.MIN_CONFIDENCE))

    if setup.style == "trend_pullback" and (not aligned or confidence < config.PULLBACK_MIN_CONFIDENCE):
        return SetupQualityDecision(False, "pullback_not_confirmed", fallback, confidence)

    if setup.style == "trend_breakout" and aligned and confidence >= 0.75:
        reason = "high_conviction_breakout"
        sizing = SizingProfile(
            risk_fraction=config.HIGH_CONVICTION_RISK,
            max_margin_ratio=config.HIGH_CONVICTION_MARGIN_RATIO,
            take_profit_r=config.TAKE_PROFIT_R,
            partial_take_profit_r=config.PARTIAL_TAKE_PROFIT_R,
            partial_exit_fraction=config.PARTIAL_EXIT_FRACTION,
            trail_atr_mult=config.TRAIL_ATR_MULT,
            hold_for_trend=True,
            leverage=choose_leverage(setup, state, higher_state, reason=reason, shock=shock, failed_breakout=False),
        )
        return SetupQualityDecision(True, reason, sizing, min(confidence, 0.98))

    if setup.style in ("trend_breakout", "trend_pullback") and state.regime == "trend":
        reason = "trend_allowed"
        risk_fraction = config.NORMAL_TREND_RISK if aligned else config.RISK_PER_TRADE
        sizing = SizingProfile(
            risk_fraction=risk_fraction,
            max_margin_ratio=config.NORMAL_MARGIN_RATIO,
            take_profit_r=config.TAKE_PROFIT_R,
            partial_take_profit_r=config.PARTIAL_TAKE_PROFIT_R,
            partial_exit_fraction=config.PARTIAL_EXIT_FRACTION,
            trail_atr_mult=config.TRAIL_ATR_MULT,
            hold_for_trend=True,
            leverage=choose_leverage(setup, state, higher_state, reason=reason, shock=shock, failed_breakout=False),
        )
        return SetupQualityDecision(True, reason, sizing, min(confidence, 0.95))

    if setup.style == "range_reversion" and state.regime == "range" and not shock:
        reason = "range_allowed"
        sizing = SizingProfile(
            risk_fraction=config.RANGE_RISK,
            max_margin_ratio=config.RANGE_MARGIN_RATIO,
            take_profit_r=config.RANGE_TAKE_PROFIT_R,
            partial_take_profit_r=config.RANGE_PARTIAL_TAKE_PROFIT_R,
            partial_exit_fraction=config.PARTIAL_EXIT_FRACTION,
            trail_atr_mult=config.TRAIL_ATR_MULT * 0.8,
            hold_for_trend=False,
            leverage=choose_leverage(setup, state, higher_state, reason=reason, shock=shock, failed_breakout=False),
        )
        return SetupQualityDecision(True, reason, sizing, setup.confidence)

    if setup.style == "volatility_breakout" and state.risk_score < 0.75:
        reason = "volatility_probe"
        sizing = SizingProfile(
            risk_fraction=config.PROBE_RISK,
            max_margin_ratio=config.RANGE_MARGIN_RATIO,
            take_profit_r=config.TAKE_PROFIT_R,
            partial_take_profit_r=config.PARTIAL_TAKE_PROFIT_R,
            partial_exit_fraction=config.PARTIAL_EXIT_FRACTION,
            trail_atr_mult=config.TRAIL_ATR_MULT,
            hold_for_trend=True,
            leverage=choose_leverage(setup, state, higher_state, reason=reason, shock=shock, failed_breakout=False),
        )
        return SetupQualityDecision(True, reason, sizing, min(max(setup.confidence, config.MIN_CONFIDENCE), 0.80))

    return SetupQualityDecision(False, "unsupported_regime_or_shock", fallback, setup.confidence)
