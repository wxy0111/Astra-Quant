"""Build one live trade decision from fresh candles."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config
from src.agent_market import choose_agent_proposal, generate_agent_setups
from src.data import add_indicators
from src.market_state import MarketState, classify_market
from src.meta_features import row_from_context
from src.meta_filter import evaluate_setup_quality
from src.meta_model import MetaModel
from src.multi_timeframe import build_higher_state_cache, build_higher_timeframe_df
from src.regime_features import regime_features_from_window
from src.regime_model import LightGBMRegimeModel
from src.risk_engine import evaluate_plan
from src.safety_margin import safety_margin_multiplier
from src.signal_engine import generate_setup
from src.trade_plan import TradePlan, TradeSetup, build_trade_plan
from src.universe_engine import instrument_profile, profile_risk_multiplier


@dataclass(frozen=True)
class LiveSignalDecision:
    """A single live signal verdict with diagnostics for logging."""

    allowed: bool
    reason: str
    plan: TradePlan | None
    setup: TradeSetup | None
    state: MarketState | None
    agent: str | None = None
    thesis: str | None = None
    meta_probability: float | None = None
    regime_label: str | None = None
    regime_probability: float | None = None
    model_multiplier: float = 1.0
    safety_reason: str | None = None


def _indicator_frame(raw: pd.DataFrame) -> pd.DataFrame:
    return add_indicators(
        raw,
        config.EMA_FAST,
        config.EMA_SLOW,
        config.EMA_TREND,
        config.ATR_PERIOD,
        config.ADX_PERIOD,
        config.DONCHIAN_PERIOD,
        config.BOLL_PERIOD,
        config.BOLL_STD,
    )


def build_live_signal(
    raw: pd.DataFrame,
    inst_id: str,
    equity_usdt: float,
    meta_model: MetaModel | None = None,
    regime_model: LightGBMRegimeModel | None = None,
    use_agent_market: bool = True,
    market_breadth: float = 0.5,
    strength_rank: float = 0.5,
) -> LiveSignalDecision:
    """Return the current live plan after the same filters used in research."""
    if len(raw) < 220:
        return LiveSignalDecision(False, "not_enough_candles", None, None, None)
    df = _indicator_frame(raw)
    if df.empty:
        return LiveSignalDecision(False, "not_enough_indicator_rows", None, None, None)

    profile = instrument_profile(inst_id)
    risk_multiplier = profile_risk_multiplier(profile)
    higher_df = build_higher_timeframe_df(raw)
    higher_state_cache = build_higher_state_cache(df, higher_df)
    row = df.iloc[-1]
    window = df
    state = classify_market(window)
    proposal = None
    if use_agent_market:
        proposal = choose_agent_proposal(generate_agent_setups(window, state))
        setup = proposal.setup if proposal is not None else None
    else:
        setup = generate_setup(window, state)
    if setup is None:
        return LiveSignalDecision(False, "no_setup", None, None, state)

    higher_state = higher_state_cache[-1] if higher_state_cache else None
    quality = evaluate_setup_quality(
        setup,
        state,
        window,
        higher_state,
        profile=profile,
        ignore_profile_disabled=use_agent_market,
    )
    if not quality.allowed:
        return LiveSignalDecision(False, quality.reason, None, setup, state, agent=proposal.agent if proposal else setup.style)

    plan_equity = equity_usdt * risk_multiplier
    plan = build_trade_plan(setup, float(row["atr"]), plan_equity, sizing=quality.sizing)
    decision = evaluate_plan(plan, state)
    if not decision.allowed or decision.plan is None:
        return LiveSignalDecision(False, decision.reason, None, setup, state, agent=proposal.agent if proposal else setup.style)
    plan = decision.plan

    agent_name = proposal.agent if proposal is not None else setup.style
    thesis = proposal.thesis if proposal is not None else None
    model_multiplier = 1.0
    meta_probability = None
    regime_label = None
    regime_probability = None
    higher_aligned = bool(higher_state and higher_state.regime == "trend" and higher_state.direction == plan.direction)

    if meta_model is not None:
        features = row_from_context(
            row,
            plan,
            higher_aligned=higher_aligned,
            market_breadth=market_breadth,
            strength_rank=strength_rank,
            window=window,
            agent=agent_name,
        )
        meta_probability = meta_model.predict_proba(features)
        meta_mult = meta_model.multiplier_for_probability(meta_probability)
        model_multiplier *= meta_mult
        if meta_mult <= 0:
            return LiveSignalDecision(False, "meta_model_rejected", None, setup, state, agent=agent_name, thesis=thesis, meta_probability=meta_probability, model_multiplier=model_multiplier)

    if regime_model is not None:
        regime_features = regime_features_from_window(window)
        regime_mult, regime_label, regime_probability = regime_model.risk_multiplier(regime_features, plan.direction)
        model_multiplier *= regime_mult
        if regime_mult <= 0:
            return LiveSignalDecision(
                False,
                "regime_model_rejected",
                None,
                setup,
                state,
                agent=agent_name,
                thesis=thesis,
                meta_probability=meta_probability,
                regime_label=regime_label,
                regime_probability=regime_probability,
                model_multiplier=model_multiplier,
            )

    safety = safety_margin_multiplier(
        plan,
        meta_probability=meta_probability,
        regime_multiplier=model_multiplier,
        higher_aligned=higher_aligned,
        market_breadth=market_breadth,
        strength_rank=strength_rank,
    )
    model_multiplier *= safety.multiplier
    if safety.multiplier <= 0:
        return LiveSignalDecision(
            False,
            safety.reason,
            None,
            setup,
            state,
            agent=agent_name,
            thesis=thesis,
            meta_probability=meta_probability,
            regime_label=regime_label,
            regime_probability=regime_probability,
            model_multiplier=model_multiplier,
            safety_reason=safety.reason,
        )

    if model_multiplier != 1.0:
        adjusted_equity = max(plan_equity * model_multiplier, 0.0)
        plan = build_trade_plan(setup, float(row["atr"]), adjusted_equity, sizing=quality.sizing)
        decision = evaluate_plan(plan, state)
        if not decision.allowed or decision.plan is None:
            return LiveSignalDecision(False, decision.reason, None, setup, state, agent=agent_name, thesis=thesis, model_multiplier=model_multiplier)
        plan = decision.plan

    return LiveSignalDecision(
        True,
        "allowed",
        plan,
        setup,
        state,
        agent=agent_name,
        thesis=thesis,
        meta_probability=meta_probability,
        regime_label=regime_label,
        regime_probability=regime_probability,
        model_multiplier=model_multiplier,
        safety_reason=safety.reason,
    )
