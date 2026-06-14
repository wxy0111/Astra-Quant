"""Run an OHLCV backtest for the V4 trend-risk strategy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.agent_market import choose_agent_proposal, generate_agent_setups, score_agent_results
from src.data import add_indicators, load_ohlcv_csv
from src.exit_engine import evaluate_exit
from src.market_state import classify_market
from src.meta_features import row_from_context
from src.meta_filter import evaluate_setup_quality
from src.meta_model import MetaModel, load_meta_model
from src.multi_timeframe import build_higher_state_cache, build_higher_timeframe_df
from src.pyramiding_engine import evaluate_pyramid_add
from src.risk_engine import evaluate_plan
from src.safety_margin import safety_margin_multiplier
from src.regime_features import regime_features_from_window
from src.regime_model import LightGBMRegimeModel, load_regime_model
from src.signal_engine import generate_setup
from src.trade_plan import TradePlan, build_trade_plan, round_price
from src.universe_engine import infer_inst_id_from_csv, instrument_profile, profile_risk_multiplier
from src.walk_forward import MarketGate, rolling_gate_allows_direction


@dataclass
class Position:
    plan: TradePlan
    entry_idx: int
    entry_ts: object
    entry_price: float
    qty: float
    best_price: float
    model_multiplier: float = 1.0
    meta_probability: float | None = None
    regime_label: str | None = None
    regime_probability: float | None = None
    safety_score: float | None = None
    safety_expected_edge_r: float | None = None
    safety_reason: str | None = None
    agent: str | None = None
    thesis: str | None = None
    agent_weight: float | None = None
    partial_taken: bool = False
    add_count: int = 0


def _entry_price(plan: TradePlan) -> float:
    slip = 1 + config.SLIPPAGE_RATE if plan.direction == "long" else 1 - config.SLIPPAGE_RATE
    return plan.entry * slip


def _slipped_price(direction: str, price: float, is_entry: bool) -> float:
    if direction == "long":
        slip = 1 + config.SLIPPAGE_RATE if is_entry else 1 - config.SLIPPAGE_RATE
    else:
        slip = 1 - config.SLIPPAGE_RATE if is_entry else 1 + config.SLIPPAGE_RATE
    return price * slip


def _exit_pnl(position: Position, exit_price: float) -> float:
    if position.plan.direction == "long":
        gross = (exit_price - position.entry_price) * position.qty
    else:
        gross = (position.entry_price - exit_price) * position.qty
    fees = (position.entry_price * position.qty + exit_price * position.qty) * config.FEE_RATE
    return gross - fees


def run_backtest_frame(
    raw,
    csv_path: str = "<frame>",
    warmup: int = 160,
    profile: str | None = None,
    inst_id: str | None = None,
    meta_model: MetaModel | None = None,
    regime_model: LightGBMRegimeModel | None = None,
    market_breadth: float = 0.5,
    strength_rank: float = 0.5,
    rolling_market_gates: dict[object, MarketGate] | None = None,
    use_agent_market: bool = False,
    allowed_agents: set[str] | None = None,
    agent_weights: dict[str, float] | None = None,
) -> dict:
    """Backtest one already-loaded OHLCV frame and return summary statistics."""
    inst_id = inst_id or infer_inst_id_from_csv(csv_path)
    profile = profile or instrument_profile(inst_id)
    risk_multiplier = profile_risk_multiplier(profile)
    df = add_indicators(
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
    higher_df = build_higher_timeframe_df(raw)
    higher_state_cache = build_higher_state_cache(df, higher_df)

    equity = config.INITIAL_EQUITY
    peak = equity
    max_drawdown = 0.0
    position: Position | None = None
    trades: list[dict] = []
    blocked = 0
    signals = 0
    blocked_reasons: dict[str, int] = {}
    meta_predictions: list[float] = []
    regime_predictions: list[dict] = []
    safety_decisions: list[dict] = []

    for i in range(warmup, len(df)):
        row = df.iloc[i]

        if position is not None:
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            plan = position.plan

            if plan.direction == "long":
                position.best_price = max(position.best_price, high)
            else:
                position.best_price = min(position.best_price, low)

            higher_state = higher_state_cache[i] if i < len(higher_state_cache) else None
            add_decision = evaluate_pyramid_add(
                plan,
                mark_price=close,
                equity=equity,
                current_qty=position.qty,
                add_count=position.add_count,
                higher_state=higher_state,
            ) if position.partial_taken else None
            if add_decision is not None and add_decision.allowed:
                add_price = _slipped_price(plan.direction, add_decision.price, is_entry=True)
                old_notional = position.entry_price * position.qty
                add_notional = add_price * add_decision.qty
                new_qty = position.qty + add_decision.qty
                if new_qty > 0:
                    position.entry_price = (old_notional + add_notional) / new_qty
                    position.qty = new_qty
                    position.add_count += 1
                    fee = add_notional * config.FEE_RATE
                    equity -= fee
                    trades.append(
                        {
                            "entry_ts": position.entry_ts,
                            "exit_ts": row["ts"],
                            "direction": plan.direction,
                            "style": plan.style,
                            "event": "pyramid_add",
                            "leverage": plan.leverage,
                            "model_multiplier": round(position.model_multiplier, 4),
                            "safety_score": position.safety_score,
                            "safety_reason": position.safety_reason,
                            "agent": position.agent,
                            "agent_weight": position.agent_weight,
                            "thesis": position.thesis,
                            "entry": round_price(add_price),
                            "exit": round_price(add_price),
                            "pnl": round(-fee, 4),
                            "equity": round(equity, 4),
                        }
                    )

            exit_decision = evaluate_exit(
                plan,
                high=high,
                low=low,
                best_price=position.best_price,
                partial_taken=position.partial_taken,
            )

            if exit_decision.partial_price is not None and not position.partial_taken and position.qty > 0:
                partial_qty = position.qty * plan.partial_exit_fraction
                partial = Position(
                    plan,
                    position.entry_idx,
                    position.entry_ts,
                    position.entry_price,
                    partial_qty,
                    position.best_price,
                    model_multiplier=position.model_multiplier,
                    meta_probability=position.meta_probability,
                    regime_label=position.regime_label,
                    regime_probability=position.regime_probability,
                    safety_score=position.safety_score,
                    safety_expected_edge_r=position.safety_expected_edge_r,
                    safety_reason=position.safety_reason,
                    agent=position.agent,
                    thesis=position.thesis,
                    agent_weight=position.agent_weight,
                    partial_taken=True,
                    add_count=position.add_count,
                )
                pnl = _exit_pnl(partial, exit_decision.partial_price)
                equity += pnl
                trades.append(
                    {
                        "entry_ts": position.entry_ts,
                        "exit_ts": row["ts"],
                        "direction": plan.direction,
                        "style": plan.style,
                        "event": "partial_take_profit",
                        "leverage": plan.leverage,
                        "model_multiplier": round(position.model_multiplier, 4),
                        "meta_probability": round(position.meta_probability, 4) if position.meta_probability is not None else None,
                        "regime_label": position.regime_label,
                        "safety_score": position.safety_score,
                        "safety_expected_edge_r": position.safety_expected_edge_r,
                        "safety_reason": position.safety_reason,
                        "agent": position.agent,
                        "agent_weight": position.agent_weight,
                        "thesis": position.thesis,
                        "entry": round_price(position.entry_price),
                        "exit": round_price(exit_decision.partial_price),
                        "pnl": round(pnl, 4),
                        "equity": round(equity, 4),
                    }
                )
                position.qty -= partial_qty
                position.partial_taken = True

            if exit_decision.final_price is not None:
                exit_slip = 1 - config.SLIPPAGE_RATE if plan.direction == "long" else 1 + config.SLIPPAGE_RATE
                final_exit = exit_decision.final_price * exit_slip
                pnl = _exit_pnl(position, final_exit)
                equity += pnl
                trades.append(
                    {
                        "entry_ts": position.entry_ts,
                        "exit_ts": row["ts"],
                        "direction": plan.direction,
                        "style": plan.style,
                        "event": "final_exit",
                        "leverage": plan.leverage,
                        "model_multiplier": round(position.model_multiplier, 4),
                        "meta_probability": round(position.meta_probability, 4) if position.meta_probability is not None else None,
                        "regime_label": position.regime_label,
                        "safety_score": position.safety_score,
                        "safety_expected_edge_r": position.safety_expected_edge_r,
                        "safety_reason": position.safety_reason,
                        "agent": position.agent,
                        "agent_weight": position.agent_weight,
                        "thesis": position.thesis,
                        "entry": round_price(position.entry_price),
                        "exit": round_price(final_exit),
                        "pnl": round(pnl, 4),
                        "equity": round(equity, 4),
                    }
                )
                position = None

        if position is None:
            window = df.iloc[: i + 1]
            state = classify_market(window)
            proposal = None
            if use_agent_market:
                proposals = generate_agent_setups(window, state)
                if agent_weights is not None:
                    proposals = [item for item in proposals if agent_weights.get(item.agent, 0.0) > 0]
                elif allowed_agents is not None:
                    proposals = [item for item in proposals if item.agent in allowed_agents]
                proposal = choose_agent_proposal(proposals)
                setup = proposal.setup if proposal is not None else None
            else:
                setup = generate_setup(window, state)
            if setup is not None:
                signals += 1
            higher_state = higher_state_cache[i] if i < len(higher_state_cache) else None
            rolling_gate = rolling_market_gates.get(row["ts"]) if rolling_market_gates else None
            quality = (
                evaluate_setup_quality(
                    setup,
                    state,
                    window,
                    higher_state,
                    profile=profile,
                    ignore_profile_disabled=use_agent_market,
                )
                if setup
                else None
            )
            model_multiplier = 1.0
            meta_probability = None
            regime_label = None
            regime_probability = None
            safety_score = None
            safety_expected_edge_r = None
            safety_reason = None
            higher_aligned = False
            agent_name = proposal.agent if proposal is not None else setup.style if setup is not None else None
            thesis = proposal.thesis if proposal is not None else None
            agent_weight = agent_weights.get(agent_name, 1.0) if agent_weights is not None and agent_name is not None else None
            if setup is not None and rolling_gate is not None and not rolling_gate_allows_direction(rolling_gate, setup.direction):
                blocked += 1
                blocked_reasons[rolling_gate.reason] = blocked_reasons.get(rolling_gate.reason, 0) + 1
                plan = None
            elif quality is not None and not quality.allowed:
                blocked += 1
                blocked_reasons[quality.reason] = blocked_reasons.get(quality.reason, 0) + 1
                plan = None
            else:
                plan_equity = equity * risk_multiplier
                plan = build_trade_plan(setup, float(row["atr"]), plan_equity, sizing=quality.sizing) if setup and quality else None
                if plan is not None and meta_model is not None:
                    higher_aligned = bool(higher_state and higher_state.regime == "trend" and higher_state.direction == plan.direction)
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
                    meta_predictions.append(meta_probability)
                    meta_mult = meta_model.multiplier_for_probability(meta_probability)
                    model_multiplier *= meta_mult
                    if meta_mult <= 0:
                        blocked += 1
                        blocked_reasons["meta_model_rejected"] = blocked_reasons.get("meta_model_rejected", 0) + 1
                        plan = None
                if plan is not None and regime_model is not None:
                    regime_features = regime_features_from_window(window)
                    regime_mult, regime_label, regime_probability = regime_model.risk_multiplier(regime_features, plan.direction)
                    regime_decision = regime_model.decision(regime_features, plan.direction)
                    regime_predictions.append(
                        {
                            "label": regime_decision.label,
                            "probability": regime_decision.probability,
                            "allowed": regime_mult > 0,
                            "multiplier": regime_mult,
                            "reason": regime_decision.reason,
                        }
                    )
                    model_multiplier *= regime_mult
                    if regime_mult <= 0:
                        blocked += 1
                        blocked_reasons[regime_decision.reason] = blocked_reasons.get(regime_decision.reason, 0) + 1
                        plan = None
                if plan is not None:
                    if agent_weight is not None:
                        model_multiplier *= agent_weight
                        if agent_weight <= 0:
                            blocked += 1
                            blocked_reasons["agent_weight_zero"] = blocked_reasons.get("agent_weight_zero", 0) + 1
                            plan = None
                if plan is not None:
                    safety = safety_margin_multiplier(
                        plan,
                        meta_probability=meta_probability,
                        regime_multiplier=model_multiplier,
                        higher_aligned=higher_aligned,
                        market_breadth=market_breadth,
                        strength_rank=strength_rank,
                    )
                    safety_score = safety.score
                    safety_expected_edge_r = safety.expected_edge_r
                    safety_reason = safety.reason
                    safety_decisions.append(
                        {
                            "score": safety.score,
                            "expected_edge_r": safety.expected_edge_r,
                            "multiplier": safety.multiplier,
                            "reason": safety.reason,
                        }
                    )
                    model_multiplier *= safety.multiplier
                    if safety.multiplier <= 0:
                        blocked += 1
                        blocked_reasons[safety.reason] = blocked_reasons.get(safety.reason, 0) + 1
                        plan = None
                if plan is not None and model_multiplier != 1.0:
                    adjusted_equity = max(plan_equity * model_multiplier, 0.0)
                    plan = build_trade_plan(setup, float(row["atr"]), adjusted_equity, sizing=quality.sizing)
            decision = evaluate_plan(plan, state)
            if decision.allowed and decision.plan is not None:
                entry = _entry_price(decision.plan)
                position = Position(
                    decision.plan,
                    i,
                    row["ts"],
                    entry,
                    decision.plan.qty,
                    entry,
                    model_multiplier=max(model_multiplier, 0.0),
                    meta_probability=meta_probability,
                    regime_label=regime_label,
                    regime_probability=regime_probability,
                    safety_score=safety_score,
                    safety_expected_edge_r=safety_expected_edge_r,
                    safety_reason=safety_reason,
                    agent=agent_name,
                    thesis=thesis,
                    agent_weight=agent_weight,
                )
            elif setup is not None:
                if quality is None or quality.allowed:
                    blocked += 1
                    blocked_reasons[decision.reason] = blocked_reasons.get(decision.reason, 0) + 1

        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)

    wins = [trade for trade in trades if trade["pnl"] > 0]
    losses = [trade for trade in trades if trade["pnl"] <= 0]
    gross_profit = sum(trade["pnl"] for trade in wins)
    gross_loss = abs(sum(trade["pnl"] for trade in losses))
    longest_losing_streak = 0
    current_losing_streak = 0
    for trade in trades:
        if trade["pnl"] <= 0:
            current_losing_streak += 1
            longest_losing_streak = max(longest_losing_streak, current_losing_streak)
        else:
            current_losing_streak = 0
    by_style: dict[str, dict[str, float]] = {}
    for trade in trades:
        item = by_style.setdefault(trade["style"], {"trades": 0, "pnl": 0.0, "wins": 0})
        item["trades"] += 1
        item["pnl"] += trade["pnl"]
        if trade["pnl"] > 0:
            item["wins"] += 1
    for item in by_style.values():
        item["pnl"] = round(item["pnl"], 4)
        item["win_rate"] = round(item["wins"] / item["trades"], 4) if item["trades"] else 0.0
    event_counts = dict(Counter(trade.get("event", "unknown") for trade in trades))
    leverages = [float(trade.get("leverage", 0.0)) for trade in trades if trade.get("event") != "pyramid_add"]
    multipliers = [float(trade.get("model_multiplier", 1.0)) for trade in trades if trade.get("event") != "pyramid_add"]
    total_pnl = equity - config.INITIAL_EQUITY
    return {
        "csv": csv_path,
        "inst_id": inst_id,
        "profile": profile,
        "risk_multiplier": risk_multiplier,
        "rows": len(df),
        "signals": signals,
        "blocked": blocked,
        "blocked_reasons": blocked_reasons,
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else None,
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
        "best_trade": round(max((trade["pnl"] for trade in trades), default=0.0), 4),
        "worst_trade": round(min((trade["pnl"] for trade in trades), default=0.0), 4),
        "avg_trade": round(sum((trade["pnl"] for trade in trades), 0.0) / len(trades), 4) if trades else 0.0,
        "longest_losing_streak": longest_losing_streak,
        "initial_equity": config.INITIAL_EQUITY,
        "final_equity": round(equity, 4),
        "total_pnl": round(total_pnl, 4),
        "return_pct": round(total_pnl / config.INITIAL_EQUITY * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "by_style": by_style,
        "by_agent": score_agent_results(trades),
        "event_counts": event_counts,
        "agent_market_used": use_agent_market,
        "allowed_agents": sorted(allowed_agents) if allowed_agents is not None else None,
        "agent_weights": agent_weights,
        "avg_leverage": round(sum(leverages) / len(leverages), 2) if leverages else 0.0,
        "max_leverage": max(leverages) if leverages else 0.0,
        "avg_model_multiplier": round(sum(multipliers) / len(multipliers), 4) if multipliers else 0.0,
        "safety_margin_used": True,
        "safety_decisions": len(safety_decisions),
        "safety_allowed": sum(1 for item in safety_decisions if item["multiplier"] > 0),
        "safety_avg_score": round(sum(item["score"] for item in safety_decisions) / len(safety_decisions), 4) if safety_decisions else None,
        "safety_avg_expected_edge_r": round(sum(item["expected_edge_r"] for item in safety_decisions) / len(safety_decisions), 4) if safety_decisions else None,
        "rolling_market_gate_used": rolling_market_gates is not None,
        "meta_model_used": meta_model is not None,
        "meta_avg_probability": round(sum(meta_predictions) / len(meta_predictions), 4) if meta_predictions else None,
        "meta_predictions": len(meta_predictions),
        "regime_model_used": regime_model is not None,
        "regime_predictions": len(regime_predictions),
        "regime_allowed": sum(1 for item in regime_predictions if item["allowed"]),
        "last_trades": trades[-5:],
    }


def run_backtest(
    csv_path: str,
    warmup: int = 160,
    profile: str | None = None,
    meta_model: MetaModel | None = None,
    regime_model: LightGBMRegimeModel | None = None,
    market_breadth: float = 0.5,
    strength_rank: float = 0.5,
    rolling_market_gates: dict[object, MarketGate] | None = None,
    use_agent_market: bool = False,
    allowed_agents: set[str] | None = None,
    agent_weights: dict[str, float] | None = None,
) -> dict:
    """Backtest one CSV and return summary statistics."""
    raw = load_ohlcv_csv(csv_path)
    return run_backtest_frame(
        raw,
        csv_path=csv_path,
        warmup=warmup,
        profile=profile,
        meta_model=meta_model,
        regime_model=regime_model,
        market_breadth=market_breadth,
        strength_rank=strength_rank,
        rolling_market_gates=rolling_market_gates,
        use_agent_market=use_agent_market,
        allowed_agents=allowed_agents,
        agent_weights=agent_weights,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V4 trend-risk backtest")
    parser.add_argument("--csv", required=True, help="Path to OKX candle CSV")
    parser.add_argument("--warmup", type=int, default=160)
    parser.add_argument("--profile", default=None, choices=["core", "liquid_alt", "watch", "unknown"])
    parser.add_argument("--meta-model", default=None, help="Optional JSON meta model used to filter entries")
    parser.add_argument("--regime-model", default=None, help="Optional JSON regime model used to filter market shape")
    parser.add_argument("--agent-market", action="store_true", help="Use local multi-agent signal market instead of the legacy single signal generator")
    parser.add_argument("--json-out", default=None, help="Optional path for JSON summary")
    args = parser.parse_args()

    result = run_backtest(
        args.csv,
        warmup=args.warmup,
        profile=args.profile,
        meta_model=load_meta_model(args.meta_model),
        regime_model=load_regime_model(args.regime_model),
        use_agent_market=args.agent_market,
    )
    print("V4 backtest summary")
    for key, value in result.items():
        if key == "last_trades":
            continue
        print(f"{key}: {value}")
    if result["last_trades"]:
        print("last_trades:")
        for trade in result["last_trades"]:
            print(trade)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, default=str, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
