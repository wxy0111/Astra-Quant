"""Run V4 backtests across a configured multi-asset universe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtest.run_v2_backtest import run_backtest, run_backtest_frame
from src.agent_market import agent_weights_from_scoreboard, allowed_agents_from_scoreboard
from src.data import load_ohlcv_csv
from src.meta_model import load_meta_model
from src.regime_model import load_regime_model
from src.symbol_score import score_symbol_result
from src.universe_engine import infer_inst_id_from_csv, instrument_profile, listed_universe
from src.walk_forward import assign_strength_ranks, market_gate, momentum_return_pct, rolling_market_gates_by_timestamp, split_train_test


def _csv_map(data_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(data_dir.glob("*.csv")):
        inst_id = infer_inst_id_from_csv(path)
        if inst_id == "UNKNOWN-USDT-SWAP":
            continue
        mapping.setdefault(inst_id, path)
    return mapping


def run_universe_backtest(
    data_dir: str,
    include_watch: bool = True,
    warmup: int = 160,
    meta_model_path: str | None = None,
    regime_model_path: str | None = None,
    use_agent_market: bool = False,
) -> dict:
    """Run one backtest per available instrument CSV and aggregate results."""
    data_path = Path(data_dir)
    available = _csv_map(data_path)
    instruments = listed_universe(include_watch=include_watch)
    meta_model = load_meta_model(meta_model_path)
    regime_model = load_regime_model(regime_model_path)
    results = []
    missing = []

    for inst_id in instruments:
        csv_path = available.get(inst_id)
        if csv_path is None:
            missing.append(inst_id)
            continue
        result = run_backtest(
            str(csv_path),
            warmup=warmup,
            profile=instrument_profile(inst_id),
            meta_model=meta_model,
            regime_model=regime_model,
            use_agent_market=use_agent_market,
        )
        result["inst_id"] = inst_id
        score = score_symbol_result(result)
        result["adaptive_score"] = score.score
        result["adaptive_state"] = score.state
        result["adaptive_weight"] = score.weight
        result["adaptive_reason"] = score.reason
        results.append(result)

    total_pnl = sum(item["total_pnl"] for item in results)
    total_trades = sum(item["trades"] for item in results)
    max_drawdown = max((item["max_drawdown_pct"] for item in results), default=0.0)
    tested = len(results)
    equal_weight_pnl = total_pnl / tested if tested else 0.0
    adaptive_weight = sum(item["adaptive_weight"] for item in results)
    adaptive_pnl = sum(item["total_pnl"] * item["adaptive_weight"] for item in results) / adaptive_weight if adaptive_weight else 0.0
    adaptive_states = {
        "active": sum(1 for item in results if item["adaptive_state"] == "active"),
        "reduced": sum(1 for item in results if item["adaptive_state"] == "reduced"),
        "paused": sum(1 for item in results if item["adaptive_state"] == "paused"),
    }
    return {
        "data_dir": str(data_path),
        "instruments": instruments,
        "tested": tested,
        "missing": missing,
        "independent_account_total_pnl": round(total_pnl, 4),
        "independent_account_return_pct_on_1000u": round(total_pnl / 1000.0 * 100, 2),
        "equal_weight_1000u_pnl": round(equal_weight_pnl, 4),
        "equal_weight_1000u_return_pct": round(equal_weight_pnl / 1000.0 * 100, 2),
        "adaptive_1000u_pnl": round(adaptive_pnl, 4),
        "adaptive_1000u_return_pct": round(adaptive_pnl / 1000.0 * 100, 2),
        "adaptive_weight_total": round(adaptive_weight, 4),
        "adaptive_states": adaptive_states,
        "total_trades": total_trades,
        "max_single_asset_drawdown_pct": max_drawdown,
        "results": results,
    }


def run_walk_forward_universe_backtest(
    data_dir: str,
    include_watch: bool = True,
    warmup: int = 160,
    meta_model_path: str | None = None,
    regime_model_path: str | None = None,
    enable_rolling_market_gate: bool = False,
    use_agent_market: bool = False,
) -> dict:
    """Select symbols on the first half, then test allocation on the second half."""
    data_path = Path(data_dir)
    available = _csv_map(data_path)
    instruments = listed_universe(include_watch=include_watch)
    meta_model = load_meta_model(meta_model_path)
    regime_model = load_regime_model(regime_model_path)
    train_records = []
    test_frames = {}
    missing = []

    for inst_id in instruments:
        csv_path = available.get(inst_id)
        if csv_path is None:
            missing.append(inst_id)
            continue
        raw = load_ohlcv_csv(str(csv_path))
        train_df, test_df = split_train_test(raw)
        profile = instrument_profile(inst_id)
        train_result = run_backtest_frame(
            train_df,
            csv_path=f"{csv_path}#train",
            warmup=warmup,
            profile=profile,
            inst_id=inst_id,
            meta_model=meta_model,
            regime_model=regime_model,
            use_agent_market=use_agent_market,
        )
        score = score_symbol_result(train_result)
        allowed_agents = (
            allowed_agents_from_scoreboard(train_result.get("by_agent", []))
            if use_agent_market
            else None
        )
        agent_weights = (
            agent_weights_from_scoreboard(train_result.get("by_agent", []))
            if use_agent_market
            else None
        )
        if use_agent_market and not allowed_agents:
            allowed_agents = {"trend_pullback_agent"}
        if use_agent_market and not any(weight > 0 for weight in (agent_weights or {}).values()):
            agent_weights = {"trend_pullback_agent": 1.0}
        if use_agent_market:
            agent_weights = dict(agent_weights or {})
            pullback_row = next(
                (row for row in train_result.get("by_agent", []) if row.get("agent") == "trend_pullback_agent"),
                None,
            )
            pullback_bad = (
                pullback_row is not None
                and float(pullback_row.get("pnl", 0.0)) < -20.0
                and float(pullback_row.get("profit_factor") or 0.0) < 0.75
            )
            if not pullback_bad:
                agent_weights["trend_pullback_agent"] = max(float(agent_weights.get("trend_pullback_agent", 0.0)), 1.0)
        train_records.append(
            {
                "inst_id": inst_id,
                "csv_path": str(csv_path),
                "profile": profile,
                "score": score,
                "train_result": train_result,
                "allowed_agents": allowed_agents,
                "agent_weights": agent_weights,
                "momentum_return_pct": round(momentum_return_pct(train_df), 4),
            }
        )
        test_frames[inst_id] = test_df

    gate = market_gate(train_records)
    ranked = assign_strength_ranks(train_records)
    rolling_gates = rolling_market_gates_by_timestamp(test_frames) if enable_rolling_market_gate else None
    results = []

    for item in ranked:
        inst_id = item["inst_id"]
        profile = item["profile"]
        test_result = run_backtest_frame(
            test_frames[inst_id],
            csv_path=f"{item['csv_path']}#test",
            warmup=warmup,
            profile=profile,
            inst_id=inst_id,
            meta_model=meta_model,
            regime_model=regime_model,
            market_breadth=gate.breadth,
            strength_rank=item["strength_rank"],
            rolling_market_gates=rolling_gates,
            use_agent_market=use_agent_market,
            allowed_agents=item.get("allowed_agents") if use_agent_market else None,
            agent_weights=item.get("agent_weights") if use_agent_market else None,
        )
        score = item["score"]
        weight = item["walk_forward_weight"] if gate.open else 0.0
        state = item["walk_forward_state"] if gate.open else "paused"
        reason = item["walk_forward_reason"] if gate.open else gate.reason
        test_result.update(
            {
                "inst_id": inst_id,
                "train_return_pct": item["train_result"]["return_pct"],
                "train_profit_factor": item["train_result"]["profit_factor"],
                "train_max_drawdown_pct": item["train_result"]["max_drawdown_pct"],
                "train_trades": item["train_result"]["trades"],
                "train_score": score.score,
                "train_score_state": score.state,
                "train_score_reason": score.reason,
                "momentum_return_pct": item["momentum_return_pct"],
                "strength_rank": item["strength_rank"],
                "walk_forward_state": state,
                "walk_forward_weight": weight,
                "walk_forward_reason": reason,
                "allowed_agents": sorted(item.get("allowed_agents") or []) if use_agent_market else None,
                "agent_weights": item.get("agent_weights") if use_agent_market else None,
            }
        )
        results.append(test_result)

    total_pnl = sum(item["total_pnl"] for item in results)
    total_trades = sum(item["trades"] for item in results)
    tested = len(results)
    equal_weight_pnl = total_pnl / tested if tested else 0.0
    wf_weight = sum(item["walk_forward_weight"] for item in results)
    wf_pnl = sum(item["total_pnl"] * item["walk_forward_weight"] for item in results) / wf_weight if wf_weight else 0.0
    states = {
        "active": sum(1 for item in results if item["walk_forward_state"] == "active"),
        "reduced": sum(1 for item in results if item["walk_forward_state"] == "reduced"),
        "paused": sum(1 for item in results if item["walk_forward_state"] == "paused"),
    }
    return {
        "mode": "walk_forward",
        "meta_model_path": meta_model_path,
        "regime_model_path": regime_model_path,
        "agent_market_used": use_agent_market,
        "data_dir": str(data_path),
        "instruments": instruments,
        "tested": tested,
        "missing": missing,
        "market_gate": {
            "open": gate.open,
            "breadth": gate.breadth,
            "average_momentum_pct": gate.average_momentum_pct,
            "reason": gate.reason,
        },
        "rolling_market_gate": {
            "enabled": rolling_gates is not None,
            "open_bars": sum(1 for item in rolling_gates.values() if item.open) if rolling_gates else 0,
            "closed_bars": sum(1 for item in rolling_gates.values() if not item.open) if rolling_gates else 0,
            "closed_reasons": {
                reason: sum(1 for item in rolling_gates.values() if item.reason == reason and not item.open)
                for reason in sorted({item.reason for item in rolling_gates.values() if not item.open})
            } if rolling_gates else {},
        },
        "equal_weight_1000u_pnl": round(equal_weight_pnl, 4),
        "equal_weight_1000u_return_pct": round(equal_weight_pnl / 1000.0 * 100, 2),
        "walk_forward_1000u_pnl": round(wf_pnl, 4),
        "walk_forward_1000u_return_pct": round(wf_pnl / 1000.0 * 100, 2),
        "walk_forward_weight_total": round(wf_weight, 4),
        "walk_forward_states": states,
        "total_trades": total_trades,
        "max_single_asset_drawdown_pct": max((item["max_drawdown_pct"] for item in results), default=0.0),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-asset universe backtest")
    parser.add_argument("--data-dir", default="backtest/data")
    parser.add_argument("--warmup", type=int, default=160)
    parser.add_argument("--no-watch", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--meta-model", default=None, help="Optional JSON meta model used to filter entries")
    parser.add_argument("--regime-model", default=None, help="Optional JSON regime model used to filter market shape")
    parser.add_argument("--rolling-market-gate", action="store_true", help="Optionally pause new entries when recent cross-asset momentum is weak or choppy")
    parser.add_argument("--agent-market", action="store_true", help="Use local multi-agent signal market instead of the legacy single signal generator")
    parser.add_argument("--json-out", default="backtest/reports/universe_backtest.json")
    args = parser.parse_args()

    if args.walk_forward:
        result = run_walk_forward_universe_backtest(
            args.data_dir,
            include_watch=not args.no_watch,
            warmup=args.warmup,
            meta_model_path=args.meta_model,
            regime_model_path=args.regime_model,
            enable_rolling_market_gate=args.rolling_market_gate,
            use_agent_market=args.agent_market,
        )
    else:
        result = run_universe_backtest(
            args.data_dir,
            include_watch=not args.no_watch,
            warmup=args.warmup,
            meta_model_path=args.meta_model,
            regime_model_path=args.regime_model,
            use_agent_market=args.agent_market,
        )
    print("Universe backtest summary")
    print(f"mode: {result.get('mode', 'standard')}")
    print(f"tested: {result['tested']}")
    print(f"missing: {result['missing']}")
    print(f"equal_weight_1000u_pnl: {result['equal_weight_1000u_pnl']}")
    print(f"equal_weight_1000u_return_pct: {result['equal_weight_1000u_return_pct']}%")
    if args.walk_forward:
        print(f"market_gate: {result['market_gate']}")
        print(f"walk_forward_1000u_pnl: {result['walk_forward_1000u_pnl']}")
        print(f"walk_forward_1000u_return_pct: {result['walk_forward_1000u_return_pct']}%")
        print(f"walk_forward_states: {result['walk_forward_states']}")
    else:
        print(f"independent_account_total_pnl: {result['independent_account_total_pnl']}")
        print(f"independent_account_return_pct_on_1000u: {result['independent_account_return_pct_on_1000u']}%")
        print(f"adaptive_1000u_pnl: {result['adaptive_1000u_pnl']}")
        print(f"adaptive_1000u_return_pct: {result['adaptive_1000u_return_pct']}%")
        print(f"adaptive_states: {result['adaptive_states']}")
    print(f"total_trades: {result['total_trades']}")
    print(f"max_single_asset_drawdown_pct: {result['max_single_asset_drawdown_pct']}")
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, default=str, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
