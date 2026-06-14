"""Run rolling multi-window walk-forward tests over a universe dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest.run_universe_backtest import _csv_map
from backtest.run_v2_backtest import run_backtest_frame
from src.agent_market import agent_weights_from_scoreboard, allowed_agents_from_scoreboard
from src.data import load_ohlcv_csv
from src.meta_model import load_meta_model
from src.regime_model import load_regime_model
from src.symbol_score import score_symbol_result
from src.universe_engine import instrument_profile, listed_universe
from src.walk_forward import assign_strength_ranks, market_gate, momentum_return_pct


@dataclass(frozen=True)
class RollingWindow:
    """One chronological train/test slice."""

    index: int
    train: pd.DataFrame
    test: pd.DataFrame
    train_start: object
    train_end: object
    test_start: object
    test_end: object


def rolling_windows(
    df: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
) -> list[RollingWindow]:
    """Return fixed-duration rolling train/test windows."""
    if df.empty:
        return []
    data = df.sort_values("ts").reset_index(drop=True)
    start = pd.Timestamp(data["ts"].iloc[0])
    end = pd.Timestamp(data["ts"].iloc[-1])
    windows: list[RollingWindow] = []
    cursor = start
    index = 0
    while True:
        train_start = cursor
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)
        if test_end > end + pd.Timedelta(seconds=1):
            break
        train = data[(data["ts"] >= train_start) & (data["ts"] < train_end)].reset_index(drop=True)
        test = data[(data["ts"] >= test_start) & (data["ts"] < test_end)].reset_index(drop=True)
        if not train.empty and not test.empty:
            windows.append(
                RollingWindow(
                    index=index,
                    train=train,
                    test=test,
                    train_start=train["ts"].iloc[0],
                    train_end=train["ts"].iloc[-1],
                    test_start=test["ts"].iloc[0],
                    test_end=test["ts"].iloc[-1],
                )
            )
            index += 1
        cursor = cursor + pd.Timedelta(days=step_days)
    return windows


def _agent_weights(train_result: dict, use_agent_market: bool) -> tuple[set[str] | None, dict[str, float] | None]:
    if not use_agent_market:
        return None, None
    allowed = allowed_agents_from_scoreboard(train_result.get("by_agent", []))
    weights = agent_weights_from_scoreboard(train_result.get("by_agent", []))
    if not allowed:
        allowed = {"trend_pullback_agent"}
    if not any(weight > 0 for weight in weights.values()):
        weights = {"trend_pullback_agent": 1.0}
    weights = dict(weights)
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
        weights["trend_pullback_agent"] = max(float(weights.get("trend_pullback_agent", 0.0)), 1.0)
    return allowed, weights


def _aggregate_agent_weights(results: list[dict]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for window in results:
        for item in window.get("results", []):
            for agent, weight in (item.get("agent_weights") or {}).items():
                row = summary.setdefault(agent, {"count": 0, "weight_sum": 0.0})
                row["count"] += 1
                row["weight_sum"] += float(weight)
    for row in summary.values():
        row["avg_weight"] = round(row["weight_sum"] / row["count"], 4) if row["count"] else 0.0
        row["weight_sum"] = round(row["weight_sum"], 4)
    return summary


def run_rolling_walk_forward(
    data_dir: str,
    include_watch: bool = False,
    warmup: int = 160,
    train_days: int = 120,
    test_days: int = 30,
    step_days: int = 30,
    meta_model_path: str | None = None,
    regime_model_path: str | None = None,
    use_agent_market: bool = False,
    max_windows: int | None = None,
) -> dict:
    """Run rolling train/test windows across available instruments."""
    data_path = Path(data_dir)
    available = _csv_map(data_path)
    instruments = listed_universe(include_watch=include_watch)
    meta_model = load_meta_model(meta_model_path)
    regime_model = load_regime_model(regime_model_path)
    frames = {}
    missing = []
    for inst_id in instruments:
        csv_path = available.get(inst_id)
        if csv_path is None:
            missing.append(inst_id)
            continue
        frames[inst_id] = load_ohlcv_csv(str(csv_path))
    if not frames:
        raise ValueError("no available frames")

    shortest = min(frames.values(), key=len)
    windows = rolling_windows(shortest, train_days=train_days, test_days=test_days, step_days=step_days)
    if max_windows is not None:
        windows = windows[: max(0, max_windows)]

    window_results = []
    for window in windows:
        train_records = []
        tests: dict[str, pd.DataFrame] = {}
        for inst_id, frame in frames.items():
            train = frame[(frame["ts"] >= window.train_start) & (frame["ts"] <= window.train_end)].reset_index(drop=True)
            test = frame[(frame["ts"] >= window.test_start) & (frame["ts"] <= window.test_end)].reset_index(drop=True)
            if len(train) <= warmup + 10 or len(test) <= warmup + 10:
                continue
            profile = instrument_profile(inst_id)
            train_result = run_backtest_frame(
                train,
                csv_path=f"{inst_id}#w{window.index}#train",
                warmup=warmup,
                profile=profile,
                inst_id=inst_id,
                meta_model=meta_model,
                regime_model=regime_model,
                use_agent_market=use_agent_market,
            )
            score = score_symbol_result(train_result)
            allowed, weights = _agent_weights(train_result, use_agent_market)
            train_records.append(
                {
                    "inst_id": inst_id,
                    "profile": profile,
                    "score": score,
                    "train_result": train_result,
                    "momentum_return_pct": round(momentum_return_pct(train), 4),
                    "allowed_agents": allowed,
                    "agent_weights": weights,
                }
            )
            tests[inst_id] = test

        gate = market_gate(train_records)
        ranked = assign_strength_ranks(train_records)
        results = []
        for item in ranked:
            inst_id = item["inst_id"]
            test = tests.get(inst_id)
            if test is None:
                continue
            test_result = run_backtest_frame(
                test,
                csv_path=f"{inst_id}#w{window.index}#test",
                warmup=warmup,
                profile=item["profile"],
                inst_id=inst_id,
                meta_model=meta_model,
                regime_model=regime_model,
                market_breadth=gate.breadth,
                strength_rank=item["strength_rank"],
                use_agent_market=use_agent_market,
                allowed_agents=item.get("allowed_agents") if use_agent_market else None,
                agent_weights=item.get("agent_weights") if use_agent_market else None,
            )
            weight = item["walk_forward_weight"] if gate.open else 0.0
            test_result.update(
                {
                    "inst_id": inst_id,
                    "train_return_pct": item["train_result"]["return_pct"],
                    "train_profit_factor": item["train_result"]["profit_factor"],
                    "train_trades": item["train_result"]["trades"],
                    "momentum_return_pct": item["momentum_return_pct"],
                    "strength_rank": item["strength_rank"],
                    "walk_forward_weight": weight,
                    "walk_forward_state": item["walk_forward_state"] if gate.open else "paused",
                    "walk_forward_reason": item["walk_forward_reason"] if gate.open else gate.reason,
                    "agent_weights": item.get("agent_weights") if use_agent_market else None,
                }
            )
            results.append(test_result)

        wf_weight = sum(item["walk_forward_weight"] for item in results)
        wf_pnl = sum(item["total_pnl"] * item["walk_forward_weight"] for item in results) / wf_weight if wf_weight else 0.0
        equal_weight_pnl = sum(item["total_pnl"] for item in results) / len(results) if results else 0.0
        window_results.append(
            {
                "window": window.index,
                "train_start": str(window.train_start),
                "train_end": str(window.train_end),
                "test_start": str(window.test_start),
                "test_end": str(window.test_end),
                "market_gate": {
                    "open": gate.open,
                    "breadth": gate.breadth,
                    "average_momentum_pct": gate.average_momentum_pct,
                    "reason": gate.reason,
                },
                "tested": len(results),
                "walk_forward_1000u_pnl": round(wf_pnl, 4),
                "walk_forward_1000u_return_pct": round(wf_pnl / 1000.0 * 100, 2),
                "equal_weight_1000u_pnl": round(equal_weight_pnl, 4),
                "equal_weight_1000u_return_pct": round(equal_weight_pnl / 1000.0 * 100, 2),
                "total_trades": sum(item["trades"] for item in results),
                "max_single_asset_drawdown_pct": max((item["max_drawdown_pct"] for item in results), default=0.0),
                "results": results,
            }
        )

    total_wf_pnl = sum(item["walk_forward_1000u_pnl"] for item in window_results)
    return {
        "mode": "rolling_walk_forward",
        "data_dir": str(data_path),
        "meta_model_path": meta_model_path,
        "regime_model_path": regime_model_path,
        "agent_market_used": use_agent_market,
        "train_days": train_days,
        "test_days": test_days,
        "step_days": step_days,
        "windows": len(window_results),
        "missing": missing,
        "sum_window_1000u_pnl": round(total_wf_pnl, 4),
        "sum_window_return_pct": round(total_wf_pnl / 1000.0 * 100, 2),
        "avg_window_return_pct": round(
            sum(item["walk_forward_1000u_return_pct"] for item in window_results) / len(window_results),
            4,
        ) if window_results else 0.0,
        "positive_windows": sum(1 for item in window_results if item["walk_forward_1000u_pnl"] > 0),
        "worst_window_return_pct": min((item["walk_forward_1000u_return_pct"] for item in window_results), default=0.0),
        "best_window_return_pct": max((item["walk_forward_1000u_return_pct"] for item in window_results), default=0.0),
        "max_window_drawdown_pct": max((item["max_single_asset_drawdown_pct"] for item in window_results), default=0.0),
        "agent_weight_summary": _aggregate_agent_weights(window_results),
        "window_results": window_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling walk-forward universe backtest")
    parser.add_argument("--data-dir", default="backtest/data_train_2y")
    parser.add_argument("--warmup", type=int, default=160)
    parser.add_argument("--no-watch", action="store_true")
    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--meta-model", default=None)
    parser.add_argument("--regime-model", default=None)
    parser.add_argument("--agent-market", action="store_true")
    parser.add_argument("--json-out", default="backtest/reports/rolling_walk_forward.json")
    args = parser.parse_args()

    result = run_rolling_walk_forward(
        args.data_dir,
        include_watch=not args.no_watch,
        warmup=args.warmup,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        meta_model_path=args.meta_model,
        regime_model_path=args.regime_model,
        use_agent_market=args.agent_market,
        max_windows=args.max_windows,
    )
    print("Rolling walk-forward summary")
    for key in (
        "windows",
        "sum_window_1000u_pnl",
        "sum_window_return_pct",
        "avg_window_return_pct",
        "positive_windows",
        "worst_window_return_pct",
        "best_window_return_pct",
        "max_window_drawdown_pct",
    ):
        print(f"{key}: {result[key]}")
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, default=str, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
