"""Build a meta-labeling dataset from downloaded OKX candles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config
from src.agent_market import generate_agent_setups
from src.data import add_indicators, load_ohlcv_csv
from src.market_state import classify_market
from src.meta_features import row_from_context
from src.meta_filter import evaluate_setup_quality
from src.meta_label import label_plan_outcome
from src.multi_timeframe import build_higher_state_cache, build_higher_timeframe_df
from src.risk_engine import evaluate_plan
from src.signal_engine import generate_setup
from src.trade_plan import build_trade_plan
from src.universe_engine import infer_inst_id_from_csv, instrument_profile, profile_risk_multiplier


def build_rows_for_csv(
    csv_path: str | Path,
    warmup: int = 160,
    max_bars: int = 96,
    context_bars: int = 500,
    stride: int = 4,
    end_date: str | None = None,
    market_breadth: float = 0.0,
    strength_rank: float = 0.0,
    agent_market: bool = False,
) -> list[dict]:
    """Return labeled candidate-signal rows for one CSV."""
    csv_path = Path(csv_path)
    inst_id = infer_inst_id_from_csv(csv_path)
    profile = instrument_profile(inst_id)
    risk_multiplier = profile_risk_multiplier(profile)
    raw = load_ohlcv_csv(str(csv_path))
    if end_date:
        cutoff = pd.Timestamp(datetime.fromisoformat(end_date.replace("Z", "+00:00")).astimezone(timezone.utc)).tz_localize(None)
        raw = raw[raw["ts"] < cutoff].reset_index(drop=True)
        if len(raw) < warmup + max_bars + 10:
            return []
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
    rows = []

    end = max(warmup, len(df) - max_bars - 1)
    for i in range(warmup, end, stride):
        row = df.iloc[i]
        window = df.iloc[max(0, i - context_bars + 1) : i + 1]
        state = classify_market(window)
        if agent_market:
            proposals = generate_agent_setups(window, state)
            candidates = [(proposal.setup, proposal.agent, proposal.priority, proposal.thesis) for proposal in proposals]
            selected_agent = proposals[0].agent if proposals else None
        else:
            setup = generate_setup(window, state)
            candidates = [(setup, None, None, None)] if setup is not None else []
            selected_agent = None
        if not candidates:
            continue
        higher_state = higher_state_cache[i] if i < len(higher_state_cache) else None
        for setup, agent, priority, thesis in candidates:
            quality = evaluate_setup_quality(
                setup,
                state,
                window,
                higher_state,
                profile=profile,
                ignore_profile_disabled=agent_market,
            )
            if not quality.allowed:
                continue
            plan = build_trade_plan(setup, float(row["atr"]), config.INITIAL_EQUITY * risk_multiplier, sizing=quality.sizing)
            decision = evaluate_plan(plan, state)
            if not decision.allowed or decision.plan is None:
                continue
            plan = decision.plan
            future = df.iloc[i + 1 : i + 1 + max_bars]
            label = label_plan_outcome(plan, future, max_bars=max_bars)
            higher_aligned = bool(higher_state and higher_state.regime == "trend" and higher_state.direction == plan.direction)
            features = row_from_context(
                row,
                plan,
                higher_aligned=higher_aligned,
                market_breadth=market_breadth,
                strength_rank=strength_rank,
                window=window,
                agent=agent,
            )
            rows.append(
                {
                    "ts": row["ts"],
                    "inst_id": inst_id,
                    "profile": profile,
                    "agent": agent,
                    "agent_priority": priority,
                    "agent_thesis": thesis,
                    "selected_agent": int(agent == selected_agent) if selected_agent is not None else 0,
                    "direction": plan.direction,
                    "style": plan.style,
                    "label": label.label,
                    "quality_label": label.quality_label,
                    "label_reason": label.reason,
                    "bars_held": label.bars_held,
                    "pnl_r": round(label.pnl_r, 4),
                    "mfe_r": round(label.mfe_r, 4),
                    "mae_r": round(label.mae_r, 4),
                    "outcome_score": label.outcome_score,
                    "reached_partial": int(label.reached_partial),
                    "outcome_class": label.outcome_class,
                    **features,
                }
            )
    return rows


def build_dataset(
    data_dir: str | Path,
    out_path: str | Path,
    warmup: int = 160,
    max_bars: int = 96,
    context_bars: int = 500,
    stride: int = 4,
    end_date: str | None = None,
    agent_market: bool = False,
) -> pd.DataFrame:
    """Build and write a dataset from all recognized CSVs in a directory."""
    data_path = Path(data_dir)
    all_rows: list[dict] = []
    for csv_path in sorted(data_path.glob("*.csv")):
        if infer_inst_id_from_csv(csv_path) == "UNKNOWN-USDT-SWAP":
            continue
        rows = build_rows_for_csv(csv_path, warmup=warmup, max_bars=max_bars, context_bars=context_bars, stride=stride, end_date=end_date, agent_market=agent_market)
        print(f"{csv_path.name}: {len(rows)} labeled signals", flush=True)
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build meta-labeling signal dataset")
    parser.add_argument("--data-dir", default="backtest/data_train_2y")
    parser.add_argument("--out", default="backtest/datasets/meta_signals_2y.csv")
    parser.add_argument("--warmup", type=int, default=160)
    parser.add_argument("--max-bars", type=int, default=96)
    parser.add_argument("--context-bars", type=int, default=500)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--end-date", default=None, help="UTC exclusive cutoff for training rows, e.g. 2026-04-15T00:00:00Z")
    parser.add_argument("--agent-market", action="store_true", help="Build rows from all local Agent proposals instead of the legacy single setup")
    args = parser.parse_args()
    build_dataset(args.data_dir, args.out, warmup=args.warmup, max_bars=args.max_bars, context_bars=args.context_bars, stride=args.stride, end_date=args.end_date, agent_market=args.agent_market)


if __name__ == "__main__":
    main()
