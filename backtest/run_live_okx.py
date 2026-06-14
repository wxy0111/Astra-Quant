"""Run one guarded live OKX decision and optionally place a real order.

Usage:
    python -m backtest.run_live_okx --inst BTC-USDT-SWAP --i-understand-live-risk
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src import config
from src.data import OKX_COLUMNS
from src.live_signal import build_live_signal
from src.meta_model import load_meta_model
from src.okx_client import (
    LiveRiskLimits,
    OKXRestClient,
    load_credentials,
    plan_to_order_sizing,
    validate_live_risk,
)
from src.regime_model import load_regime_model


DEFAULT_META_MODEL = "backtest/models/meta_model_lgbm_2y_until_20260415_agent_quality_t050.json"


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _candles_to_frame(candles: list[list[str]]) -> pd.DataFrame:
    rows = [row[: len(OKX_COLUMNS)] for row in candles if len(row) >= len(OKX_COLUMNS)]
    df = pd.DataFrame(rows, columns=OKX_COLUMNS)
    if df.empty:
        raise ValueError("OKX returned no candles")
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
    for col in ("open", "high", "low", "close", "vol"):
        df[col] = df[col].astype(float)
    return df.sort_values("ts").reset_index(drop=True)


def _usdt_equity(balance: dict[str, Any]) -> float:
    details = balance.get("details") or []
    for item in details:
        if item.get("ccy") == "USDT":
            for key in ("availEq", "eq", "cashBal"):
                raw = item.get(key)
                if raw not in (None, ""):
                    return float(raw)
    for key in ("totalEq", "adjEq"):
        raw = balance.get(key)
        if raw not in (None, ""):
            return float(raw)
    raise ValueError("could not infer USDT equity from OKX balance response")


def _has_open_position(positions: list[dict[str, Any]], inst_id: str) -> bool:
    for pos in positions:
        if pos.get("instId") != inst_id:
            continue
        raw = pos.get("pos", "0")
        try:
            if abs(float(raw)) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _append_log(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Place one guarded live OKX order from the current strategy signal")
    parser.add_argument("--inst", default=config.INST_ID)
    parser.add_argument("--bar", default=config.BAR)
    parser.add_argument("--td-mode", choices=["isolated", "cross"], default="isolated")
    parser.add_argument("--candle-limit", type=int, default=300)
    parser.add_argument("--meta-model", default=DEFAULT_META_MODEL)
    parser.add_argument("--regime-model", default=None)
    parser.add_argument("--equity-usdt", type=float, default=None, help="Override account equity used for sizing")
    parser.add_argument("--max-risk-usdt", type=float, default=config.LIVE_DEFAULT_MAX_RISK_USDT)
    parser.add_argument("--max-margin-usdt", type=float, default=config.LIVE_DEFAULT_MAX_MARGIN_USDT)
    parser.add_argument("--max-notional-usdt", type=float, default=config.LIVE_DEFAULT_MAX_NOTIONAL_USDT)
    parser.add_argument("--max-leverage", type=int, default=config.LIVE_DEFAULT_MAX_LEVERAGE)
    parser.add_argument("--allow-existing-position", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate the live order but do not send it")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--log", default="logs/live_orders.jsonl")
    parser.add_argument("--i-understand-live-risk", action="store_true", help="Required for real live orders")
    args = parser.parse_args()

    credentials = load_credentials(args.env)
    client = OKXRestClient(credentials)
    meta_model = load_meta_model(args.meta_model) if args.meta_model else None
    regime_model = load_regime_model(args.regime_model) if args.regime_model else None

    candles = client.get_candles(args.inst, args.bar, limit=args.candle_limit)
    raw = _candles_to_frame(candles)
    balance = client.get_balance("USDT")
    positions = client.get_positions(args.inst)
    account_equity = _usdt_equity(balance)
    equity = args.equity_usdt if args.equity_usdt is not None else account_equity

    if not args.allow_existing_position and _has_open_position(positions, args.inst):
        raise RuntimeError(f"{args.inst} already has an open position; use --allow-existing-position to override")

    decision = build_live_signal(
        raw,
        args.inst,
        equity_usdt=equity,
        meta_model=meta_model,
        regime_model=regime_model,
        use_agent_market=True,
    )
    log_row: dict[str, Any] = {
        "ts": int(time.time()),
        "inst_id": args.inst,
        "bar": args.bar,
        "simulated": credentials.simulated,
        "dry_run": args.dry_run,
        "account_equity": account_equity,
        "sizing_equity": equity,
        "decision": decision,
    }

    if not decision.allowed or decision.plan is None:
        _append_log(args.log, {**log_row, "event": "blocked"})
        print(json.dumps(_json_safe({**log_row, "event": "blocked"}), indent=2, ensure_ascii=False))
        return

    instrument = client.get_instrument(args.inst)
    sizing = plan_to_order_sizing(decision.plan, args.inst, instrument)
    limits = LiveRiskLimits(
        max_risk_usdt=args.max_risk_usdt,
        max_margin_usdt=args.max_margin_usdt,
        max_notional_usdt=args.max_notional_usdt,
        max_leverage=args.max_leverage,
    )
    validate_live_risk(decision.plan, sizing, limits)

    prepared = {
        **log_row,
        "event": "prepared",
        "order_sizing": sizing,
        "limits": limits,
    }
    if args.dry_run:
        _append_log(args.log, prepared)
        print(json.dumps(_json_safe(prepared), indent=2, ensure_ascii=False))
        return

    if not args.i_understand_live_risk:
        raise RuntimeError("real orders require --i-understand-live-risk")
    if credentials.simulated:
        raise RuntimeError("OKX_FLAG=1 is simulated trading; set OKX_FLAG=0 only when you intentionally want live trading")

    client.set_leverage(args.inst, decision.plan.leverage, td_mode=args.td_mode)
    client_order_id = f"okxnew{int(time.time())}"
    order_response = client.place_market_order_with_protection(
        inst_id=args.inst,
        td_mode=args.td_mode,
        side=sizing.side,
        size=sizing.size,
        take_profit=decision.plan.take_profit,
        stop_loss=decision.plan.stop,
        client_order_id=client_order_id,
    )
    placed = {
        **prepared,
        "event": "live_order_sent",
        "client_order_id": client_order_id,
        "order_response": order_response,
    }
    _append_log(args.log, placed)
    print(json.dumps(_json_safe(placed), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
