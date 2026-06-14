"""Continuously scan configured symbols and place guarded OKX live orders.

Usage:
    python -m backtest.run_live_loop_okx --dry-run
    python -m backtest.run_live_loop_okx --i-understand-live-risk
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from backtest.run_live_okx import _append_log, _candles_to_frame, _has_open_position, _json_safe, _usdt_equity
from src import config
from src.dashboard import run_dashboard
from src.live_signal import build_live_signal
from src.meta_model import load_meta_model
from src.okx_client import LiveRiskLimits, OKXRestClient, load_credentials, plan_to_order_sizing, validate_live_risk
from src.regime_model import load_regime_model
from src.universe_engine import listed_universe


DEFAULT_META_MODEL = "backtest/models/meta_model_lgbm_2y_until_20260415_agent_quality_t050.json"
DEFAULT_CLOUDFLARED = r"C:\Program Files (x86)\cloudflared\cloudflared.exe"


def _symbols_from_args(value: str | None, include_watch: bool) -> list[str]:
    """Return symbols selected for the live loop.

    Args:
        value: Optional comma-separated instrument ids.
        include_watch: Whether to include watch-list symbols when `value` is
            omitted.

    Returns:
        Ordered instrument ids without duplicates.
    """
    if value:
        raw = [item.strip().upper() for item in value.split(",") if item.strip()]
    else:
        raw = listed_universe(include_watch=include_watch)
    seen = set()
    symbols = []
    for item in raw:
        if item in seen:
            continue
        seen.add(item)
        symbols.append(item)
    return symbols


def _load_loop_state(path: str | Path) -> dict[str, Any]:
    """Load live-loop cooldown state from disk.

    Args:
        path: JSON state path.

    Returns:
        Mutable state dictionary.
    """
    state_path = Path(path)
    if not state_path.exists():
        return {"last_order_ts_by_inst": {}}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"last_order_ts_by_inst": {}}


def _save_loop_state(path: str | Path, state: dict[str, Any]) -> None:
    """Persist live-loop cooldown state.

    Args:
        path: JSON state path.
        state: State dictionary to write.
    """
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _has_pending_order(orders: list[dict[str, Any]], algo_orders: list[dict[str, Any]], inst_id: str) -> bool:
    """Return whether an instrument already has open regular or algo orders.

    Args:
        orders: Regular OKX open orders.
        algo_orders: OKX algo/conditional orders.
        inst_id: Instrument id to check.

    Returns:
        True when the symbol already has pending exchange-side instructions.
    """
    return any(order.get("instId") == inst_id for order in orders) or any(order.get("instId") == inst_id for order in algo_orders)


def _cooldown_active(state: dict[str, Any], inst_id: str, cooldown_sec: int, now: int) -> bool:
    """Return whether an instrument is still inside order cooldown.

    Args:
        state: Live-loop state dictionary.
        inst_id: Instrument id.
        cooldown_sec: Cooldown seconds after a sent order.
        now: Current epoch seconds.

    Returns:
        True if the last order timestamp is too recent.
    """
    last = int(state.get("last_order_ts_by_inst", {}).get(inst_id, 0))
    return last > 0 and now - last < cooldown_sec


def _count_open_positions(positions: list[dict[str, Any]]) -> int:
    """Count non-zero OKX positions.

    Args:
        positions: OKX position rows.

    Returns:
        Number of non-zero positions.
    """
    count = 0
    for position in positions:
        try:
            if abs(float(position.get("pos", 0) or 0)) > 0:
                count += 1
        except (TypeError, ValueError):
            continue
    return count


def _cloudflared_path() -> str | None:
    """Return a usable cloudflared executable path if available.

    Returns:
        Absolute default path or PATH-discovered executable, otherwise None.
    """
    if Path(DEFAULT_CLOUDFLARED).exists():
        return DEFAULT_CLOUDFLARED
    return shutil.which("cloudflared")


def _start_tunnel() -> subprocess.Popen | None:
    """Start Cloudflare Tunnel and mirror its output into this terminal.

    Returns:
        The cloudflared process, or None if cloudflared is unavailable.
    """
    executable = _cloudflared_path()
    if executable is None:
        print("cloudflared not found; phone tunnel is disabled.", flush=True)
        return None
    process = subprocess.Popen(
        [executable, "tunnel", "--protocol", "http2", "--url", f"http://{config.WEB_HOST}:{config.WEB_PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    def pump_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[tunnel] {line.rstrip()}", flush=True)

    threading.Thread(target=pump_output, daemon=True).start()
    return process


def run_one_cycle(args: argparse.Namespace, client: OKXRestClient, state: dict[str, Any], symbols: list[str], meta_model: Any, regime_model: Any) -> dict[str, Any]:
    """Scan all symbols once and place at most the configured number of orders.

    Args:
        args: Parsed command-line arguments.
        client: Authenticated OKX client.
        state: Mutable live-loop cooldown state.
        symbols: Symbols to scan.
        meta_model: Optional loaded Meta model.
        regime_model: Optional loaded Regime model.

    Returns:
        Cycle summary used for logging and console output.
    """
    now = int(time.time())
    balance = client.get_balance("USDT")
    positions = client.get_positions()
    orders = client.get_open_orders()
    algo_orders = client.get_algo_orders()
    account_equity = _usdt_equity(balance)
    equity = args.equity_usdt if args.equity_usdt is not None else account_equity
    limits = LiveRiskLimits(
        max_risk_usdt=args.max_risk_usdt,
        max_margin_usdt=args.max_margin_usdt,
        max_notional_usdt=args.max_notional_usdt,
        max_leverage=args.max_leverage,
    )
    cycle = {
        "ts": now,
        "event": "live_loop_cycle",
        "dry_run": args.dry_run,
        "symbols": symbols,
        "account_equity": account_equity,
        "open_positions": _count_open_positions(positions),
        "placed": [],
        "blocked": [],
        "errors": [],
    }
    if cycle["open_positions"] >= args.max_open_positions:
        cycle["blocked"].append({"inst_id": "*", "reason": "max_open_positions_reached"})
        return cycle

    placed_count = 0
    for inst_id in symbols:
        if placed_count >= args.max_orders_per_cycle:
            break
        try:
            if _has_open_position(positions, inst_id):
                cycle["blocked"].append({"inst_id": inst_id, "reason": "existing_position"})
                continue
            if _has_pending_order(orders, algo_orders, inst_id):
                cycle["blocked"].append({"inst_id": inst_id, "reason": "existing_order_or_algo"})
                continue
            if _cooldown_active(state, inst_id, args.cooldown_minutes * 60, now):
                cycle["blocked"].append({"inst_id": inst_id, "reason": "cooldown"})
                continue

            raw = _candles_to_frame(client.get_candles(inst_id, args.bar, limit=args.candle_limit))
            decision = build_live_signal(raw, inst_id, equity_usdt=equity, meta_model=meta_model, regime_model=regime_model, use_agent_market=True)
            base_row = {
                "inst_id": inst_id,
                "decision": decision,
                "last_price": float(raw["close"].iloc[-1]),
            }
            if not decision.allowed or decision.plan is None:
                cycle["blocked"].append({**base_row, "reason": decision.reason})
                continue

            instrument = client.get_instrument(inst_id)
            sizing = plan_to_order_sizing(decision.plan, inst_id, instrument)
            validate_live_risk(decision.plan, sizing, limits)
            prepared = {**base_row, "order_sizing": sizing, "limits": limits}

            if args.dry_run:
                cycle["placed"].append({**prepared, "event": "dry_run_prepared"})
                placed_count += 1
                continue

            client.set_leverage(inst_id, decision.plan.leverage, td_mode=args.td_mode)
            client_order_id = f"okxloop{now}{placed_count}"
            response = client.place_market_order_with_protection(
                inst_id=inst_id,
                td_mode=args.td_mode,
                side=sizing.side,
                size=sizing.size,
                take_profit=decision.plan.take_profit,
                stop_loss=decision.plan.stop,
                client_order_id=client_order_id,
            )
            state.setdefault("last_order_ts_by_inst", {})[inst_id] = now
            cycle["placed"].append({**prepared, "event": "live_order_sent", "client_order_id": client_order_id, "order_response": response})
            placed_count += 1
        except Exception as exc:
            cycle["errors"].append({"inst_id": inst_id, "error": str(exc)})
    return cycle


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously scan symbols and place guarded OKX live orders")
    parser.add_argument("--symbols", default=None, help="Comma-separated instruments. Defaults to core + liquid alt universe")
    parser.add_argument("--include-watch", action="store_true")
    parser.add_argument("--bar", default=config.BAR)
    parser.add_argument("--td-mode", choices=["isolated", "cross"], default="isolated")
    parser.add_argument("--interval-sec", type=int, default=config.LIVE_LOOP_INTERVAL_SEC)
    parser.add_argument("--candle-limit", type=int, default=300)
    parser.add_argument("--max-orders-per-cycle", type=int, default=config.LIVE_LOOP_MAX_ORDERS_PER_CYCLE)
    parser.add_argument("--max-open-positions", type=int, default=config.LIVE_LOOP_MAX_OPEN_POSITIONS)
    parser.add_argument("--cooldown-minutes", type=int, default=config.LIVE_LOOP_COOLDOWN_MINUTES)
    parser.add_argument("--meta-model", default=DEFAULT_META_MODEL)
    parser.add_argument("--regime-model", default=None)
    parser.add_argument("--equity-usdt", type=float, default=None)
    parser.add_argument("--max-risk-usdt", type=float, default=config.LIVE_DEFAULT_MAX_RISK_USDT)
    parser.add_argument("--max-margin-usdt", type=float, default=config.LIVE_DEFAULT_MAX_MARGIN_USDT)
    parser.add_argument("--max-notional-usdt", type=float, default=config.LIVE_DEFAULT_MAX_NOTIONAL_USDT)
    parser.add_argument("--max-leverage", type=int, default=config.LIVE_DEFAULT_MAX_LEVERAGE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--log", default="logs/live_orders.jsonl")
    parser.add_argument("--state", default="logs/live_loop_state.json")
    parser.add_argument("--dashboard", action="store_true", help="Start the local dashboard in this same Python process")
    parser.add_argument("--tunnel", action="store_true", help="Start Cloudflare Tunnel in this same terminal")
    parser.add_argument("--i-understand-live-risk", action="store_true")
    args = parser.parse_args()

    credentials = load_credentials(args.env)
    if not args.dry_run and not args.i_understand_live_risk:
        raise RuntimeError("real live loop orders require --i-understand-live-risk")
    if not args.dry_run and credentials.simulated:
        raise RuntimeError("OKX_FLAG=1 is simulated trading; set OKX_FLAG=0 only when you intentionally want live trading")

    client = OKXRestClient(credentials)
    symbols = _symbols_from_args(args.symbols, include_watch=args.include_watch)
    meta_model = load_meta_model(args.meta_model) if args.meta_model else None
    regime_model = load_regime_model(args.regime_model) if args.regime_model else None
    state = _load_loop_state(args.state)

    if args.dashboard:
        thread = threading.Thread(target=run_dashboard, kwargs={"handle_signals": False}, daemon=True)
        thread.start()
        print(f"Dashboard running at http://{config.WEB_HOST}:{config.WEB_PORT}", flush=True)
    tunnel_process = _start_tunnel() if args.tunnel else None

    print(f"Live loop scanning {len(symbols)} symbols every {args.interval_sec}s. dry_run={args.dry_run}", flush=True)
    try:
        while True:
            try:
                cycle = run_one_cycle(args, client, state, symbols, meta_model, regime_model)
            except Exception as exc:
                cycle = {
                    "ts": int(time.time()),
                    "event": "live_loop_error",
                    "dry_run": args.dry_run,
                    "symbols": symbols,
                    "error": str(exc),
                }
            finally:
                _save_loop_state(args.state, state)
            _append_log(args.log, cycle)
            print(json.dumps(_json_safe(cycle), ensure_ascii=False), flush=True)
            time.sleep(max(args.interval_sec, 5))
    finally:
        if tunnel_process is not None and tunnel_process.poll() is None:
            tunnel_process.terminate()


if __name__ == "__main__":
    main()
