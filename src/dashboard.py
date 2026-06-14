"""Local dashboard for strategy reports, live decisions, and startup status.

The dashboard is read-only. It shows local backtest reports, recent live-order
logs, and the most important runtime settings without exposing API secrets.

本地看板只读展示：回测报告、实盘执行日志和关键配置；不会展示 API 密钥。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from aiohttp import web

from src import config
from src.data import OKX_COLUMNS
from src.live_signal import build_live_signal
from src.meta_model import load_meta_model
from src.okx_client import OKXRestClient, load_credentials
from src.regime_model import load_regime_model
from src.universe_engine import listed_universe


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT_DIR / "backtest" / "reports"
LOG_DIR = ROOT_DIR / "logs"
LIVE_LOG_PATH = LOG_DIR / "live_orders.jsonl"
DEFAULT_META_MODEL = ROOT_DIR / "backtest" / "models" / "meta_model_live.json"


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON report and return an empty dict on parse failure.

    Args:
        path: Local JSON file path.

    Returns:
        Parsed dictionary, or a small error object if the file is invalid.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive dashboard path
        return {"error": str(exc), "path": str(path)}


def _compact_report(path: Path) -> dict[str, Any]:
    """Return the headline fields used by the dashboard report table.

    Args:
        path: Report JSON path under `backtest/reports`.

    Returns:
        A compact report summary safe for browser rendering.
    """
    data = _read_json(path)
    return {
        "name": path.name,
        "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "mode": data.get("mode"),
        "tested": data.get("tested"),
        "return_pct": data.get("walk_forward_1000u_return_pct", data.get("equal_weight_1000u_return_pct", data.get("return_pct"))),
        "pnl": data.get("walk_forward_1000u_pnl", data.get("equal_weight_1000u_pnl", data.get("pnl"))),
        "trades": data.get("total_trades", data.get("trades")),
        "drawdown_pct": data.get("max_single_asset_drawdown_pct", data.get("max_drawdown_pct")),
        "market_gate": data.get("market_gate"),
    }


def list_reports(limit: int = 24) -> list[dict[str, Any]]:
    """List recent JSON reports sorted newest first.

    Args:
        limit: Maximum number of reports returned.

    Returns:
        Compact report summaries.
    """
    if not REPORT_DIR.exists():
        return []
    paths = sorted(REPORT_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    return [_compact_report(path) for path in paths[:limit]]


def read_live_events(limit: int = 40) -> list[dict[str, Any]]:
    """Read recent live execution JSONL events.

    Args:
        limit: Maximum number of events returned.

    Returns:
        Newest-first live execution events.
    """
    if not LIVE_LOG_PATH.exists():
        return []
    lines = LIVE_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]
    events = []
    for line in reversed(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"event": "invalid_log_line", "raw": line[:240]})
    return events


def live_replay(limit: int = 80) -> dict[str, Any]:
    """Parse live-loop JSONL logs into dashboard replay rows.

    Args:
        limit: Maximum number of raw log events to inspect from the tail.

    Returns:
        Aggregated replay data with cycles, placed rows, blocked rows, and
        symbol-level summaries.
    """
    events = read_live_events(limit=limit)
    cycles = [event for event in events if event.get("event") in ("live_loop_cycle", "live_loop_error")]
    latest_cycle = cycles[0] if cycles else None
    placed_rows = []
    blocked_rows = []
    by_symbol: dict[str, dict[str, Any]] = {}

    for cycle in cycles:
        cycle_ts = cycle.get("ts")
        for row in cycle.get("placed", []) or []:
            inst_id = row.get("inst_id", "")
            decision = row.get("decision") or {}
            plan = decision.get("plan") or {}
            item = {
                "ts": cycle_ts,
                "inst_id": inst_id,
                "event": row.get("event", "placed"),
                "direction": plan.get("direction"),
                "agent": decision.get("agent"),
                "entry": plan.get("entry"),
                "stop": plan.get("stop"),
                "take_profit": plan.get("take_profit"),
                "qty": plan.get("qty"),
                "margin": plan.get("margin"),
                "risk_usdt": plan.get("risk_usdt"),
                "meta_probability": decision.get("meta_probability"),
                "last_price": row.get("last_price"),
            }
            placed_rows.append(item)
            by_symbol[inst_id] = {**by_symbol.get(inst_id, {}), **item, "last_status": "placed"}
        for row in cycle.get("blocked", []) or []:
            inst_id = row.get("inst_id", "")
            decision = row.get("decision") or {}
            state = decision.get("state") or {}
            item = {
                "ts": cycle_ts,
                "inst_id": inst_id,
                "reason": row.get("reason"),
                "regime": state.get("regime"),
                "market_direction": state.get("direction"),
                "agent": decision.get("agent"),
                "last_price": row.get("last_price"),
            }
            blocked_rows.append(item)
            by_symbol[inst_id] = {**by_symbol.get(inst_id, {}), **item, "last_status": "blocked"}

    return {
        "cycles": cycles[:20],
        "latest_cycle": latest_cycle,
        "placed": placed_rows[:40],
        "blocked": blocked_rows[:120],
        "by_symbol": [value for _, value in sorted(by_symbol.items())],
    }


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe object for dataclasses and pandas timestamps.

    Args:
        value: Arbitrary object from strategy models or OKX responses.

    Returns:
        JSON-serializable representation.
    """
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
    """Convert OKX REST candles into the local OHLCV DataFrame format.

    Args:
        candles: Raw OKX candle rows.

    Returns:
        Chronological OHLCV DataFrame.
    """
    rows = [row[: len(OKX_COLUMNS)] for row in candles if len(row) >= len(OKX_COLUMNS)]
    df = pd.DataFrame(rows, columns=OKX_COLUMNS)
    if df.empty:
        raise ValueError("OKX returned no candles")
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
    for column in ("open", "high", "low", "close", "vol"):
        df[column] = df[column].astype(float)
    return df.sort_values("ts").reset_index(drop=True)


def _usdt_equity(balance: dict[str, Any]) -> dict[str, float]:
    """Extract USDT account fields from OKX balance response.

    Args:
        balance: OKX account balance row.

    Returns:
        Dashboard account summary with equity and available equity.
    """
    result = {"total_equity": 0.0, "usdt_equity": 0.0, "available_usdt": 0.0}
    for key in ("totalEq", "adjEq"):
        raw = balance.get(key)
        if raw not in (None, ""):
            result["total_equity"] = float(raw)
            break
    for item in balance.get("details") or []:
        if item.get("ccy") != "USDT":
            continue
        for source, target in (("eq", "usdt_equity"), ("availEq", "available_usdt")):
            raw = item.get(source)
            if raw not in (None, ""):
                result[target] = float(raw)
    if result["usdt_equity"] <= 0:
        result["usdt_equity"] = result["total_equity"]
    if result["available_usdt"] <= 0:
        result["available_usdt"] = result["usdt_equity"]
    return result


def _position_summary(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact non-zero position rows for the dashboard.

    Args:
        positions: OKX position rows.

    Returns:
        Compact position summaries.
    """
    rows = []
    for position in positions:
        try:
            size = float(position.get("pos", 0) or 0)
        except (TypeError, ValueError):
            size = 0.0
        if abs(size) <= 0:
            continue
        rows.append(
            {
                "inst_id": position.get("instId"),
                "side": position.get("posSide") or ("long" if size > 0 else "short"),
                "size": size,
                "avg_px": position.get("avgPx"),
                "mark_px": position.get("markPx"),
                "upl": position.get("upl"),
                "upl_ratio": position.get("uplRatio"),
                "liq_px": position.get("liqPx"),
                "lever": position.get("lever"),
                "margin": position.get("margin"),
            }
        )
    return rows


def _order_summary(orders: list[dict[str, Any]], algo_orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine regular and algo orders into compact dashboard rows.

    Args:
        orders: Regular open orders from OKX.
        algo_orders: Open TP/SL or conditional algo orders from OKX.

    Returns:
        Compact order rows.
    """
    regular = [
        {
            "type": "order",
            "inst_id": order.get("instId"),
            "side": order.get("side"),
            "ord_type": order.get("ordType"),
            "price": order.get("px"),
            "size": order.get("sz"),
            "state": order.get("state"),
        }
        for order in orders
    ]
    algos = [
        {
            "type": "algo",
            "inst_id": order.get("instId"),
            "side": order.get("side"),
            "ord_type": order.get("ordType"),
            "trigger": order.get("triggerPx") or order.get("tpTriggerPx") or order.get("slTriggerPx"),
            "tp": order.get("tpTriggerPx"),
            "sl": order.get("slTriggerPx"),
            "size": order.get("sz"),
            "state": order.get("state"),
        }
        for order in algo_orders
    ]
    return regular + algos


def _signal_summary(decision: Any, inst_id: str, last_price: float) -> dict[str, Any]:
    """Convert a live signal decision into one dashboard card row.

    Args:
        decision: LiveSignalDecision produced by `build_live_signal`.
        inst_id: OKX instrument id.
        last_price: Latest close price.

    Returns:
        Compact signal row.
    """
    plan = decision.plan
    state = decision.state
    return {
        "inst_id": inst_id,
        "last_price": round(last_price, 8),
        "allowed": decision.allowed,
        "reason": decision.reason,
        "agent": decision.agent,
        "thesis": decision.thesis,
        "direction": plan.direction if plan else None,
        "style": plan.style if plan else (decision.setup.style if decision.setup else None),
        "entry": plan.entry if plan else None,
        "stop": plan.stop if plan else None,
        "take_profit": plan.take_profit if plan else None,
        "partial_take_profit": plan.partial_take_profit if plan else None,
        "qty": plan.qty if plan else None,
        "margin": plan.margin if plan else None,
        "risk_usdt": plan.risk_usdt if plan else None,
        "leverage": plan.leverage if plan else None,
        "meta_probability": round(decision.meta_probability, 4) if decision.meta_probability is not None else None,
        "model_multiplier": round(decision.model_multiplier, 4),
        "regime": state.regime if state else None,
        "market_direction": state.direction if state else None,
        "risk_score": round(state.risk_score, 4) if state else None,
        "confidence": round(state.confidence, 4) if state else None,
    }


def live_market_snapshot(candle_limit: int = 300) -> dict[str, Any]:
    """Build an account, orders, positions, and per-symbol signal snapshot.

    Args:
        candle_limit: Number of recent OKX candles used for each signal.

    Returns:
        Dashboard-safe live market snapshot. Errors are captured in the payload
        so the page can still render.
    """
    try:
        credentials = load_credentials(ROOT_DIR / ".env")
        client = OKXRestClient(credentials, timeout=6)
        balance = client.get_balance("USDT")
        positions = client.get_positions()
        orders = client.get_open_orders()
        algo_orders = client.get_algo_orders()
        account = _usdt_equity(balance)
        equity = account["usdt_equity"] or config.INITIAL_EQUITY
        meta_model = load_meta_model(DEFAULT_META_MODEL) if DEFAULT_META_MODEL.exists() else None
        regime_model = load_regime_model(None)
        signals = []
        for inst_id in listed_universe(include_watch=False):
            try:
                raw = _candles_to_frame(client.get_candles(inst_id, config.BAR, limit=candle_limit))
                decision = build_live_signal(raw, inst_id, equity_usdt=equity, meta_model=meta_model, regime_model=regime_model, use_agent_market=True)
                signals.append(_signal_summary(decision, inst_id, float(raw["close"].iloc[-1])))
            except Exception as exc:
                signals.append({"inst_id": inst_id, "allowed": False, "reason": f"signal_error: {exc}"})
        return {
            "ok": True,
            "simulated": credentials.simulated,
            "account": account,
            "positions": _position_summary(positions),
            "orders": _order_summary(orders, algo_orders),
            "signals": signals,
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "simulated": None,
            "account": {"total_equity": 0.0, "usdt_equity": 0.0, "available_usdt": 0.0},
            "positions": [],
            "orders": [],
            "signals": [],
            "error": str(exc),
        }


def config_snapshot() -> dict[str, Any]:
    """Return dashboard-safe runtime settings.

    Returns:
        Public configuration values; secrets from `.env` are intentionally not
        included.
    """
    return {
        "default_inst_id": config.INST_ID,
        "bar": config.BAR,
        "web": f"{config.WEB_HOST}:{config.WEB_PORT}",
        "core_universe": config.CORE_UNIVERSE,
        "liquid_alt_universe": config.LIQUID_ALT_UNIVERSE,
        "watch_universe": config.WATCH_UNIVERSE,
        "initial_equity": config.INITIAL_EQUITY,
        "risk_per_trade": config.RISK_PER_TRADE,
        "max_position_margin_ratio": config.MAX_POSITION_MARGIN_RATIO,
        "live_max_risk_usdt": config.LIVE_DEFAULT_MAX_RISK_USDT,
        "live_max_margin_usdt": config.LIVE_DEFAULT_MAX_MARGIN_USDT,
        "live_max_notional_usdt": config.LIVE_DEFAULT_MAX_NOTIONAL_USDT,
        "live_max_leverage": config.LIVE_DEFAULT_MAX_LEVERAGE,
        "meta_model": "meta_model_live.json",
    }


def dashboard_state() -> dict[str, Any]:
    """Build the full API payload consumed by the browser.

    Returns:
        Combined report, log, config, and process status snapshot.
    """
    reports = list_reports()
    live_events = read_live_events()
    return {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cwd": str(ROOT_DIR),
        "python_env": os.environ.get("VIRTUAL_ENV", ""),
        "reports": reports,
        "latest_report": reports[0] if reports else None,
        "live_events": live_events,
        "replay": live_replay(),
        "latest_live_event": live_events[0] if live_events else None,
        "config": config_snapshot(),
    }


async def handle_index(_: web.Request) -> web.Response:
    """Serve the dashboard HTML shell."""
    return web.Response(text=DASHBOARD_HTML, content_type="text/html")


async def handle_state(_: web.Request) -> web.Response:
    """Serve the JSON state snapshot."""
    return web.json_response(dashboard_state())


async def handle_live(_: web.Request) -> web.Response:
    """Serve the read-only OKX account and live-signal snapshot."""
    return web.json_response(_json_safe(live_market_snapshot()))


async def handle_report(request: web.Request) -> web.Response:
    """Serve one full JSON report selected by filename.

    Args:
        request: aiohttp request with a safe `name` query parameter.
    """
    name = Path(request.query.get("name", "")).name
    path = REPORT_DIR / name
    if not name or not path.exists() or path.parent.resolve() != REPORT_DIR.resolve():
        raise web.HTTPNotFound(text="report not found")
    return web.json_response(_read_json(path))


def create_app() -> web.Application:
    """Create the aiohttp dashboard application.

    Returns:
        Configured aiohttp application.
    """
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/state", handle_state)
    app.router.add_get("/api/live", handle_live)
    app.router.add_get("/api/report", handle_report)
    return app


def run_dashboard(host: str | None = None, port: int | None = None, handle_signals: bool = True) -> None:
    """Run the dashboard server until interrupted.

    Args:
        host: Optional bind host; defaults to `config.WEB_HOST`.
        port: Optional bind port; defaults to `config.WEB_PORT`.
        handle_signals: Whether aiohttp should install signal handlers. Use
            False when running inside a background thread.
    """
    web.run_app(create_app(), host=host or config.WEB_HOST, port=port or config.WEB_PORT, handle_signals=handle_signals)


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OKX New Dashboard</title>
  <style>
    :root {
      --bg: #101114;
      --panel: #181b20;
      --line: #2b3139;
      --text: #f2f5f8;
      --muted: #aab2bd;
      --green: #21b26b;
      --red: #e05a5a;
      --cyan: #4fb7d8;
      --amber: #d9a441;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
      letter-spacing: 0;
    }
    header {
      border-bottom: 1px solid var(--line);
      padding: 18px clamp(16px, 3vw, 34px);
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      background: #14171b;
    }
    h1 { margin: 0; font-size: 24px; font-weight: 700; }
    .sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
    .pill {
      border: 1px solid var(--line);
      color: var(--muted);
      padding: 7px 10px;
      border-radius: 6px;
      font-size: 13px;
      white-space: nowrap;
    }
    main { padding: 22px clamp(14px, 3vw, 34px) 34px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 15px;
      min-width: 0;
    }
    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; }
    .value { font-size: 28px; font-weight: 700; margin-top: 7px; overflow-wrap: anywhere; }
    .small { color: var(--muted); font-size: 12px; margin-top: 5px; }
    .good { color: var(--green); }
    .bad { color: var(--red); }
    .cyan { color: var(--cyan); }
    .amber { color: var(--amber); }
    .section {
      display: grid;
      grid-template-columns: 1.35fr .9fr;
      gap: 14px;
      align-items: start;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 16px;
      font-weight: 650;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }
    th { color: var(--muted); font-weight: 600; }
    .events { display: grid; gap: 9px; }
    .event {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #14171b;
    }
    .event-title { display: flex; justify-content: space-between; gap: 8px; }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: var(--muted);
      font-size: 12px;
      margin: 8px 0 0;
    }
    .settings {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      font-size: 13px;
    }
    .setting {
      border-bottom: 1px solid var(--line);
      padding: 8px 0;
      min-width: 0;
    }
    button {
      background: #20252c;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 11px;
      cursor: pointer;
    }
    button:hover { border-color: var(--cyan); }
    @media (max-width: 980px) {
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .section { grid-template-columns: 1fr; }
    }
    @media (max-width: 560px) {
      header { align-items: flex-start; flex-direction: column; }
      .grid { grid-template-columns: 1fr; }
      .settings { grid-template-columns: 1fr; }
      .value { font-size: 23px; }
      table { font-size: 12px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>OKX New 本地看板</h1>
      <div class="sub">Agent Market · LightGBM Meta · Live Guard</div>
    </div>
    <div class="pill" id="updated">loading</div>
  </header>
  <main>
    <section class="grid">
      <div class="panel"><div class="label">最新收益</div><div class="value" id="ret">--</div><div class="small" id="report-name">等待报告</div></div>
      <div class="panel"><div class="label">PnL</div><div class="value" id="pnl">--</div><div class="small">1000U research view</div></div>
      <div class="panel"><div class="label">交易数</div><div class="value cyan" id="trades">--</div><div class="small">total trades</div></div>
      <div class="panel"><div class="label">最大回撤</div><div class="value amber" id="dd">--</div><div class="small">single-asset drawdown</div></div>
    </section>

    <section class="section">
      <div class="panel">
        <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:10px">
          <h2>最近回测报告</h2>
          <button onclick="loadState()">刷新</button>
        </div>
        <table>
          <thead><tr><th>文件</th><th>收益</th><th>PnL</th><th>交易</th><th>回撤</th></tr></thead>
          <tbody id="reports"></tbody>
        </table>
      </div>
      <div class="panel">
        <h2>关键配置</h2>
        <div class="settings" id="settings"></div>
      </div>
    </section>

    <section class="section" style="margin-top:14px">
      <div class="panel">
        <h2>实盘执行日志</h2>
        <div class="events" id="events"></div>
      </div>
      <div class="panel">
        <h2>系统状态</h2>
        <pre id="system">loading</pre>
      </div>
    </section>
  </main>
  <script>
    const pct = value => value === null || value === undefined ? "--" : `${Number(value).toFixed(2)}%`;
    const money = value => value === null || value === undefined ? "--" : Number(value).toFixed(4);
    function classFor(value) {
      const n = Number(value || 0);
      return n >= 0 ? "good" : "bad";
    }
    async function loadState() {
      const response = await fetch("/api/state");
      const state = await response.json();
      document.getElementById("updated").textContent = `更新 ${state.updated_at}`;
      const latest = state.latest_report || {};
      const ret = document.getElementById("ret");
      ret.textContent = pct(latest.return_pct);
      ret.className = `value ${classFor(latest.return_pct)}`;
      document.getElementById("pnl").textContent = money(latest.pnl);
      document.getElementById("trades").textContent = latest.trades ?? "--";
      document.getElementById("dd").textContent = pct(latest.drawdown_pct);
      document.getElementById("report-name").textContent = latest.name || "暂无报告";

      document.getElementById("reports").innerHTML = (state.reports || []).map(row => `
        <tr>
          <td>${row.name}<div class="small">${row.modified}</div></td>
          <td class="${classFor(row.return_pct)}">${pct(row.return_pct)}</td>
          <td>${money(row.pnl)}</td>
          <td>${row.trades ?? "--"}</td>
          <td>${pct(row.drawdown_pct)}</td>
        </tr>`).join("") || `<tr><td colspan="5">暂无报告</td></tr>`;

      document.getElementById("settings").innerHTML = Object.entries(state.config || {}).map(([key, value]) => `
        <div class="setting"><div class="label">${key}</div><div>${Array.isArray(value) ? value.join(", ") : value}</div></div>
      `).join("");

      document.getElementById("events").innerHTML = (state.live_events || []).slice(0, 10).map(event => `
        <div class="event">
          <div class="event-title"><strong>${event.event || "event"}</strong><span class="small">${event.inst_id || ""}</span></div>
          <pre>${JSON.stringify(event, null, 2)}</pre>
        </div>
      `).join("") || `<div class="small">暂无实盘日志，运行 run_live_okx 后会出现。</div>`;

      document.getElementById("system").textContent = JSON.stringify({
        cwd: state.cwd,
        python_env: state.python_env,
        latest_live_event: state.latest_live_event && state.latest_live_event.event
      }, null, 2);
    }
    loadState();
    setInterval(loadState, 5000);
  </script>
</body>
</html>
"""


DASHBOARD_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OKX New Live Board</title>
  <style>
    :root {
      --bg: #0f1115;
      --panel: #171b21;
      --panel2: #1d222a;
      --line: #2c3440;
      --text: #f3f6f9;
      --muted: #9ea8b5;
      --green: #20b26b;
      --red: #e35d5d;
      --cyan: #48b8d8;
      --amber: #d6a23f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      letter-spacing: 0;
    }
    header {
      background: #141820;
      border-bottom: 1px solid var(--line);
      padding: 16px clamp(14px, 3vw, 32px);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    h1 { margin: 0; font-size: 23px; }
    h2 { margin: 0 0 12px; font-size: 16px; }
    .sub { color: var(--muted); margin-top: 4px; font-size: 13px; }
    .pill {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 7px 10px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    main { padding: 18px clamp(12px, 3vw, 32px) 32px; }
    .top {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }
    .value {
      margin-top: 7px;
      font-size: 25px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .small { color: var(--muted); font-size: 12px; margin-top: 5px; }
    .green { color: var(--green); }
    .red { color: var(--red); }
    .cyan { color: var(--cyan); }
    .amber { color: var(--amber); }
    .section {
      display: grid;
      grid-template-columns: 1.45fr .9fr;
      gap: 12px;
      align-items: start;
      margin-top: 12px;
    }
    .signals {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .signal {
      background: var(--panel2);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }
    .signal.allowed { border-color: rgba(32, 178, 107, .8); }
    .signal.blocked { border-color: rgba(227, 93, 93, .55); }
    .row {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      border-bottom: 1px solid rgba(255,255,255,.06);
      padding: 7px 0;
      font-size: 13px;
    }
    .row span:first-child { color: var(--muted); }
    .row span:last-child { text-align: right; overflow-wrap: anywhere; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 9px 7px;
      vertical-align: top;
    }
    th { color: var(--muted); font-weight: 600; }
    .stack { display: grid; gap: 10px; }
    .event {
      background: var(--panel2);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }
    pre {
      color: var(--muted);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 12px;
      margin: 7px 0 0;
    }
    button {
      background: #222832;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 11px;
      cursor: pointer;
    }
    button:hover { border-color: var(--cyan); }
    @media (max-width: 1100px) {
      .top { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .signals { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .section { grid-template-columns: 1fr; }
    }
    @media (max-width: 620px) {
      header { flex-direction: column; align-items: flex-start; }
      .top { grid-template-columns: 1fr; }
      .signals { grid-template-columns: 1fr; }
      .value { font-size: 22px; }
      table { font-size: 12px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>OKX New 实盘看板</h1>
      <div class="sub">账户 · 持仓 · 挂单 · 多币种 Agent 信号 · 止盈止损计划</div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <button onclick="refreshAll()">刷新</button>
      <div class="pill" id="updated">loading</div>
    </div>
  </header>

  <main>
    <section class="top">
      <div class="panel"><div class="label">账户模式</div><div class="value" id="mode">--</div><div class="small" id="live-error"></div></div>
      <div class="panel"><div class="label">总权益 USDT</div><div class="value cyan" id="total-eq">--</div><div class="small">OKX total equity</div></div>
      <div class="panel"><div class="label">可用 USDT</div><div class="value" id="avail-eq">--</div><div class="small">available equity</div></div>
      <div class="panel"><div class="label">持仓数量</div><div class="value amber" id="pos-count">--</div><div class="small">non-zero positions</div></div>
      <div class="panel"><div class="label">挂单/止损止盈</div><div class="value" id="order-count">--</div><div class="small">regular + algo orders</div></div>
    </section>

    <section class="panel">
      <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:10px">
        <h2>各币当前信号</h2>
        <div class="small">每 30 秒刷新；绿色为当前允许入场，红色为被过滤</div>
      </div>
      <div class="signals" id="signals"></div>
    </section>

    <section class="section">
      <div class="panel">
        <h2>持仓</h2>
        <table>
          <thead><tr><th>币种</th><th>方向</th><th>张数</th><th>均价</th><th>标记价</th><th>浮盈亏</th><th>强平价</th></tr></thead>
          <tbody id="positions"></tbody>
        </table>
      </div>
      <div class="panel">
        <h2>挂单 / 止盈止损</h2>
        <table>
          <thead><tr><th>类型</th><th>币种</th><th>方向</th><th>价格/触发</th><th>TP</th><th>SL</th><th>数量</th></tr></thead>
          <tbody id="orders"></tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <div class="panel">
        <h2>自动循环回放</h2>
        <table>
          <thead><tr><th>时间</th><th>账户</th><th>持仓</th><th>准备/下单</th><th>过滤</th><th>错误</th></tr></thead>
          <tbody id="cycles"></tbody>
        </table>
      </div>
      <div class="panel">
        <h2>各币最近状态</h2>
        <table>
          <thead><tr><th>币种</th><th>状态</th><th>原因</th><th>市场</th><th>价格</th></tr></thead>
          <tbody id="symbol-replay"></tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <div class="panel">
        <h2>计划/下单记录</h2>
        <table>
          <thead><tr><th>时间</th><th>币种</th><th>方向</th><th>Agent</th><th>入场</th><th>止损</th><th>止盈</th><th>保证金</th></tr></thead>
          <tbody id="placed-replay"></tbody>
        </table>
      </div>
      <div class="panel">
        <h2>过滤回放</h2>
        <table>
          <thead><tr><th>时间</th><th>币种</th><th>原因</th><th>市场</th><th>方向</th></tr></thead>
          <tbody id="blocked-replay"></tbody>
        </table>
      </div>
    </section>
  </main>

  <script>
    const num = value => value === null || value === undefined || value === "" ? "--" : Number(value).toFixed(4);
    const pct = value => value === null || value === undefined ? "--" : `${Number(value).toFixed(2)}%`;
    const cls = value => Number(value || 0) >= 0 ? "green" : "red";
    const text = value => value === null || value === undefined || value === "" ? "--" : value;

    function signalCard(row) {
      const sideClass = row.allowed ? "allowed" : "blocked";
      const badgeClass = row.allowed ? "green" : "red";
      return `<div class="signal ${sideClass}">
        <div style="display:flex;justify-content:space-between;gap:8px;margin-bottom:8px">
          <strong>${row.inst_id}</strong>
          <span class="${badgeClass}">${row.allowed ? "可入场" : "过滤"}</span>
        </div>
        <div class="row"><span>最新价</span><span>${num(row.last_price)}</span></div>
        <div class="row"><span>方向</span><span>${text(row.direction)}</span></div>
        <div class="row"><span>Agent</span><span>${text(row.agent)}</span></div>
        <div class="row"><span>原因</span><span>${text(row.reason)}</span></div>
        <div class="row"><span>入场</span><span>${num(row.entry)}</span></div>
        <div class="row"><span>止损</span><span class="red">${num(row.stop)}</span></div>
        <div class="row"><span>止盈</span><span class="green">${num(row.take_profit)}</span></div>
        <div class="row"><span>分批止盈</span><span>${num(row.partial_take_profit)}</span></div>
        <div class="row"><span>数量/保证金</span><span>${text(row.qty)} / ${text(row.margin)}</span></div>
        <div class="row"><span>风险/杠杆</span><span>${text(row.risk_usdt)} / ${text(row.leverage)}x</span></div>
        <div class="row"><span>Meta概率</span><span>${row.meta_probability ?? "--"}</span></div>
        <div class="row"><span>市场状态</span><span>${text(row.regime)} · ${text(row.market_direction)}</span></div>
      </div>`;
    }

    async function loadLive() {
      const response = await fetch("/api/live");
      const live = await response.json();
      document.getElementById("mode").textContent = live.simulated === true ? "模拟盘" : live.simulated === false ? "实盘" : "--";
      document.getElementById("live-error").textContent = live.error || "";
      document.getElementById("total-eq").textContent = num(live.account && live.account.total_equity);
      document.getElementById("avail-eq").textContent = num(live.account && live.account.available_usdt);
      document.getElementById("pos-count").textContent = (live.positions || []).length;
      document.getElementById("order-count").textContent = (live.orders || []).length;
      document.getElementById("signals").innerHTML = (live.signals || []).map(signalCard).join("") || `<div class="small">暂无信号。请检查 API 或网络。</div>`;
      document.getElementById("positions").innerHTML = (live.positions || []).map(row => `
        <tr><td>${text(row.inst_id)}</td><td>${text(row.side)}</td><td>${text(row.size)}</td><td>${text(row.avg_px)}</td><td>${text(row.mark_px)}</td><td class="${cls(row.upl)}">${text(row.upl)}</td><td>${text(row.liq_px)}</td></tr>
      `).join("") || `<tr><td colspan="7">暂无持仓</td></tr>`;
      document.getElementById("orders").innerHTML = (live.orders || []).map(row => `
        <tr><td>${text(row.type)} / ${text(row.ord_type)}</td><td>${text(row.inst_id)}</td><td>${text(row.side)}</td><td>${text(row.price || row.trigger)}</td><td>${text(row.tp)}</td><td>${text(row.sl)}</td><td>${text(row.size)}</td></tr>
      `).join("") || `<tr><td colspan="7">暂无挂单或保护单</td></tr>`;
    }

    async function loadState() {
      const response = await fetch("/api/state");
      const state = await response.json();
      document.getElementById("updated").textContent = `更新 ${state.updated_at}`;
      const replay = state.replay || {};
      document.getElementById("cycles").innerHTML = (replay.cycles || []).slice(0, 12).map(row => `
        <tr>
          <td>${row.ts ? new Date(row.ts * 1000).toLocaleTimeString() : "--"}</td>
          <td>${num(row.account_equity)}</td>
          <td>${text(row.open_positions)}</td>
          <td class="green">${(row.placed || []).length}</td>
          <td>${(row.blocked || []).length}</td>
          <td class="red">${row.error || ((row.errors || []).length)}</td>
        </tr>
      `).join("") || `<tr><td colspan="6">暂无循环日志</td></tr>`;
      document.getElementById("symbol-replay").innerHTML = (replay.by_symbol || []).map(row => `
        <tr>
          <td>${text(row.inst_id)}</td>
          <td class="${row.last_status === "placed" ? "green" : "red"}">${text(row.last_status)}</td>
          <td>${text(row.reason)}</td>
          <td>${text(row.regime)}</td>
          <td>${num(row.last_price)}</td>
        </tr>
      `).join("") || `<tr><td colspan="5">暂无币种回放</td></tr>`;
      document.getElementById("placed-replay").innerHTML = (replay.placed || []).map(row => `
        <tr>
          <td>${row.ts ? new Date(row.ts * 1000).toLocaleTimeString() : "--"}</td>
          <td>${text(row.inst_id)}</td>
          <td>${text(row.direction)}</td>
          <td>${text(row.agent)}</td>
          <td>${num(row.entry)}</td>
          <td class="red">${num(row.stop)}</td>
          <td class="green">${num(row.take_profit)}</td>
          <td>${num(row.margin)}</td>
        </tr>
      `).join("") || `<tr><td colspan="8">暂无计划或下单记录</td></tr>`;
      document.getElementById("blocked-replay").innerHTML = (replay.blocked || []).slice(0, 30).map(row => `
        <tr>
          <td>${row.ts ? new Date(row.ts * 1000).toLocaleTimeString() : "--"}</td>
          <td>${text(row.inst_id)}</td>
          <td>${text(row.reason)}</td>
          <td>${text(row.regime)}</td>
          <td>${text(row.market_direction)}</td>
        </tr>
      `).join("") || `<tr><td colspan="5">暂无过滤记录</td></tr>`;
    }

    async function refreshAll() {
      await Promise.all([loadState(), loadLive()]);
    }
    refreshAll();
    setInterval(loadState, 5000);
    setInterval(loadLive, 30000);
  </script>
</body>
</html>
"""
