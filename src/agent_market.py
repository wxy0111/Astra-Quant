"""Local agent signal market for strategy proposal and scoring."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from src import config
from src.market_state import MarketState
from src.trade_plan import TradeSetup


@dataclass(frozen=True)
class AgentProposal:
    """One structured trade idea emitted by a local strategy agent."""

    agent: str
    setup: TradeSetup
    priority: float
    thesis: str


def _trend_pullback_agent(window: pd.DataFrame, state: MarketState) -> AgentProposal | None:
    if state.regime != "trend" or state.direction not in ("long", "short") or len(window) < 2:
        return None
    row = window.iloc[-1]
    prev = window.iloc[-2]
    close = float(row["close"])
    atr = float(row["atr"])
    if state.direction == "long":
        touched = float(row["low"]) <= float(row["ema_fast"]) + config.PULLBACK_ATR_BAND * atr
        recovered = close > float(row["ema_fast"])
        previous_near = float(prev["close"]) <= float(prev["ema_fast"]) + config.PULLBACK_ATR_BAND * float(prev["atr"])
        if touched and recovered and previous_near:
            setup = TradeSetup("long", "trend_pullback", close, min(float(row["low"]), float(row["ema_slow"])), state.confidence, state.reasons + ("agent_trend_pullback",))
            return AgentProposal("trend_pullback_agent", setup, 0.75, "trend pullback recovered above fast EMA")
    else:
        touched = float(row["high"]) >= float(row["ema_fast"]) - config.PULLBACK_ATR_BAND * atr
        recovered = close < float(row["ema_fast"])
        previous_near = float(prev["close"]) >= float(prev["ema_fast"]) - config.PULLBACK_ATR_BAND * float(prev["atr"])
        if touched and recovered and previous_near:
            setup = TradeSetup("short", "trend_pullback", close, max(float(row["high"]), float(row["ema_slow"])), state.confidence, state.reasons + ("agent_trend_pullback",))
            return AgentProposal("trend_pullback_agent", setup, 0.75, "trend pullback rejected at fast EMA")
    return None


def _trend_breakout_agent(window: pd.DataFrame, state: MarketState) -> AgentProposal | None:
    if state.regime != "trend" or state.direction not in ("long", "short"):
        return None
    row = window.iloc[-1]
    close = float(row["close"])
    atr = float(row["atr"])
    if state.direction == "long" and close > float(row["donchian_high"]) + config.BREAKOUT_BUFFER_ATR * atr:
        setup = TradeSetup("long", "trend_breakout", close, float(row["donchian_low"]), state.confidence, state.reasons + ("agent_trend_breakout",))
        return AgentProposal("trend_breakout_agent", setup, 0.65, "trend continuation breakout above Donchian high")
    if state.direction == "short" and close < float(row["donchian_low"]) - config.BREAKOUT_BUFFER_ATR * atr:
        setup = TradeSetup("short", "trend_breakout", close, float(row["donchian_high"]), state.confidence, state.reasons + ("agent_trend_breakout",))
        return AgentProposal("trend_breakout_agent", setup, 0.65, "trend continuation breakout below Donchian low")
    return None


def _range_reversion_agent(window: pd.DataFrame, state: MarketState) -> AgentProposal | None:
    if state.regime != "range":
        return None
    row = window.iloc[-1]
    close = float(row["close"])
    atr = float(row["atr"])
    if close < float(row["boll_lower"]):
        setup = TradeSetup("long", "range_reversion", close, close - config.STOP_ATR_MULT * atr, 0.55, state.reasons + ("agent_range_reversion",))
        return AgentProposal("range_reversion_agent", setup, 0.45, "range lower-band mean reversion")
    if close > float(row["boll_upper"]):
        setup = TradeSetup("short", "range_reversion", close, close + config.STOP_ATR_MULT * atr, 0.55, state.reasons + ("agent_range_reversion",))
        return AgentProposal("range_reversion_agent", setup, 0.45, "range upper-band mean reversion")
    return None


def _volatility_breakout_agent(window: pd.DataFrame, state: MarketState) -> AgentProposal | None:
    if len(window) < 30 or state.risk_score < 0.35:
        return None
    row = window.iloc[-1]
    close = float(row["close"])
    atr = float(row["atr"])
    body = close - float(row["open"])
    if abs(body) < atr * 0.8:
        return None
    direction = "long" if body > 0 else "short"
    stop = close - config.STOP_ATR_MULT * atr if direction == "long" else close + config.STOP_ATR_MULT * atr
    setup = TradeSetup(direction, "volatility_breakout", close, stop, min(0.72, state.confidence + 0.08), state.reasons + ("agent_volatility_breakout",))
    return AgentProposal("volatility_breakout_agent", setup, 0.35, "high-risk expansion candle continuation probe")


def generate_agent_setups(window: pd.DataFrame, state: MarketState) -> list[AgentProposal]:
    """Return all local agent proposals for the latest market window."""
    proposals = [
        agent(window, state)
        for agent in (
            _trend_pullback_agent,
            _trend_breakout_agent,
            _range_reversion_agent,
            _volatility_breakout_agent,
        )
    ]
    return sorted((proposal for proposal in proposals if proposal is not None), key=lambda item: item.priority, reverse=True)


def choose_agent_proposal(proposals: list[AgentProposal]) -> AgentProposal | None:
    """Select the highest-priority proposal for execution."""
    return proposals[0] if proposals else None


def allowed_agents_from_scoreboard(
    rows: list[dict],
    min_trades: int = 5,
    min_profit_factor: float = 1.0,
) -> set[str]:
    """Return agents with enough positive recent fit to keep trading."""
    allowed: set[str] = set()
    for row in rows:
        if int(row.get("trades", 0)) < min_trades:
            continue
        if float(row.get("pnl", 0.0)) <= 0:
            continue
        profit_factor = row.get("profit_factor")
        if profit_factor is None or float(profit_factor) < min_profit_factor:
            continue
        allowed.add(str(row.get("agent")))
    return allowed


def agent_weights_from_scoreboard(
    rows: list[dict],
    min_trades: int = 5,
) -> dict[str, float]:
    """Map recent per-agent fit to a conservative sizing multiplier."""
    weights: dict[str, float] = {}
    for row in rows:
        agent = str(row.get("agent"))
        trades = int(row.get("trades", 0))
        pnl = float(row.get("pnl", 0.0))
        win_rate = float(row.get("win_rate", 0.0))
        worst_trade = abs(float(row.get("worst_trade", 0.0)))
        profit_factor_raw = row.get("profit_factor")
        profit_factor = float(profit_factor_raw) if profit_factor_raw is not None else 0.0

        if trades < min_trades:
            weights[agent] = 0.25 if pnl > 0 else 0.0
            continue
        if pnl <= 0 or profit_factor < 0.90:
            weights[agent] = 0.0
            continue

        score = 0.0
        score += min(pnl / 30.0, 1.0) * 35.0
        score += min(max(profit_factor - 1.0, 0.0) / 1.0, 1.0) * 35.0
        score += min(max(win_rate - 0.40, 0.0) / 0.25, 1.0) * 20.0
        score -= min(worst_trade / 20.0, 1.0) * 10.0

        if score >= 70:
            weights[agent] = 1.25
        elif score >= 48:
            weights[agent] = 1.0
        elif score >= 15:
            weights[agent] = 0.75
        else:
            weights[agent] = 0.50
    return weights


def score_agent_results(trades: list[dict]) -> list[dict]:
    """Aggregate closed trade events by emitting agent."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for trade in trades:
        agent = trade.get("agent")
        if not agent or trade.get("event") == "pyramid_add":
            continue
        groups[str(agent)].append(trade)
    rows = []
    for agent, items in sorted(groups.items()):
        pnls = [float(item.get("pnl", 0.0)) for item in items]
        wins = [pnl for pnl in pnls if pnl > 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl <= 0))
        rows.append(
            {
                "agent": agent,
                "trades": len(items),
                "pnl": round(sum(pnls), 4),
                "win_rate": round(len(wins) / len(items), 4) if items else 0.0,
                "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else None,
                "avg_trade": round(sum(pnls) / len(items), 4) if items else 0.0,
                "best_trade": round(max(pnls), 4) if pnls else 0.0,
                "worst_trade": round(min(pnls), 4) if pnls else 0.0,
            }
        )
    return rows
