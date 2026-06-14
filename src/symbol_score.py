"""Rolling symbol-fit scoring for adaptive universe allocation."""

from __future__ import annotations

from dataclasses import dataclass

from src import config


@dataclass(frozen=True)
class SymbolScore:
    """A compact allocation verdict for one instrument."""

    score: float
    state: str
    weight: float
    reason: str


def _num(result: dict, key: str, default: float = 0.0) -> float:
    value = result.get(key, default)
    if value is None:
        return default
    return float(value)


def score_symbol_result(result: dict) -> SymbolScore:
    """Score recent strategy fit from backtest metrics.

    The score is intentionally conservative: large negative return, weak profit
    factor, or deep drawdown can pause a symbol even when win rate looks passable.
    """
    trades = int(_num(result, "trades"))
    return_pct = _num(result, "return_pct")
    profit_factor = _num(result, "profit_factor")
    win_rate = _num(result, "win_rate")
    drawdown_pct = _num(result, "max_drawdown_pct")

    if trades < config.MIN_SYMBOL_SCORE_TRADES:
        return SymbolScore(0.0, "paused", 0.0, "insufficient_trades")
    if drawdown_pct >= config.SYMBOL_PAUSE_DRAWDOWN_PCT:
        return SymbolScore(0.0, "paused", 0.0, "drawdown_too_high")
    if return_pct <= config.SYMBOL_PAUSE_RETURN_PCT and profit_factor <= config.SYMBOL_PAUSE_PROFIT_FACTOR:
        return SymbolScore(0.0, "paused", 0.0, "negative_recent_fit")

    score = 50.0
    score += return_pct * 1.6
    score += (profit_factor - 1.0) * 35.0
    score += (win_rate - 0.40) * 45.0
    score -= max(drawdown_pct - 10.0, 0.0) * 0.7
    score = round(max(0.0, min(score, 100.0)), 2)

    if score >= config.SYMBOL_ACTIVE_SCORE and profit_factor >= 1.0 and return_pct >= 0:
        return SymbolScore(score, "active", 1.0, "recent_fit_active")
    if score >= config.SYMBOL_REDUCED_SCORE and profit_factor >= 0.80:
        return SymbolScore(score, "reduced", config.SYMBOL_REDUCED_WEIGHT, "recent_fit_reduced")
    return SymbolScore(score, "paused", 0.0, "score_too_low")
