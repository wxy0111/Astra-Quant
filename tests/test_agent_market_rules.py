"""Focused checks for local agent signal-market plumbing.

Run with:
    python tests/test_agent_market_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_market import (
    agent_weights_from_scoreboard,
    allowed_agents_from_scoreboard,
    generate_agent_setups,
    score_agent_results,
)
from src.market_state import MarketState


def _window() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "close": 100.0,
                "open": 101.0,
                "high": 102.0,
                "low": 99.0,
                "atr": 2.0,
                "atr_pct": 0.02,
                "ema_fast": 101.0,
                "ema_slow": 100.0,
                "ema_trend": 99.0,
                "donchian_high": 104.0,
                "donchian_low": 96.0,
                "boll_upper": 106.0,
                "boll_lower": 98.0,
                "boll_mid": 102.0,
                "adx": 28.0,
                "plus_di": 30.0,
                "minus_di": 10.0,
            },
            {
                "close": 103.0,
                "open": 100.0,
                "high": 105.0,
                "low": 100.0,
                "atr": 2.0,
                "atr_pct": 0.02,
                "ema_fast": 102.0,
                "ema_slow": 100.5,
                "ema_trend": 99.5,
                "donchian_high": 102.0,
                "donchian_low": 96.0,
                "boll_upper": 106.0,
                "boll_lower": 98.0,
                "boll_mid": 102.0,
                "adx": 30.0,
                "plus_di": 32.0,
                "minus_di": 8.0,
            },
        ]
    )


def test_agents_generate_named_structured_setups() -> None:
    state = MarketState("trend", "long", 0.80, 0.1, ("bull_ema_stack",))
    proposals = generate_agent_setups(_window(), state)
    assert proposals
    assert all(proposal.agent for proposal in proposals)
    assert all(proposal.setup.direction in ("long", "short") for proposal in proposals)


def test_agent_scoreboard_summarizes_trade_results_by_agent() -> None:
    rows = score_agent_results(
        [
            {"agent": "trend_agent", "pnl": 3.0, "event": "final_exit"},
            {"agent": "trend_agent", "pnl": -1.0, "event": "final_exit"},
            {"agent": "range_agent", "pnl": 2.0, "event": "partial_take_profit"},
        ]
    )
    by_agent = {row["agent"]: row for row in rows}
    assert by_agent["trend_agent"]["trades"] == 2
    assert by_agent["trend_agent"]["pnl"] == 2.0
    assert by_agent["trend_agent"]["win_rate"] == 0.5
    assert by_agent["range_agent"]["trades"] == 1


def test_allowed_agents_require_positive_fit() -> None:
    allowed = allowed_agents_from_scoreboard(
        [
            {"agent": "good", "trades": 8, "pnl": 12.0, "profit_factor": 1.4},
            {"agent": "loss", "trades": 8, "pnl": -3.0, "profit_factor": 0.8},
            {"agent": "thin", "trades": 2, "pnl": 4.0, "profit_factor": 3.0},
        ],
        min_trades=5,
        min_profit_factor=1.0,
    )
    assert allowed == {"good"}


def test_agent_weights_scale_with_recent_fit_and_keep_small_probe() -> None:
    weights = agent_weights_from_scoreboard(
        [
            {"agent": "excellent", "trades": 12, "pnl": 30.0, "profit_factor": 2.2, "win_rate": 0.65, "worst_trade": -2.0},
            {"agent": "ok", "trades": 8, "pnl": 6.0, "profit_factor": 1.15, "win_rate": 0.50, "worst_trade": -3.0},
            {"agent": "loss", "trades": 9, "pnl": -5.0, "profit_factor": 0.7, "win_rate": 0.30, "worst_trade": -8.0},
            {"agent": "thin", "trades": 2, "pnl": 8.0, "profit_factor": 3.0, "win_rate": 1.0, "worst_trade": -1.0},
        ]
    )
    assert weights["excellent"] == 1.25
    assert weights["ok"] in (0.75, 1.0)
    assert weights["loss"] == 0.0
    assert weights["thin"] == 0.25


if __name__ == "__main__":
    test_agents_generate_named_structured_setups()
    test_agent_scoreboard_summarizes_trade_results_by_agent()
    test_allowed_agents_require_positive_fit()
    test_agent_weights_scale_with_recent_fit_and_keep_small_probe()
    print("Agent market rule tests passed")
