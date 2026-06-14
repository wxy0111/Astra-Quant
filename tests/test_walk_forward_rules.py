"""Focused checks for walk-forward universe selection.

Run with:
    python tests/test_walk_forward_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.symbol_score import SymbolScore
from src.walk_forward import (
    assign_strength_ranks,
    market_gate,
    rolling_gate_allows_direction,
    rolling_market_gates_by_timestamp,
    split_train_test,
)


def test_split_train_test_preserves_order() -> None:
    df = pd.DataFrame({"close": list(range(10))})
    train, test = split_train_test(df, train_fraction=0.6)
    assert train["close"].tolist() == [0, 1, 2, 3, 4, 5]
    assert test["close"].tolist() == [6, 7, 8, 9]


def test_market_gate_closes_when_breadth_is_weak() -> None:
    closed = market_gate(
        [
            {"momentum_return_pct": -4.0, "score": SymbolScore(0, "paused", 0.0, "bad")},
            {"momentum_return_pct": 2.0, "score": SymbolScore(45, "reduced", 0.35, "ok")},
            {"momentum_return_pct": -1.0, "score": SymbolScore(0, "paused", 0.0, "bad")},
        ]
    )
    assert not closed.open
    assert closed.reason == "weak_market_breadth"


def test_strength_rank_keeps_only_top_positive_candidates() -> None:
    ranked = assign_strength_ranks(
        [
            {"inst_id": "A", "momentum_return_pct": 10.0, "score": SymbolScore(70, "active", 1.0, "ok")},
            {"inst_id": "B", "momentum_return_pct": 2.0, "score": SymbolScore(45, "reduced", 0.35, "ok")},
            {"inst_id": "C", "momentum_return_pct": -3.0, "score": SymbolScore(70, "active", 1.0, "ok")},
        ],
        min_rank=0.50,
    )
    by_id = {item["inst_id"]: item for item in ranked}
    assert by_id["A"]["walk_forward_state"] == "active"
    assert by_id["B"]["walk_forward_state"] == "reduced"
    assert by_id["C"]["walk_forward_state"] == "paused"


def test_top_momentum_can_receive_probe_weight_when_score_is_weak() -> None:
    ranked = assign_strength_ranks(
        [
            {"inst_id": "A", "momentum_return_pct": 12.0, "score": SymbolScore(20, "paused", 0.0, "score_too_low")},
            {"inst_id": "B", "momentum_return_pct": 3.0, "score": SymbolScore(0, "paused", 0.0, "score_too_low")},
            {"inst_id": "C", "momentum_return_pct": -2.0, "score": SymbolScore(70, "active", 1.0, "ok")},
        ],
        min_rank=0.50,
    )
    by_id = {item["inst_id"]: item for item in ranked}
    assert by_id["A"]["walk_forward_state"] == "reduced"
    assert by_id["A"]["walk_forward_reason"] == "relative_strength_probe"
    assert by_id["B"]["walk_forward_state"] == "paused"


def test_reduced_score_with_negative_train_return_is_paused() -> None:
    ranked = assign_strength_ranks(
        [
            {
                "inst_id": "A",
                "momentum_return_pct": 20.0,
                "score": SymbolScore(45, "reduced", 0.35, "recent_fit_reduced"),
                "train_result": {"return_pct": -1.5},
            },
            {
                "inst_id": "B",
                "momentum_return_pct": 10.0,
                "score": SymbolScore(70, "active", 1.0, "recent_fit_active"),
                "train_result": {"return_pct": 2.0},
            },
        ],
        min_rank=0.50,
    )
    by_id = {item["inst_id"]: item for item in ranked}
    assert by_id["A"]["walk_forward_state"] == "paused"
    assert by_id["A"]["walk_forward_reason"] == "reduced_score_negative_train_return"
    assert by_id["B"]["walk_forward_state"] == "active"


def test_rolling_market_gate_closes_when_recent_test_breadth_turns_weak() -> None:
    frames = {
        "A": pd.DataFrame({"ts": [1, 2, 3, 4], "close": [100.0, 99.0, 98.0, 97.0]}),
        "B": pd.DataFrame({"ts": [1, 2, 3, 4], "close": [100.0, 101.0, 100.0, 99.0]}),
        "C": pd.DataFrame({"ts": [1, 2, 3, 4], "close": [100.0, 101.0, 102.0, 103.0]}),
    }
    gates = rolling_market_gates_by_timestamp(
        frames,
        lookback_bars=3,
        min_breadth=0.50,
        min_average_momentum_pct=0.0,
        min_average_abs_momentum_pct=0.0,
    )
    assert not gates[4].open
    assert gates[4].reason == "rolling_weak_market_breadth"


def test_rolling_market_gate_is_direction_aware_for_weak_markets() -> None:
    frames = {
        "A": pd.DataFrame({"ts": [1, 2, 3, 4], "close": [100.0, 99.0, 98.0, 97.0]}),
        "B": pd.DataFrame({"ts": [1, 2, 3, 4], "close": [100.0, 99.5, 99.0, 98.5]}),
        "C": pd.DataFrame({"ts": [1, 2, 3, 4], "close": [100.0, 100.5, 100.2, 100.0]}),
    }
    gate = rolling_market_gates_by_timestamp(
        frames,
        lookback_bars=3,
        min_breadth=0.50,
        min_average_momentum_pct=0.0,
        min_average_abs_momentum_pct=0.0,
    )[4]
    assert not rolling_gate_allows_direction(gate, "long")
    assert rolling_gate_allows_direction(gate, "short")


if __name__ == "__main__":
    test_split_train_test_preserves_order()
    test_market_gate_closes_when_breadth_is_weak()
    test_strength_rank_keeps_only_top_positive_candidates()
    test_top_momentum_can_receive_probe_weight_when_score_is_weak()
    test_reduced_score_with_negative_train_return_is_paused()
    test_rolling_market_gate_closes_when_recent_test_breadth_turns_weak()
    test_rolling_market_gate_is_direction_aware_for_weak_markets()
    print("Walk-forward rule tests passed")
