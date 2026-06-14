"""Focused checks for guarded OKX live loop helpers.

Run with:
    python tests/test_live_loop_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.run_live_loop_okx import _cooldown_active, _count_open_positions, _has_pending_order, _symbols_from_args


def test_symbols_from_args_preserves_order_and_uniqueness() -> None:
    assert _symbols_from_args("BTC-USDT-SWAP,ETH-USDT-SWAP,BTC-USDT-SWAP", include_watch=False) == [
        "BTC-USDT-SWAP",
        "ETH-USDT-SWAP",
    ]


def test_pending_order_detects_regular_and_algo_orders() -> None:
    assert _has_pending_order([{"instId": "BTC-USDT-SWAP"}], [], "BTC-USDT-SWAP")
    assert _has_pending_order([], [{"instId": "ETH-USDT-SWAP"}], "ETH-USDT-SWAP")
    assert not _has_pending_order([], [], "SOL-USDT-SWAP")


def test_cooldown_blocks_recent_symbol() -> None:
    state = {"last_order_ts_by_inst": {"BTC-USDT-SWAP": 100}}
    assert _cooldown_active(state, "BTC-USDT-SWAP", cooldown_sec=60, now=120)
    assert not _cooldown_active(state, "BTC-USDT-SWAP", cooldown_sec=60, now=200)


def test_count_open_positions_ignores_zero_rows() -> None:
    rows = [{"pos": "0"}, {"pos": "1.2"}, {"pos": "-0.5"}, {"pos": ""}]
    assert _count_open_positions(rows) == 2


if __name__ == "__main__":
    test_symbols_from_args_preserves_order_and_uniqueness()
    test_pending_order_detects_regular_and_algo_orders()
    test_cooldown_blocks_recent_symbol()
    test_count_open_positions_ignores_zero_rows()
    print("Live loop rule tests passed")
