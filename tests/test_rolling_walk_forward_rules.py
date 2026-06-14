"""Focused checks for rolling walk-forward window construction.

Run with:
    python tests/test_rolling_walk_forward_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.run_rolling_walk_forward import rolling_windows


def test_rolling_windows_are_chronological_and_fixed_size() -> None:
    df = pd.DataFrame({"ts": pd.date_range("2026-01-01", periods=20, freq="D"), "close": range(20)})
    windows = rolling_windows(df, train_days=6, test_days=3, step_days=3)
    assert len(windows) == 4
    first = windows[0]
    assert first.train["ts"].iloc[0] == pd.Timestamp("2026-01-01")
    assert first.train["ts"].iloc[-1] == pd.Timestamp("2026-01-06")
    assert first.test["ts"].iloc[0] == pd.Timestamp("2026-01-07")
    assert first.test["ts"].iloc[-1] == pd.Timestamp("2026-01-09")
    assert windows[1].train["ts"].iloc[0] == pd.Timestamp("2026-01-04")


if __name__ == "__main__":
    test_rolling_windows_are_chronological_and_fixed_size()
    print("Rolling walk-forward rule tests passed")
