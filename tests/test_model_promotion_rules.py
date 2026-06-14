"""Focused checks for LightGBM model promotion gates.

Run with:
    python tests/test_model_promotion_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.train_and_promote_model import _score_passes


def test_score_passes_requires_return_and_drawdown() -> None:
    assert _score_passes({"walk_forward_1000u_return_pct": 5.1, "max_single_asset_drawdown_pct": 7.9}, 5.0, 8.0)
    assert not _score_passes({"walk_forward_1000u_return_pct": 4.9, "max_single_asset_drawdown_pct": 7.9}, 5.0, 8.0)
    assert not _score_passes({"walk_forward_1000u_return_pct": 5.1, "max_single_asset_drawdown_pct": 8.1}, 5.0, 8.0)


if __name__ == "__main__":
    test_score_passes_requires_return_and_drawdown()
    print("Model promotion rule tests passed")
