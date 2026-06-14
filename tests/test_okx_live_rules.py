"""Focused checks for guarded OKX live execution helpers.

Run with:
    python tests/test_okx_live_rules.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.okx_client import LiveRiskLimits, load_credentials, plan_to_order_sizing, round_down_to_step, validate_live_risk
from src.trade_plan import TradePlan


def _plan() -> TradePlan:
    return TradePlan(
        direction="long",
        style="trend_pullback",
        entry=100.0,
        stop=96.0,
        take_profit=112.0,
        partial_take_profit=106.0,
        trailing_distance=5.0,
        qty=0.037,
        margin=0.74,
        risk_usdt=0.148,
        risk_fraction=0.01,
        reward_risk=3.0,
        confidence=0.82,
        partial_exit_fraction=0.3,
        breakeven_arm_r=1.0,
        hold_for_trend=True,
        leverage=5,
        reasons=("test",),
    )


def test_round_down_to_okx_step() -> None:
    assert round_down_to_step(3.789, "0.1") == "3.7"
    assert round_down_to_step(3.0, "1") == "3"


def test_plan_to_swap_contract_size_uses_contract_value_and_lot_size() -> None:
    instrument = {"ctVal": "0.01", "lotSz": "1", "minSz": "1"}
    sizing = plan_to_order_sizing(_plan(), "BTC-USDT-SWAP", instrument)
    assert sizing.side == "buy"
    assert sizing.size == "3"
    assert sizing.contracts == 3.0


def test_live_risk_rejects_over_limit_plan() -> None:
    sizing = plan_to_order_sizing(_plan(), "BTC-USDT-SWAP", {"ctVal": "0.01", "lotSz": "1", "minSz": "1"})
    try:
        validate_live_risk(_plan(), sizing, LiveRiskLimits(max_risk_usdt=0.01, max_margin_usdt=10.0, max_notional_usdt=10.0, max_leverage=5))
    except ValueError as exc:
        assert "max_risk_usdt" in str(exc)
    else:
        raise AssertionError("expected risk limit rejection")


def test_credentials_load_from_env_file(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "OKX_API_KEY=key",
                "OKX_API_SECRET=secret",
                "OKX_PASSPHRASE=pass",
                "OKX_FLAG=1",
            ]
        ),
        encoding="utf-8",
    )
    old = {key: os.environ.pop(key, None) for key in ("OKX_API_KEY", "OKX_API_SECRET", "OKX_PASSPHRASE", "OKX_FLAG")}
    try:
        credentials = load_credentials(env_path)
        assert credentials.api_key == "key"
        assert credentials.simulated
    finally:
        for key, value in old.items():
            if value is not None:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)


if __name__ == "__main__":
    test_round_down_to_okx_step()
    test_plan_to_swap_contract_size_uses_contract_value_and_lot_size()
    test_live_risk_rejects_over_limit_plan()
    with tempfile.TemporaryDirectory() as tmp:
        test_credentials_load_from_env_file(Path(tmp))
    print("OKX live rule tests passed")
