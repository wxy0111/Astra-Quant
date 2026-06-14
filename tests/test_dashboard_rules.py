"""Focused checks for the local dashboard data layer.

Run with:
    python tests/test_dashboard_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard import config_snapshot, dashboard_state, list_reports, live_replay


def test_dashboard_config_snapshot_hides_secrets() -> None:
    snapshot = config_snapshot()
    assert "OKX_API_KEY" not in snapshot
    assert snapshot["default_inst_id"].endswith("-USDT-SWAP")
    assert snapshot["live_max_risk_usdt"] > 0


def test_dashboard_state_has_expected_sections() -> None:
    state = dashboard_state()
    assert "reports" in state
    assert "live_events" in state
    assert "replay" in state
    assert "config" in state
    assert state["updated_at"]


def test_report_listing_is_safe_when_reports_exist_or_missing() -> None:
    reports = list_reports(limit=3)
    assert isinstance(reports, list)
    for report in reports:
        assert report["name"].endswith(".json")


def test_live_replay_has_expected_sections() -> None:
    replay = live_replay(limit=5)
    assert "cycles" in replay
    assert "placed" in replay
    assert "blocked" in replay
    assert "by_symbol" in replay


if __name__ == "__main__":
    test_dashboard_config_snapshot_hides_secrets()
    test_dashboard_state_has_expected_sections()
    test_report_listing_is_safe_when_reports_exist_or_missing()
    test_live_replay_has_expected_sections()
    print("Dashboard rule tests passed")
