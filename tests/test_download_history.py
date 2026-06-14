"""Focused checks for historical download date handling.

Run with:
    python tests/test_download_history.py
"""

from __future__ import annotations

import sys
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.download_okx_history import parse_end_datetime


def test_parse_end_datetime_accepts_date_only_as_utc_midnight() -> None:
    parsed = parse_end_datetime("2026-04-15")
    assert parsed.year == 2026
    assert parsed.month == 4
    assert parsed.day == 15
    assert parsed.hour == 0
    assert parsed.tzinfo == timezone.utc


def test_parse_end_datetime_accepts_iso_datetime_with_z() -> None:
    parsed = parse_end_datetime("2026-04-15T12:30:00Z")
    assert parsed.hour == 12
    assert parsed.minute == 30
    assert parsed.tzinfo == timezone.utc


if __name__ == "__main__":
    test_parse_end_datetime_accepts_date_only_as_utc_midnight()
    test_parse_end_datetime_accepts_iso_datetime_with_z()
    print("Download history tests passed")
