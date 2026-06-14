"""Download OKX historical candlesticks from the public API.

Usage:
    python -m backtest.download_okx_history --inst ETH-USDT-SWAP --bar 15m --days 60
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp

BASE_URL = "https://www.okx.com"
HISTORY_PATH = "/api/v5/market/history-candles"
OUT_DIR = Path("backtest") / "data"


def parse_end_datetime(value: str | None) -> datetime:
    """Parse a UTC end timestamp for sample-window downloads."""
    if not value:
        return datetime.now(timezone.utc)
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    if "T" not in normalized and " " not in normalized:
        normalized = normalized + "T00:00:00+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_ts(ms: int) -> str:
    """Format exchange timestamps defensively for Windows and odd API rows."""
    try:
        if ms <= 0:
            return str(ms)
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError):
        return str(ms)


async def fetch_batch(session: aiohttp.ClientSession, inst_id: str, bar: str, after: str | None = None) -> list[list[str]]:
    """Fetch one batch of candles older than `after`."""
    params = {"instId": inst_id, "bar": bar, "limit": "100"}
    if after:
        params["after"] = after
    for attempt in range(5):
        async with session.get(BASE_URL + HISTORY_PATH, params=params) as response:
            data = await response.json()
        if data.get("code") == "0":
            return data["data"]
        if data.get("code") == "50011":
            await asyncio.sleep(0.5 * (attempt + 1))
            continue
        raise RuntimeError(f"OKX error: {data.get('code')} {data.get('msg')}")
    raise RuntimeError("Too many OKX rate-limit retries")


async def download(inst_id: str, bar: str, days: int, out_path: Path, end_ts: datetime | None = None, quiet: bool = False) -> None:
    """Download history ending now, oldest-to-newest, into a CSV."""
    end_ts = end_ts or datetime.now(timezone.utc)
    start_ts = end_ts - timedelta(days=days)
    rows: list[list[str]] = []

    async with aiohttp.ClientSession() as session:
        after = str(int(end_ts.timestamp() * 1000))
        while True:
            batch = await fetch_batch(session, inst_id, bar, after=after)
            batch = [row for row in batch if row and row[0].isdigit() and int(row[0]) > 0]
            if not batch:
                break
            rows.extend(batch)
            oldest_ts = int(batch[-1][0])
            after = str(oldest_ts)
            if oldest_ts < int(start_ts.timestamp() * 1000):
                break
            await asyncio.sleep(0.12)
            if not quiet:
                print(f"fetched {len(rows)} rows, oldest={format_ts(oldest_ts)}", flush=True)

    rows.sort(key=lambda item: int(item[0]))
    deduped = []
    seen = set()
    for row in rows:
        if row[0] in seen:
            continue
        seen.add(row[0])
        deduped.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
        writer.writerows(deduped)
    print(f"Saved {len(deduped)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OKX historical candles")
    parser.add_argument("--inst", default="ETH-USDT-SWAP")
    parser.add_argument("--bar", default="15m")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--end-date", default=None, help="UTC end date/time, e.g. 2026-04-15 or 2026-04-15T12:00:00Z")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else OUT_DIR / f"{args.inst}_{args.bar}_{args.days}d.csv"
    end_ts = parse_end_datetime(args.end_date)
    start = time.time()
    print(f"Window end={end_ts.isoformat()} days={args.days}")
    asyncio.run(download(args.inst, args.bar, args.days, out_path, end_ts=end_ts, quiet=args.quiet))
    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
