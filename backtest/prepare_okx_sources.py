"""Prepare local C:\\okx logs and cache files for V2 OHLCV backtests."""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import pandas as pd

PRICE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) .*? - price=(?P<price>\d+(?:\.\d+)?)"
)


def _ticks_to_ohlcv(ticks: pd.DataFrame, bar: str) -> pd.DataFrame:
    """Aggregate timestamped price ticks into OHLCV candles."""
    if ticks.empty:
        raise ValueError("No ticks found")
    df = ticks.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["ts", "price"]).sort_values("ts")
    df = df.drop_duplicates(subset=["ts", "price"])
    candles = (
        df.set_index("ts")["price"]
        .resample(bar)
        .ohlc()
        .dropna()
        .reset_index()
    )
    candles["vol"] = 0.0
    candles["volCcy"] = 0.0
    candles["volCcyQuote"] = 0.0
    candles["confirm"] = 1
    candles["ts"] = candles["ts"].astype("int64") // 1_000_000
    if int(candles["ts"].median()) < 100_000_000_000:
        candles["ts"] = candles["ts"] * 1000
    return candles[["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"]]


def convert_logs(log_dir: Path, out_path: Path, bar: str) -> None:
    """Parse BARS MARKET/INFO price lines and save OHLCV CSV."""
    rows: list[tuple[str, float]] = []
    for path in sorted(log_dir.glob("boll_pin_*.log")):
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = PRICE_RE.search(line)
                if match:
                    rows.append((match.group("ts"), float(match.group("price"))))
    candles = _ticks_to_ohlcv(pd.DataFrame(rows, columns=["ts", "price"]), bar)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candles.to_csv(out_path, index=False)
    print(f"Saved {len(candles)} candles from {len(rows)} log ticks to {out_path}")


def convert_cache(cache_path: Path, out_path: Path, bar: str) -> None:
    """Load a cached tick DataFrame and save OHLCV CSV."""
    obj = pickle.load(cache_path.open("rb"))
    if not hasattr(obj, "columns"):
        raise TypeError(f"{cache_path} is not a pandas DataFrame")
    if "ts" not in obj.columns or "price" not in obj.columns:
        raise ValueError(f"{cache_path} missing ts/price columns")
    candles = _ticks_to_ohlcv(obj[["ts", "price"]], bar)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candles.to_csv(out_path, index=False)
    print(f"Saved {len(candles)} candles from {len(obj)} cache ticks to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare C:\\okx local sources for V2 backtest")
    parser.add_argument("--source", choices=["logs", "cache"], required=True)
    parser.add_argument("--path", required=True, help="Log directory or cache pkl path")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bar", default="15min", help="Pandas resample bar, e.g. 15min, 1h")
    args = parser.parse_args()

    if args.source == "logs":
        convert_logs(Path(args.path), Path(args.out), args.bar)
    else:
        convert_cache(Path(args.path), Path(args.out), args.bar)


if __name__ == "__main__":
    main()
