"""Download OKX candles for the configured multi-asset universe."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.universe_engine import listed_universe


def main() -> None:
    parser = argparse.ArgumentParser(description="Download configured OKX universe candles")
    parser.add_argument("--bar", default="15m")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--out-dir", default="backtest/data")
    parser.add_argument("--end-date", default=None, help="UTC end date/time passed to each symbol download")
    parser.add_argument("--no-watch", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for inst_id in listed_universe(include_watch=not args.no_watch):
        out = out_dir / f"{inst_id}_{args.bar}_{args.days}d.csv"
        if args.skip_existing and out.exists() and out.stat().st_size > 1024:
            print(f"Skipping existing {inst_id} -> {out}")
            continue
        cmd = [
            sys.executable,
            "-m",
            "backtest.download_okx_history",
            "--inst",
            inst_id,
            "--bar",
            args.bar,
            "--days",
            str(args.days),
            "--out",
            str(out),
        ]
        if args.end_date:
            cmd += ["--end-date", args.end_date]
        if args.quiet:
            cmd += ["--quiet"]
        print(f"Downloading {inst_id} -> {out}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
