"""Download data, train a candidate LightGBM model, validate it, and promote.

The script promotes a candidate only when both recent and OOS walk-forward tests
meet configured return and drawdown thresholds. Promotion backs up the current
live model before replacing it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LIVE_MODEL = Path("backtest/models/meta_model_live.json")


def _run(cmd: list[str]) -> None:
    """Run one subprocess and stream output.

    Args:
        cmd: Command argument list.
    """
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON object.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _score_passes(report: dict[str, Any], min_return_pct: float, max_drawdown_pct: float) -> bool:
    """Return whether a validation report passes promotion gates.

    Args:
        report: Universe walk-forward report.
        min_return_pct: Minimum walk-forward return percentage.
        max_drawdown_pct: Maximum allowed single-asset drawdown percentage.

    Returns:
        True if return and drawdown gates pass.
    """
    return_pct = float(report.get("walk_forward_1000u_return_pct", -999.0))
    drawdown_pct = float(report.get("max_single_asset_drawdown_pct", 999.0))
    return return_pct >= min_return_pct and drawdown_pct <= max_drawdown_pct


def _backup_live_model(tag: str) -> Path | None:
    """Back up the current live model if it exists.

    Args:
        tag: Timestamp or run tag used in the backup filename.

    Returns:
        Backup path, or None when no live model exists.
    """
    if not LIVE_MODEL.exists():
        return None
    backup = LIVE_MODEL.with_name(f"meta_model_live.backup_{tag}.json")
    shutil.copy2(LIVE_MODEL, backup)
    return backup


def promote(candidate_model: Path, tag: str) -> Path | None:
    """Promote a candidate model to the live model path.

    Args:
        candidate_model: Candidate model path.
        tag: Run tag used for backup naming.

    Returns:
        Backup path if a previous live model was present.
    """
    backup = _backup_live_model(tag)
    LIVE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_model, LIVE_MODEL)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and promote the Astra-Quant live meta model")
    parser.add_argument("--train-days", type=int, default=730)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--bar", default="15m")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--target-column", default="quality_label")
    parser.add_argument("--data-train-dir", default="backtest/data_train_2y")
    parser.add_argument("--data-recent-dir", default="backtest/data")
    parser.add_argument("--data-oos-dir", default="backtest/data_oos_20260214_20260415")
    parser.add_argument("--min-recent-return-pct", type=float, default=5.0)
    parser.add_argument("--min-oos-return-pct", type=float, default=0.0)
    parser.add_argument("--max-recent-drawdown-pct", type=float, default=12.0)
    parser.add_argument("--max-oos-drawdown-pct", type=float, default=8.0)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-promote", action="store_true")
    args = parser.parse_args()

    tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dataset = Path(f"backtest/datasets/meta_signals_agent_candidate_{tag}.csv")
    model = Path(f"backtest/models/meta_model_candidate_{tag}.json")
    metrics = Path(f"backtest/reports/meta_model_candidate_{tag}_metrics.json")
    recent_report = Path(f"backtest/reports/promote_recent_{tag}.json")
    oos_report = Path(f"backtest/reports/promote_oos_{tag}.json")
    summary_path = Path(f"backtest/reports/promote_summary_{tag}.json")

    if not args.skip_download:
        _run([sys.executable, "-m", "backtest.download_universe_history", "--bar", args.bar, "--days", str(args.train_days), "--out-dir", args.data_train_dir, "--no-watch", "--quiet"])
        _run([sys.executable, "-m", "backtest.download_universe_history", "--bar", args.bar, "--days", str(args.test_days), "--out-dir", args.data_recent_dir, "--no-watch", "--quiet"])

    _run([
        sys.executable,
        "-m",
        "backtest.build_meta_dataset",
        "--data-dir",
        args.data_train_dir,
        "--out",
        str(dataset),
        "--stride",
        str(args.stride),
        "--agent-market",
    ])
    _run([
        sys.executable,
        "-m",
        "backtest.train_meta_model",
        "--model-type",
        "lightgbm",
        "--target-column",
        args.target_column,
        "--dataset",
        str(dataset),
        "--model-out",
        str(model),
        "--metrics-out",
        str(metrics),
        "--threshold",
        str(args.threshold),
    ])
    _run([
        sys.executable,
        "-m",
        "backtest.run_universe_backtest",
        "--data-dir",
        args.data_recent_dir,
        "--no-watch",
        "--walk-forward",
        "--agent-market",
        "--meta-model",
        str(model),
        "--json-out",
        str(recent_report),
    ])
    _run([
        sys.executable,
        "-m",
        "backtest.run_universe_backtest",
        "--data-dir",
        args.data_oos_dir,
        "--no-watch",
        "--walk-forward",
        "--agent-market",
        "--meta-model",
        str(model),
        "--json-out",
        str(oos_report),
    ])

    recent = _load_json(recent_report)
    oos = _load_json(oos_report)
    recent_ok = _score_passes(recent, args.min_recent_return_pct, args.max_recent_drawdown_pct)
    oos_ok = _score_passes(oos, args.min_oos_return_pct, args.max_oos_drawdown_pct)
    promoted = bool(args.force_promote or (recent_ok and oos_ok))
    backup = promote(model, tag) if promoted else None
    summary = {
        "tag": tag,
        "candidate_model": str(model),
        "live_model": str(LIVE_MODEL),
        "promoted": promoted,
        "force_promote": args.force_promote,
        "backup": str(backup) if backup else None,
        "recent": {
            "report": str(recent_report),
            "return_pct": recent.get("walk_forward_1000u_return_pct"),
            "drawdown_pct": recent.get("max_single_asset_drawdown_pct"),
            "passed": recent_ok,
        },
        "oos": {
            "report": str(oos_report),
            "return_pct": oos.get("walk_forward_1000u_return_pct"),
            "drawdown_pct": oos.get("max_single_asset_drawdown_pct"),
            "passed": oos_ok,
        },
        "gates": {
            "min_recent_return_pct": args.min_recent_return_pct,
            "min_oos_return_pct": args.min_oos_return_pct,
            "max_recent_drawdown_pct": args.max_recent_drawdown_pct,
            "max_oos_drawdown_pct": args.max_oos_drawdown_pct,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
