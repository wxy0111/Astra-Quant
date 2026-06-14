"""Build a future-regime training dataset from downloaded OHLCV data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data import load_ohlcv_csv
from src.regime_features import regime_features_from_window
from src.regime_label import label_future_regime
from src.universe_engine import infer_inst_id_from_csv


def _cutoff(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.Timestamp(datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)).tz_localize(None)


def build_rows_for_csv(
    csv_path: str | Path,
    context_bars: int = 96 * 120,
    horizon_bars: int = 96 * 3,
    stride: int = 96,
    end_date: str | None = None,
) -> list[dict]:
    """Build regime rows for one instrument."""
    csv_path = Path(csv_path)
    inst_id = infer_inst_id_from_csv(csv_path)
    raw = load_ohlcv_csv(str(csv_path))
    cutoff = _cutoff(end_date)
    if cutoff is not None:
        raw = raw[raw["ts"] < cutoff].reset_index(drop=True)
    rows = []
    end = len(raw) - horizon_bars
    for i in range(context_bars, end, stride):
        window = raw.iloc[i - context_bars : i]
        future = raw.iloc[i : i + horizon_bars]
        features = regime_features_from_window(window)
        label = label_future_regime(future)
        rows.append(
            {
                "ts": raw.iloc[i]["ts"],
                "inst_id": inst_id,
                "label": label.label,
                "future_return": round(label.future_return, 6),
                "future_efficiency": round(label.efficiency, 6),
                "future_realized_vol": round(label.realized_vol, 6),
                "label_reason": label.reason,
                **features,
            }
        )
    return rows


def build_dataset(
    data_dir: str | Path,
    out_path: str | Path,
    context_bars: int = 96 * 120,
    horizon_bars: int = 96 * 3,
    stride: int = 96,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Build and write all regime rows."""
    rows = []
    for csv_path in sorted(Path(data_dir).glob("*.csv")):
        if infer_inst_id_from_csv(csv_path) == "UNKNOWN-USDT-SWAP":
            continue
        item_rows = build_rows_for_csv(csv_path, context_bars=context_bars, horizon_bars=horizon_bars, stride=stride, end_date=end_date)
        print(f"{csv_path.name}: {len(item_rows)} regime rows", flush=True)
        rows.extend(item_rows)
    df = pd.DataFrame(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build future-regime dataset")
    parser.add_argument("--data-dir", default="backtest/data_train_2y")
    parser.add_argument("--out", default="backtest/datasets/regime_2y.csv")
    parser.add_argument("--context-bars", type=int, default=96 * 120)
    parser.add_argument("--horizon-bars", type=int, default=96 * 3)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()
    build_dataset(args.data_dir, args.out, context_bars=args.context_bars, horizon_bars=args.horizon_bars, stride=args.stride, end_date=args.end_date)


if __name__ == "__main__":
    main()
