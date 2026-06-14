"""Multi-asset universe grouping and profile helpers."""

from __future__ import annotations

from pathlib import Path

from src import config


def listed_universe(include_watch: bool = True) -> list[str]:
    """Return configured instruments without duplicates, preserving priority."""
    instruments = config.CORE_UNIVERSE + config.LIQUID_ALT_UNIVERSE
    if include_watch:
        instruments += config.WATCH_UNIVERSE
    seen = set()
    result = []
    for inst_id in instruments:
        if inst_id in seen:
            continue
        seen.add(inst_id)
        result.append(inst_id)
    return result


def instrument_profile(inst_id: str) -> str:
    """Return core, liquid_alt, watch, or unknown for an instrument."""
    if inst_id in config.CORE_UNIVERSE:
        return "core"
    if inst_id in config.LIQUID_ALT_UNIVERSE:
        return "liquid_alt"
    if inst_id in config.WATCH_UNIVERSE:
        return "watch"
    return "unknown"


def profile_risk_multiplier(profile: str) -> float:
    """Return the risk multiplier used by research backtests."""
    if profile == "core":
        return config.CORE_RISK_MULTIPLIER
    if profile == "liquid_alt":
        return config.LIQUID_ALT_RISK_MULTIPLIER
    if profile == "watch":
        return config.WATCH_RISK_MULTIPLIER
    return config.WATCH_RISK_MULTIPLIER


def infer_inst_id_from_csv(path: str | Path) -> str:
    """Infer OKX instrument id from a CSV filename when possible."""
    name = Path(path).name.upper()
    for inst_id in listed_universe(include_watch=True):
        token = inst_id.upper()
        if token in name:
            return inst_id
        compact = token.replace("-USDT-SWAP", "")
        if compact and compact in name:
            return inst_id
    return "UNKNOWN-USDT-SWAP"
