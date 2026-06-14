"""LightGBM future-regime model loading and decision logic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb


@dataclass(frozen=True)
class RegimeDecision:
    """Regime-model verdict for a candidate trade."""

    allowed: bool
    label: str
    probability: float
    reason: str


@dataclass
class LightGBMRegimeModel:
    """Saved LightGBM multiclass model for future market shape."""

    feature_names: list[str]
    labels: list[str]
    booster_text: str
    threshold: float = 0.35
    _booster: lgb.Booster | None = field(default=None, init=False, repr=False)

    def booster(self) -> lgb.Booster:
        if self._booster is None:
            self._booster = lgb.Booster(model_str=self.booster_text)
        return self._booster

    def _x(self, features: dict[str, float]) -> list[float]:
        return [float(features.get(name, 0.0)) for name in self.feature_names]

    def predict(self, features: dict[str, float]) -> tuple[str, float]:
        probs = self.booster().predict([self._x(features)])[0]
        index = int(max(range(len(probs)), key=lambda idx: probs[idx]))
        return self.labels[index], float(probs[index])

    def decision(self, features: dict[str, float], direction: str) -> RegimeDecision:
        label, probability = self.predict(features)
        if probability < self.threshold:
            return RegimeDecision(False, label, probability, "regime_probability_low")
        if label in ("chop", "high_vol") and probability >= 0.78:
            return RegimeDecision(False, label, probability, "regime_risk_block")
        if direction == "long" and label == "trend_down" and probability >= 0.65:
            return RegimeDecision(False, label, probability, "regime_direction_mismatch")
        if direction == "short" and label == "trend_up" and probability >= 0.65:
            return RegimeDecision(False, label, probability, "regime_direction_mismatch")
        reason = "regime_direction_aligned" if (direction == "long" and label == "trend_up") or (direction == "short" and label == "trend_down") else "regime_not_blocked"
        return RegimeDecision(True, label, probability, reason)

    def multiplier_for_label(self, label: str, probability: float, direction: str) -> float:
        """Map predicted future shape to a risk multiplier."""
        if probability < self.threshold:
            return 0.50
        if label == "high_vol" and probability >= 0.65:
            return 0.0
        if label == "chop":
            if probability >= 0.78:
                return 0.0
            if probability >= 0.62:
                return 0.35
            return 0.60
        if direction == "long" and label == "trend_down":
            return 0.0 if probability >= 0.65 else 0.35
        if direction == "short" and label == "trend_up":
            return 0.0 if probability >= 0.65 else 0.35
        if direction == "long" and label == "trend_up":
            return 1.15
        if direction == "short" and label == "trend_down":
            return 1.15
        return 0.65

    def risk_multiplier(self, features: dict[str, float], direction: str) -> tuple[float, str, float]:
        label, probability = self.predict(features)
        return self.multiplier_for_label(label, probability, direction), label, probability

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "model_type": "lightgbm_regime",
                    "feature_names": self.feature_names,
                    "labels": self.labels,
                    "booster_text": self.booster_text,
                    "threshold": self.threshold,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def load_regime_model(path: str | Path | None) -> LightGBMRegimeModel | None:
    """Load a regime model if a path was provided."""
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return LightGBMRegimeModel(
        feature_names=list(data["feature_names"]),
        labels=list(data["labels"]),
        booster_text=str(data["booster_text"]),
        threshold=float(data.get("threshold", 0.35)),
    )
