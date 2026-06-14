"""Small dependency-light logistic model for meta-label filtering."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb


@dataclass(frozen=True)
class LogisticMetaModel:
    """A saved logistic regression model with standardization baked in."""

    feature_names: list[str]
    weights: list[float]
    bias: float
    threshold: float
    means: list[float] | None = None
    scales: list[float] | None = None

    def _x(self, features: dict[str, float]) -> list[float]:
        values = [float(features.get(name, 0.0)) for name in self.feature_names]
        if self.means is None or self.scales is None:
            return values
        return [(value - mean) / scale for value, mean, scale in zip(values, self.means, self.scales)]

    def predict_proba(self, features: dict[str, float]) -> float:
        z = self.bias + sum(weight * value for weight, value in zip(self.weights, self._x(features)))
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def allows(self, features: dict[str, float]) -> bool:
        return self.predict_proba(features) >= self.threshold

    def multiplier_for_probability(self, probability: float) -> float:
        """Map predicted signal quality to a risk multiplier."""
        if probability < 0.50:
            return 0.0
        if probability < self.threshold:
            return 0.50
        if probability >= 0.70:
            return 1.20
        return 1.0

    def risk_multiplier(self, features: dict[str, float]) -> float:
        return self.multiplier_for_probability(self.predict_proba(features))

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "weights": self.weights,
                    "bias": self.bias,
                    "threshold": self.threshold,
                    "means": self.means,
                    "scales": self.scales,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


@dataclass
class LightGBMMetaModel:
    """A saved LightGBM binary classifier for meta-label filtering."""

    feature_names: list[str]
    booster_text: str
    threshold: float
    _booster: lgb.Booster | None = field(default=None, init=False, repr=False)

    def _x(self, features: dict[str, float]) -> list[float]:
        return [float(features.get(name, 0.0)) for name in self.feature_names]

    def booster(self) -> lgb.Booster:
        if self._booster is None:
            self._booster = lgb.Booster(model_str=self.booster_text)
        return self._booster

    def predict_proba(self, features: dict[str, float]) -> float:
        return float(self.booster().predict([self._x(features)])[0])

    def allows(self, features: dict[str, float]) -> bool:
        return self.predict_proba(features) >= self.threshold

    def multiplier_for_probability(self, probability: float) -> float:
        """Map predicted signal quality to a risk multiplier."""
        if probability < 0.50:
            return 0.0
        if probability < self.threshold:
            return 0.50
        if probability >= 0.70:
            return 1.20
        return 1.0

    def risk_multiplier(self, features: dict[str, float]) -> float:
        return self.multiplier_for_probability(self.predict_proba(features))

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "model_type": "lightgbm",
                    "feature_names": self.feature_names,
                    "booster_text": self.booster_text,
                    "threshold": self.threshold,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


MetaModel = LogisticMetaModel | LightGBMMetaModel


def load_meta_model(path: str | Path | None) -> MetaModel | None:
    """Load a meta model if a path was provided."""
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("model_type") == "lightgbm":
        return LightGBMMetaModel(
            feature_names=list(data["feature_names"]),
            booster_text=str(data["booster_text"]),
            threshold=float(data["threshold"]),
        )
    return LogisticMetaModel(
        feature_names=list(data["feature_names"]),
        weights=[float(item) for item in data["weights"]],
        bias=float(data["bias"]),
        threshold=float(data["threshold"]),
        means=[float(item) for item in data["means"]] if data.get("means") is not None else None,
        scales=[float(item) for item in data["scales"]] if data.get("scales") is not None else None,
    )
