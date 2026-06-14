"""Minimal OKX REST client and live-order risk helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.trade_plan import TradePlan


BASE_URL = "https://www.okx.com"


@dataclass(frozen=True)
class OKXCredentials:
    """API credentials loaded from environment variables or a local .env file."""

    api_key: str
    api_secret: str
    passphrase: str
    flag: str = "0"

    @property
    def simulated(self) -> bool:
        return self.flag == "1"


@dataclass(frozen=True)
class OrderSizing:
    """Exchange-sized order details derived from a research TradePlan."""

    inst_id: str
    side: str
    size: str
    contracts: float
    notional_usdt: float


@dataclass(frozen=True)
class LiveRiskLimits:
    """Hard limits applied immediately before sending an order to OKX."""

    max_risk_usdt: float
    max_margin_usdt: float
    max_notional_usdt: float
    max_leverage: int


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE pairs without overriding already-set environment variables."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_credentials(env_path: str | Path = ".env") -> OKXCredentials:
    """Load OKX credentials from .env/environment and fail closed if incomplete."""
    load_dotenv(env_path)
    api_key = os.getenv("OKX_API_KEY", "").strip()
    api_secret = os.getenv("OKX_API_SECRET", os.getenv("OKX_SECRET_KEY", "")).strip()
    passphrase = os.getenv("OKX_PASSPHRASE", "").strip()
    flag = os.getenv("OKX_FLAG", "0").strip()
    missing = [
        name
        for name, value in (
            ("OKX_API_KEY", api_key),
            ("OKX_API_SECRET or OKX_SECRET_KEY", api_secret),
            ("OKX_PASSPHRASE", passphrase),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"missing OKX credentials: {', '.join(missing)}")
    if flag not in ("0", "1"):
        raise ValueError("OKX_FLAG must be 0 for live trading or 1 for simulated trading")
    return OKXCredentials(api_key=api_key, api_secret=api_secret, passphrase=passphrase, flag=flag)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time() % 1) * 1000):03d}Z"


def _sign(secret: str, timestamp: str, method: str, path: str, body: str) -> str:
    payload = f"{timestamp}{method.upper()}{path}{body}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def _decimal(value: float | str) -> Decimal:
    return Decimal(str(value))


def round_down_to_step(value: float, step: float | str) -> str:
    """Round a positive number down to an OKX lot-size step."""
    step_decimal = _decimal(step)
    if step_decimal <= 0:
        raise ValueError("step must be positive")
    rounded = (_decimal(value) / step_decimal).to_integral_value(rounding=ROUND_DOWN) * step_decimal
    return format(rounded.normalize(), "f")


def _okx_code_ok(data: dict[str, Any]) -> bool:
    return str(data.get("code")) == "0"


class OKXRestClient:
    """Small synchronous REST client for the OKX v5 endpoints this bot needs."""

    def __init__(self, credentials: OKXCredentials | None = None, base_url: str = BASE_URL, timeout: int = 15):
        self.credentials = credentials
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, params: dict[str, Any] | None = None, private: bool = False) -> dict[str, Any]:
        method = method.upper()
        params = {key: value for key, value in (params or {}).items() if value is not None}
        body = ""
        signed_path = path
        url = self.base_url + path
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "okx-new-live-runner/1.0",
        }

        if method == "GET" and params:
            query = urlencode(params)
            signed_path = f"{path}?{query}"
            url = f"{url}?{query}"
        elif method != "GET":
            body = json.dumps(params, separators=(",", ":"))

        if private:
            if self.credentials is None:
                raise ValueError("private OKX request requires credentials")
            timestamp = _timestamp()
            headers.update(
                {
                    "OK-ACCESS-KEY": self.credentials.api_key,
                    "OK-ACCESS-SIGN": _sign(self.credentials.api_secret, timestamp, method, signed_path, body),
                    "OK-ACCESS-TIMESTAMP": timestamp,
                    "OK-ACCESS-PASSPHRASE": self.credentials.passphrase,
                }
            )
            if self.credentials.simulated:
                headers["x-simulated-trading"] = "1"

        request = Request(url, data=body.encode("utf-8") if body else None, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OKX HTTP {exc.code}: {detail}") from exc
        if not _okx_code_ok(data):
            raise RuntimeError(f"OKX error {data.get('code')}: {data.get('msg')} data={data.get('data')}")
        return data

    def get_candles(self, inst_id: str, bar: str = "15m", limit: int = 300) -> list[list[str]]:
        data = self.request("GET", "/api/v5/market/candles", {"instId": inst_id, "bar": bar, "limit": str(limit)})
        return list(data.get("data", []))

    def get_balance(self, ccy: str = "USDT") -> dict[str, Any]:
        data = self.request("GET", "/api/v5/account/balance", {"ccy": ccy}, private=True)
        return data["data"][0] if data.get("data") else {}

    def get_positions(self, inst_id: str | None = None) -> list[dict[str, Any]]:
        params = {"instId": inst_id} if inst_id else {}
        data = self.request("GET", "/api/v5/account/positions", params, private=True)
        return list(data.get("data", []))

    def get_open_orders(self, inst_id: str | None = None, inst_type: str = "SWAP") -> list[dict[str, Any]]:
        """Return currently open regular orders.

        Args:
            inst_id: Optional instrument id. When omitted, OKX returns open
                orders for the requested instrument type.
            inst_type: OKX instrument type, usually `SWAP` for this project.

        Returns:
            Open order rows from OKX.
        """
        params = {"instType": inst_type}
        if inst_id:
            params["instId"] = inst_id
        data = self.request("GET", "/api/v5/trade/orders-pending", params, private=True)
        return list(data.get("data", []))

    def get_algo_orders(self, inst_id: str | None = None, order_type: str = "conditional") -> list[dict[str, Any]]:
        """Return open algo orders such as stop-loss and take-profit orders.

        Args:
            inst_id: Optional instrument id.
            order_type: OKX algo order type. `conditional` covers the attached
                TP/SL protection used by the live runner.

        Returns:
            Open algo order rows from OKX.
        """
        params = {"ordType": order_type}
        if inst_id:
            params["instId"] = inst_id
        data = self.request("GET", "/api/v5/trade/orders-algo-pending", params, private=True)
        return list(data.get("data", []))

    def get_instrument(self, inst_id: str, inst_type: str = "SWAP") -> dict[str, Any]:
        data = self.request("GET", "/api/v5/public/instruments", {"instType": inst_type, "instId": inst_id})
        rows = data.get("data", [])
        if not rows:
            raise RuntimeError(f"instrument not found: {inst_id}")
        return rows[0]

    def set_leverage(self, inst_id: str, leverage: int, td_mode: str = "isolated", mgn_ccy: str | None = None) -> dict[str, Any]:
        params = {"instId": inst_id, "lever": str(leverage), "mgnMode": td_mode}
        if mgn_ccy:
            params["mgnCcy"] = mgn_ccy
        return self.request("POST", "/api/v5/account/set-leverage", params, private=True)

    def place_market_order_with_protection(
        self,
        inst_id: str,
        td_mode: str,
        side: str,
        size: str,
        take_profit: float,
        stop_loss: float,
        client_order_id: str,
    ) -> dict[str, Any]:
        params = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": side,
            "ordType": "market",
            "sz": size,
            "clOrdId": client_order_id,
            "attachAlgoOrds": [
                {
                    "tpTriggerPx": str(take_profit),
                    "tpOrdPx": "-1",
                    "slTriggerPx": str(stop_loss),
                    "slOrdPx": "-1",
                }
            ],
        }
        return self.request("POST", "/api/v5/trade/order", params, private=True)


def plan_to_order_sizing(plan: TradePlan, inst_id: str, instrument: dict[str, Any]) -> OrderSizing:
    """Convert a base-asset research quantity into OKX swap contracts."""
    ct_val = float(instrument.get("ctVal", 1.0) or 1.0)
    lot_size = instrument.get("lotSz", "1")
    min_size = float(instrument.get("minSz", lot_size) or lot_size)
    raw_contracts = plan.qty / ct_val
    size = round_down_to_step(raw_contracts, lot_size)
    contracts = float(size)
    if contracts < min_size:
        raise ValueError(f"order size {contracts} contracts is below OKX minSz {min_size}")
    side = "buy" if plan.direction == "long" else "sell"
    notional = plan.entry * plan.qty
    return OrderSizing(inst_id=inst_id, side=side, size=size, contracts=contracts, notional_usdt=notional)


def validate_live_risk(plan: TradePlan, sizing: OrderSizing, limits: LiveRiskLimits) -> None:
    """Raise if a live order exceeds hard safety limits."""
    if plan.risk_usdt > limits.max_risk_usdt:
        raise ValueError(f"plan risk {plan.risk_usdt} exceeds max_risk_usdt {limits.max_risk_usdt}")
    if plan.margin > limits.max_margin_usdt:
        raise ValueError(f"plan margin {plan.margin} exceeds max_margin_usdt {limits.max_margin_usdt}")
    if sizing.notional_usdt > limits.max_notional_usdt:
        raise ValueError(f"plan notional {sizing.notional_usdt:.4f} exceeds max_notional_usdt {limits.max_notional_usdt}")
    if plan.leverage > limits.max_leverage:
        raise ValueError(f"plan leverage {plan.leverage} exceeds max_leverage {limits.max_leverage}")
