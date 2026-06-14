# Astra-Quant

Astra-Quant is a local OKX crypto futures research and execution system. It
combines multi-agent signal generation, LightGBM meta-label filtering, dynamic
risk control, walk-forward backtesting, guarded live execution, and a mobile
dashboard.

- Market-regime detection: trend, range, high-volatility chop, and crash risk.
- Multi-agent signal market: trend pullback, trend breakout, range reversion,
  and volatility breakout agents.
- Trade-plan generation: entry, invalidation, stop loss, take profit, and size.
- Meta filtering: Agent-aware LightGBM signal-quality scoring.
- Dynamic risk and leverage overlay: account risk, volatility risk, low-confidence blocking,
  and setup-specific leverage.
- Exit engine: breakeven arming, partial exits, and trend trailing stops.
- Offline OHLCV backtesting, rolling walk-forward tests, and OOS validation.
- Guarded OKX live loop with hard limits, cooldowns, position checks, and
  attached TP/SL protection.
- Local mobile-friendly dashboard with account, position, order, signal, and
  log-replay views.

The older `C:\okx` program is a Bollinger mean-reversion system. This project is
designed separately as the new Agent/LightGBM trend-risk framework.

> Warning: live trading is risky. Keep real API keys in `.env`, start with
> `start_live_loop_dry_run.bat`, and use small hard limits before enabling real
> orders.

## Quick Start

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Create `.env` from `.env.example` before using live account features:

```env
OKX_API_KEY=your-api-key
OKX_API_SECRET=your-api-secret
OKX_PASSPHRASE=your-passphrase
OKX_FLAG=0
```

Download public OKX candles:

```powershell
python -m backtest.download_okx_history --inst ETH-USDT-SWAP --bar 15m --days 60
```

Download the configured universe, including `HYPE-USDT-SWAP`:

```powershell
python -m backtest.download_universe_history --bar 15m --days 60 --no-watch
```

Run a V3 backtest:

```powershell
python -m backtest.run_v2_backtest --csv backtest\data\ETH-USDT-SWAP_15m_60d.csv
```

Run a multi-asset universe backtest:

```powershell
python -m backtest.run_universe_backtest --data-dir backtest\data --no-watch --json-out backtest\reports\universe_backtest.json
```

Run a stricter walk-forward universe test:

```powershell
python -m backtest.run_universe_backtest --data-dir backtest\data --no-watch --walk-forward --json-out backtest\reports\universe_walk_forward_recent_60d.json
```

## Local Dashboard And Phone Access

The local dashboard follows the older `C:\okx` workflow: start it from one batch
file, then open the Cloudflare Tunnel URL on your phone.

```powershell
.\start.bat
```

Local URL:

```text
http://127.0.0.1:8080
```

If Cloudflare Tunnel is installed at
`C:\Program Files (x86)\cloudflared\cloudflared.exe`, `start.bat` will start it
automatically and print a `trycloudflare.com` URL. Open that URL on your phone.

Useful helper scripts:

```powershell
.\run_tunnelflare_8080.bat
.\stop_tunnelflare_8080.bat
```

The dashboard is read-only. It shows account equity, available USDT, current
positions, regular orders, TP/SL algo orders, and the current Agent signal for
each configured trading symbol. Each signal card includes direction, Agent,
entry, stop loss, take profit, partial take profit, quantity, margin, risk,
leverage, Meta probability, and market regime. API secrets from `.env` are not
shown.

## Live OKX Execution

Live execution is intentionally separated from research backtests. API secrets
are read from `.env` or environment variables and are never stored in source
files. Create `.env` from `.env.example`:

```env
OKX_API_KEY=your-api-key
OKX_API_SECRET=your-api-secret
OKX_PASSPHRASE=your-passphrase
OKX_FLAG=0
```

Use `OKX_FLAG=1` for OKX simulated trading and `OKX_FLAG=0` for live trading.
Always run a dry run first:

```powershell
python -m backtest.run_live_okx --inst BTC-USDT-SWAP --dry-run --max-risk-usdt 2 --max-margin-usdt 20 --max-notional-usdt 100 --max-leverage 5
```

Send one guarded live order:

```powershell
python -m backtest.run_live_okx --inst BTC-USDT-SWAP --i-understand-live-risk --max-risk-usdt 2 --max-margin-usdt 20 --max-notional-usdt 100 --max-leverage 5
```

Run the automatic live loop in dry-run mode:

```powershell
.\start_live_loop_dry_run.bat
```

Run the automatic live loop with real orders:

```powershell
.\start_live_loop.bat
```

`start_live_loop.bat` now starts the auto-trading loop, local dashboard, and
Cloudflare Tunnel in one terminal. The terminal prints live-loop JSON records
and `[tunnel]` lines containing the `trycloudflare.com` phone URL. The dashboard
no longer focuses on raw reports; it shows account state, positions, orders,
current symbol signals, and a visual replay parsed from `logs/live_orders.jsonl`.

The live loop scans the configured core and liquid-alt universe, skips symbols
with existing positions or pending TP/SL orders, applies per-trade hard limits,
uses a per-symbol cooldown, and places at most one new order per scan cycle by
default. Stop it with `CTRL+C`.

The live runner pulls fresh candles, builds the current Agent/Meta signal,
rejects existing positions by default, validates hard risk limits, sets
instrument leverage, and places a market order with attached full-position
take-profit and stop-loss protection. It writes JSONL logs to
`logs/live_orders.jsonl`.

Train and apply the LightGBM meta-label model:

```powershell
python -m backtest.build_meta_dataset --data-dir backtest\data_train_2y --out backtest\datasets\meta_signals_2y_until_20260415.csv --end-date 2026-04-15T00:00:00Z
python -m backtest.train_meta_model --model-type lightgbm --dataset backtest\datasets\meta_signals_2y_until_20260415.csv --model-out backtest\models\meta_model_lgbm_2y_until_20260415.json --threshold 0.60
python -m backtest.run_universe_backtest --data-dir backtest\data --no-watch --meta-model backtest\models\meta_model_lgbm_2y_until_20260415.json
```

The richer meta-label dataset includes momentum, realized volatility, ATR rank,
volume z-score, MFE/MAE, partial-profit outcomes, and `quality_label`:

```powershell
python -m backtest.build_meta_dataset --data-dir backtest\data_train_2y --out backtest\datasets\meta_signals_2y_until_20260415_rich.csv --end-date 2026-04-15T00:00:00Z
python -m backtest.train_meta_model --model-type lightgbm --target-column quality_label --dataset backtest\datasets\meta_signals_2y_until_20260415_rich.csv --model-out backtest\models\meta_model_lgbm_2y_until_20260415_rich_quality.json --threshold 0.55
```

Build and test the Agent-identity LightGBM meta model:

```powershell
python -m backtest.build_meta_dataset --data-dir backtest\data_train_2y --out backtest\datasets\meta_signals_2y_until_20260415_agent.csv --end-date 2026-04-15T00:00:00Z --stride 4 --agent-market
python -m backtest.train_meta_model --model-type lightgbm --target-column quality_label --dataset backtest\datasets\meta_signals_2y_until_20260415_agent.csv --model-out backtest\models\meta_model_lgbm_2y_until_20260415_agent_quality_t050.json --threshold 0.50
python -m backtest.run_universe_backtest --data-dir backtest\data --no-watch --walk-forward --agent-market --meta-model backtest\models\meta_model_lgbm_2y_until_20260415_agent_quality_t050.json
```

This model adds one-hot Agent identity features for `trend_pullback_agent`,
`trend_breakout_agent`, `range_reversion_agent`, and `volatility_breakout_agent`.
On the current test files, the `quality_label` threshold `0.50` version produced
`+14.73%` on the recent 60-day walk-forward test and `+1.18%` on the
`2026-02-14` to `2026-04-15` OOS set, with higher drawdown than the conservative
meta versions.

Train and apply the future-regime model:

```powershell
python -m backtest.build_regime_dataset --data-dir backtest\data_train_2y --out backtest\datasets\regime_2y_until_20260415_h1d.csv --end-date 2026-04-15T00:00:00Z --horizon-bars 96 --stride 48
python -m backtest.train_regime_model --dataset backtest\datasets\regime_2y_until_20260415_h1d.csv --model-out backtest\models\regime_model_lgbm_2y_until_20260415_h1d.json --threshold 0.35
python -m backtest.run_universe_backtest --data-dir backtest\data --no-watch --meta-model backtest\models\meta_model_lgbm_2y_until_20260415.json --regime-model backtest\models\regime_model_lgbm_2y_until_20260415_h1d.json
```

The universe report includes three portfolio views:

- `equal_weight_1000u_return_pct`: every tested symbol receives the same 1000U allocation share.
- `adaptive_1000u_return_pct`: symbols are scored from recent return, profit factor, win rate, and drawdown, then marked `active`, `reduced`, or `paused`.
- `walk_forward_1000u_return_pct`: the first half of the data chooses the market state and symbols, then the second half is used as the test window.
- `independent_account_return_pct_on_1000u`: research-only sum where each symbol is treated as if it had its own 1000U account.

## Method

The strategy separates decisions into four modules:

1. `market_state.py` scores the current market regime.
2. `multi_timeframe.py` builds higher-timeframe confirmation.
3. `signal_engine.py` creates a directional setup only when the regime allows it.
4. `meta_filter.py` decides whether the setup deserves normal, high, or probe risk.
5. `leverage_engine.py` selects leverage from setup quality and market risk.
6. `trade_plan.py` converts the setup into entry, stop, take-profit, leverage, and size.
7. `exit_engine.py` manages partial exits, breakeven, and trailing stops.
8. `risk_engine.py` does final plan validation before simulation.
9. `symbol_score.py` pauses or reduces symbols whose recent fit is weak.
10. `walk_forward.py` applies market breadth and relative-strength selection before the test window.
11. `meta_model.py` loads the LightGBM meta-label model used to filter candidate entries.
12. `regime_model.py` loads a LightGBM future-shape model used as a conservative market-risk filter.

## Dynamic Leverage

Leverage is no longer a single global value for every trade. `LEVERAGE = 5`
remains the neutral fallback, while accepted setups can choose:

- Probe / failed-breakout risk: `2x`
- Range reversion: `3x`
- Normal trend: around `6x`
- High-conviction trend breakout: up to `10x`

The plan still sizes by risk first. Leverage only changes the maximum notional
capacity and margin estimate. Before live trading, add exchange-level liquidation
distance checks against OKX instrument metadata and account mode.

## Multi-Asset Universe

The initial multi-asset research universe is grouped by liquidity and risk:

```python
CORE_UNIVERSE = [
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP",
]

LIQUID_ALT_UNIVERSE = [
    "XRP-USDT-SWAP",
    "DOGE-USDT-SWAP",
    "BNB-USDT-SWAP",
    "ADA-USDT-SWAP",
    "LINK-USDT-SWAP",
    "HYPE-USDT-SWAP",
]

WATCH_UNIVERSE = [
    "AVAX-USDT-SWAP",
    "LTC-USDT-SWAP",
    "SUI-USDT-SWAP",
    "TON-USDT-SWAP",
]
```

`HYPE-USDT-SWAP` is included in `LIQUID_ALT_UNIVERSE`, so it should receive
alt-coin sizing and portfolio risk limits rather than core BTC/ETH/SOL sizing.

The universe tooling is research-only:

- `download_universe_history.py` downloads one CSV per configured instrument.
- `run_universe_backtest.py` runs each available CSV and aggregates results.
- `universe_engine.py` maps each instrument to `core`, `liquid_alt`, or `watch`.
- `symbol_score.py` labels instruments as `active`, `reduced`, or `paused` for
  adaptive allocation.
- `walk_forward.py` prevents same-window selection bias by selecting symbols on
  the training half and measuring results on the later test half.
- Liquid-alt symbols such as `HYPE-USDT-SWAP` use a lower risk multiplier than
  core BTC/ETH/SOL symbols.

This makes the strategy easier to test: trend detection can improve without
touching order execution, and risk rules can be tightened without rewriting the
signal logic.

## Safety

This is research software only. Futures trading with leverage can lose money
quickly. Backtest results are not a guarantee of future performance.
