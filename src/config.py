"""Runtime configuration for the OKX trend-risk research and live runner.

This file is intentionally organized like a control panel. Each section groups
settings by the behavior they affect, with short English and Chinese comments.

本文件按“控制面板”方式整理。每个分区对应一类策略行为，并提供简短中英文注释。

Notes:
    Keep API secrets in `.env`; do not add them here.
    API 密钥只放在 `.env`，不要写入本文件。
"""


# =============================================================================
# 1. Runtime And Dashboard / 运行与看板
# =============================================================================

# Default instrument and bar used by one-symbol commands.
# 单币种命令的默认交易品种和K线周期。
INST_ID = "ETH-USDT-SWAP"
BAR = "15m"

# Local dashboard binding. Use 127.0.0.1 for local-only access; Cloudflare
# Tunnel exposes this local port to your phone.
# 本地看板监听地址。127.0.0.1 仅本机访问；Cloudflare Tunnel 会把端口映射到手机。
WEB_HOST = "127.0.0.1"
WEB_PORT = 8080
DASHBOARD_REFRESH_SECONDS = 5

# Live runner defaults. Real orders still require the command-line confirmation
# flag in `backtest.run_live_okx`.
# 实盘执行默认保护。真实下单仍需要命令行确认参数。
LIVE_DEFAULT_MAX_RISK_USDT = 2.0
LIVE_DEFAULT_MAX_MARGIN_USDT = 20.0
LIVE_DEFAULT_MAX_NOTIONAL_USDT = 100.0
LIVE_DEFAULT_MAX_LEVERAGE = 5
LIVE_LOOP_INTERVAL_SEC = 60
LIVE_LOOP_MAX_ORDERS_PER_CYCLE = 1
LIVE_LOOP_MAX_OPEN_POSITIONS = 2
LIVE_LOOP_COOLDOWN_MINUTES = 60


# =============================================================================
# 2. Universe And Instrument Profiles / 币种池与品种分层
# =============================================================================

# Core instruments get the normal research risk multiplier.
# 核心币种使用标准研究风险权重。
CORE_UNIVERSE = [
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP",
]

# Liquid alts are tradable but sized smaller than core instruments.
# 高流动性山寨币可交易，但仓位低于核心币种。
LIQUID_ALT_UNIVERSE = [
    "XRP-USDT-SWAP",
    "DOGE-USDT-SWAP",
    "BNB-USDT-SWAP",
    "ADA-USDT-SWAP",
    "LINK-USDT-SWAP",
    "HYPE-USDT-SWAP",
]

# Watch-list instruments are research candidates and receive the smallest risk.
# 观察池币种用于研究验证，默认风险最小。
WATCH_UNIVERSE = [
    "AVAX-USDT-SWAP",
    "LTC-USDT-SWAP",
    "SUI-USDT-SWAP",
    "TON-USDT-SWAP",
]

# Profile-level risk multipliers used by backtests and live sizing.
# 按币种分层调整资金风险。
CORE_RISK_MULTIPLIER = 1.0
LIQUID_ALT_RISK_MULTIPLIER = 0.75
WATCH_RISK_MULTIPLIER = 0.50

# Styles disabled in the legacy single-signal path. Agent-market mode may
# override this so the model can evaluate all agents.
# 旧单信号路径禁用的形态。Agent 市场模式可绕过，用于让模型评估所有 Agent。
DISABLED_STYLES_BY_PROFILE = {
    "core": ("range_reversion", "trend_breakout"),
    "liquid_alt": ("range_reversion", "trend_breakout"),
    "watch": ("range_reversion", "trend_breakout"),
    "unknown": ("range_reversion", "trend_breakout"),
}


# =============================================================================
# 3. Portfolio Selection And Walk-Forward / 组合选择与走步测试
# =============================================================================

# Symbol-score thresholds decide active/reduced/paused states.
# 币种评分阈值决定 active/reduced/paused 状态。
MIN_SYMBOL_SCORE_TRADES = 30
SYMBOL_PAUSE_RETURN_PCT = -8.0
SYMBOL_PAUSE_PROFIT_FACTOR = 0.72
SYMBOL_PAUSE_DRAWDOWN_PCT = 22.0
SYMBOL_ACTIVE_SCORE = 55.0
SYMBOL_REDUCED_SCORE = 35.0
SYMBOL_REDUCED_WEIGHT = 0.35

# Walk-forward split and market breadth gates.
# 走步训练/测试切分与全市场过滤。
WALK_FORWARD_TRAIN_FRACTION = 0.50
MIN_MARKET_BREADTH = 0.35
MIN_MARKET_AVG_MOMENTUM_PCT = -2.0
MIN_RELATIVE_STRENGTH_RANK = 0.50

# Momentum probes allow small exposure to strong symbols during weak filters.
# 动量试探允许强势币种在弱过滤环境中保留小仓位。
MIN_MOMENTUM_PROBE_RANK = 0.67
MIN_MOMENTUM_PROBE_RETURN_PCT = 5.0
MIN_MOMENTUM_PROBE_TRAIN_RETURN_PCT = -10.0
MOMENTUM_PROBE_WEIGHT = 0.25

# Rolling market gates are used inside longer walk-forward tests.
# 滚动市场门控用于更长周期的走步测试。
ROLLING_MARKET_LOOKBACK_BARS = 96 * 5
ROLLING_MIN_MARKET_BREADTH = 0.45
ROLLING_MIN_MARKET_AVG_MOMENTUM_PCT = -1.0
ROLLING_MIN_MARKET_AVG_ABS_MOMENTUM_PCT = 1.2


# =============================================================================
# 4. Account, Fees, Sizing, And Leverage / 账户、手续费、仓位与杠杆
# =============================================================================

# Research account equity. Live execution can override this with real balance.
# 研究回测本金；实盘可用真实账户权益覆盖。
INITIAL_EQUITY = 1000.0

# Exchange-cost assumptions used by backtests.
# 回测使用的交易成本假设。
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002

# Leverage ladder selected by setup quality.
# 按信号质量选择的杠杆档位。
LEVERAGE = 5
MIN_LEVERAGE = 1
PROBE_LEVERAGE = 2
RANGE_LEVERAGE = 3
NORMAL_TREND_LEVERAGE = 6
HIGH_CONVICTION_LEVERAGE = 10
MAX_LEVERAGE = 12

# Baseline risk controls before model multipliers are applied.
# 模型倍率生效前的基础风控。
RISK_PER_TRADE = 0.01
MAX_POSITION_MARGIN_RATIO = 0.25
MIN_CONFIDENCE = 0.55
MIN_REWARD_RISK = 1.4


# =============================================================================
# 5. Indicator Parameters / 技术指标参数
# =============================================================================

# Trend and channel features.
# 趋势与通道类特征。
EMA_FAST = 20
EMA_SLOW = 50
EMA_TREND = 100
ATR_PERIOD = 14
ADX_PERIOD = 14
DONCHIAN_PERIOD = 20
BOLL_PERIOD = 20
BOLL_STD = 2.0

# Regime thresholds for trend, volatility, and crash risk.
# 判断趋势、波动率和极端风险的阈值。
ADX_TREND_THRESHOLD = 22.0
ADX_STRONG_THRESHOLD = 30.0
ATR_HIGH_VOL_PERCENTILE = 0.80
ATR_LOW_VOL_PERCENTILE = 0.25
CRASH_ATR_MULT = 2.2


# =============================================================================
# 6. Entry, Exit, And Signal Geometry / 入场、出场与信号结构
# =============================================================================

# Pullback and breakout geometry.
# 回调与突破信号的结构参数。
PULLBACK_ATR_BAND = 0.8
BREAKOUT_BUFFER_ATR = 0.10

# R-based stop, target, and trailing settings.
# 基于R倍数的止损、止盈与移动止损。
STOP_ATR_MULT = 1.8
TRAIL_ATR_MULT = 3.2
TAKE_PROFIT_R = 4.0
PARTIAL_TAKE_PROFIT_R = 2.0
PARTIAL_EXIT_FRACTION = 0.30
BREAKEVEN_ARM_R = 1.0

# Range setups use tighter profit-taking than trend setups.
# 震荡反转使用更短的止盈距离。
RANGE_TAKE_PROFIT_R = 2.2
RANGE_PARTIAL_TAKE_PROFIT_R = 1.4


# =============================================================================
# 7. Quality Filters And Meta-Model Inputs / 信号质量过滤与模型输入
# =============================================================================

# Higher-timeframe confirmation used by quality filters.
# 更高周期确认。
HIGHER_TIMEFRAME = "1h"
REQUIRE_ALIGNED_BREAKOUT = True
MIN_BREAKOUT_CONFIDENCE = 0.78
PULLBACK_MIN_CONFIDENCE = 0.80

# Recent shock filter reduces or blocks entries after expansion candles.
# 近期冲击过滤：放量大波动后降低或阻止入场。
SHOCK_LOOKBACK_BARS = 8
SHOCK_ATR_MULT = 2.2
SHOCK_COOLDOWN_BARS = 6

# Quality-dependent risk profiles.
# 按信号质量划分的风险档位。
HIGH_CONVICTION_RISK = 0.022
NORMAL_TREND_RISK = 0.014
RANGE_RISK = 0.008
PROBE_RISK = 0.006
HIGH_CONVICTION_MARGIN_RATIO = 0.45
NORMAL_MARGIN_RATIO = 0.30
RANGE_MARGIN_RATIO = 0.18

# Failed-breakout filters protect against wick-only breakouts.
# 假突破过滤，避免只靠影线触发的劣质突破。
FAILED_BREAKOUT_WICK_RATIO = 0.55
FAILED_BREAKOUT_BODY_RECLAIM_RATIO = 0.45
MIN_EXPECTED_MOVE_R = 1.6


# =============================================================================
# 8. Pyramiding / 顺势加仓
# =============================================================================

# Pyramiding is only considered after the trade moves in favor of the plan.
# 加仓只在价格朝有利方向运行后考虑。
PYRAMID_ENABLED = True
PYRAMID_MAX_ADDS = 2
PYRAMID_TRIGGER_R = 2.2
PYRAMID_STEP_R = 1.0
PYRAMID_ADD_FRACTION = 0.35
PYRAMID_MIN_HIGHER_CONFIDENCE = 0.72
PYRAMID_MAX_TOTAL_MARGIN_RATIO = 0.70
