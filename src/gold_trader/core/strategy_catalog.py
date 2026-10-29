"""
Aegis Alpha — Quantitative Strategy Catalog & Architecture Repository.

This module serves as the central persistent strategy vault for the autonomous trading engine,
storing all production, staged, benchmarked, and research strategies along with their:
- Exact technical entry/exit algorithms & parameters
- Two-stage risk execution policies
- Machine learning meta-labeling conviction thresholds
- 3-Year & 5-Year audited performance metrics (Win Rate, Profit Factor, Net P&L, Max DD)
- Microstructure rules, session windows, and operational runbook commands.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class StrategyStatus(str, Enum):
    ACTIVE_PRODUCTION = "ACTIVE_PRODUCTION"        # Live execution on Model 1 (Demo/Prod)
    ACTIVE_YIELD_A = "ACTIVE_YIELD_A"              # Live execution on Model 2 (Yield A)
    BENCHMARKED_READY = "BENCHMARKED_READY"        # >54%–56% WR verified, ready for deployment
    STAGED_FOR_ACTIVATION = "STAGED_FOR_ACTIVATION"# Formatted and staged for explicit user launch
    RESEARCH_BENCHMARKED = "RESEARCH_BENCHMARKED"  # Benchmarked with historical datasets
    DISQUALIFIED = "DISQUALIFIED"                  # Mathematically invalid or negative drift


class RiskArchetype(str, Enum):
    FLAT_RISK = "FLAT_RISK"
    TIGHT_TRAIL_HARVEST = "TIGHT_TRAIL_HARVEST"
    NOISE_ADAPTIVE_LADDER = "NOISE_ADAPTIVE_LADDER"
    FIXED_FRACTIONAL = "FIXED_FRACTIONAL"
    SINGLE_STAGE_TARGET = "SINGLE_STAGE_TARGET"


@dataclass
class StrategyProfile:
    id: str
    name: str
    short_desc: str
    category: str
    status: StrategyStatus
    risk_archetype: RiskArchetype
    supported_assets: List[str]
    timeframe: str
    default_risk_usd: float
    target_win_rate_pct: float

    # Core Technical Execution Rules
    technical_rules: Dict[str, Any] = field(default_factory=dict)

    # Machine Learning & Gatekeeper Parameters
    ml_parameters: Dict[str, Any] = field(default_factory=dict)

    # Historical Empirical Benchmark Metrics (Audited 3Y & 5Y)
    benchmark_metrics: Dict[str, Any] = field(default_factory=dict)

    # Activation CLI Command
    cli_activation_command: str = ""

    # Operational Notes & Risk Warnings
    notes_and_caveats: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["risk_archetype"] = self.risk_archetype.value
        return d


# ==============================================================================
# STRATEGY REPOSITORY DEFINITIONS
# ==============================================================================

STRATEGY_CATALOG: Dict[str, StrategyProfile] = {
    # --------------------------------------------------------------------------
    # 1. Model 1 Production Baseline (Flat Risk Touch)
    # --------------------------------------------------------------------------
    "prod_flat_risk_touch": StrategyProfile(
        id="prod_flat_risk_touch",
        name="Production High-Velocity Donchian Touch (Model 1 Baseline)",
        short_desc="1H Donchian 10 Touch + SuperTrend (10, 3.0) + 50 EMA trend filter with two-stage 1.25R scale-out and dynamic trailing runner.",
        category="Breakout & Momentum",
        status=StrategyStatus.ACTIVE_PRODUCTION,
        risk_archetype=RiskArchetype.FLAT_RISK,
        supported_assets=["XAUUSD", "BTCUSD", "ETHUSD"],
        timeframe="1h",
        default_risk_usd=75.44,
        target_win_rate_pct=52.0,
        technical_rules={
            "trend_filter": "50 EMA (BUY if Close > 50 EMA with non-negative slope; SELL if Close < 50 EMA with non-positive slope)",
            "entry_trigger": "Intrabar touch of previous 10-bar Donchian channel (High >= Upper Band or Low <= Lower Band)",
            "confirmation": "SuperTrend (10, 3.0) direction alignment + Rejection wick guard: wick <= 1.8 * max(body, 0.15*ATR)",
            "initial_stop_loss": "Dynamic ATR: max(1.5 * ATR_1h, min_sl) where min_sl = $8.00 Gold, $500.00 BTC, $20.00 ETH",
            "stage_1_target": "1.25R (+50% volume scaled out into realized cash)",
            "break_even_lock": "Stop-loss shifts to Entry + offset (+$0.50 Gold, +$30.00 BTC, +$2.00 ETH) at Stage 1 target",
            "stage_2_runner": "Dynamic trailing stop behind peak excursion (Buffer: $4.50 Gold, $350.00 BTC, $10.00 ETH) to 3.0R TP2",
            "session_window": "07:00–19:00 UTC (Gold only); 24/7/365 (Bitcoin and Ethereum)",
            "concurrency": "Uncapped (Paced by 1-trade-per-bar lock; trailing runners run concurrently)",
        },
        ml_parameters={
            "model_type": "LightGBM Purged Triple-Barrier Meta-Labeler",
            "min_probability": 0.44,
            "feature_count": 20,
            "horizon_bars": 48,
        },
        benchmark_metrics={
            "3_year_results": {
                "XAUUSD": {"win_rate_pct": 52.2, "profit_factor": 1.15, "payoff_ratio": 1.06, "net_pnl_usd": 12784.45, "max_dd_usd": 4337.80, "trades": 2278},
                "BTCUSD": {"win_rate_pct": 50.8, "profit_factor": 1.10, "payoff_ratio": 1.07, "net_pnl_usd": 17931.95, "max_dd_usd": 3568.99, "trades": 4836},
                "ETHUSD": {"win_rate_pct": 52.0, "profit_factor": 1.28, "payoff_ratio": 1.18, "net_pnl_usd": 46223.20, "max_dd_usd": 3188.00, "trades": 4448},
            },
            "5_year_portfolio": {
                "total_trades": 11434,
                "net_pnl_usd": 83435.40,
                "blended_sharpe": 1.84,
                "psr_pct": 99.8,
            },
        },
        cli_activation_command="bash scripts/start_auto_trader.sh 60 1.25 10.0 1h COMPOUNDING_TOUCH",
        notes_and_caveats="Two-stage execution is mathematically vital. Never remove the 50% scale-out at +1.0R–1.25R, as it guarantees that trades advancing past 1.0R become net winners even if the runner pulls back to break-even.",
    ),

    # --------------------------------------------------------------------------
    # 2. Model 2 Profile A Yield Engine (YieldA)
    # --------------------------------------------------------------------------
    "profile_a_yield_engine": StrategyProfile(
        id="profile_a_yield_engine",
        name="Profile A Crypto Yield Engine (Model 2 YieldA)",
        short_desc="High-precision institutional crypto engine with tight trailing stops and strict Max 1 Concurrency for monthly treasury surplus sweeps.",
        category="Institutional Yield & Treasury",
        status=StrategyStatus.ACTIVE_YIELD_A,
        risk_archetype=RiskArchetype.TIGHT_TRAIL_HARVEST,
        supported_assets=["BTCUSD", "ETHUSD"],
        timeframe="1h",
        default_risk_usd=75.00,
        target_win_rate_pct=54.5,
        technical_rules={
            "trend_filter": "50 EMA slope alignment",
            "entry_trigger": "1H Donchian 10 Intrabar Touch with wholesale discount pricing",
            "confirmation": "SuperTrend (10, 3.0) + candle color alignment",
            "initial_stop_loss": "Dynamic ATR: max(1.5 * ATR_1h, min_sl) (Min: $500 BTC, $20 ETH)",
            "stage_1_target": "1.00R (+50% scale-out)",
            "break_even_lock": "BE + $30.00 BTC / BE + $2.00 ETH",
            "stage_2_runner": "Tight Trailing Stop: $200.00 (BTC) / $6.00 (ETH) locking runner profits before mean-reversion",
            "concurrency": "Strict Max 1 Concurrent Position per asset (re-entry locked until active position exits)",
            "treasury_harvest": "$1,500.00 Capital Baseline Anchor. Month-end surplus swept to treasury.",
        },
        ml_parameters={
            "model_type": "LightGBM Meta-Labeler",
            "min_probability": 0.48,
            "feature_count": 20,
        },
        benchmark_metrics={
            "3_year_results": {
                "BTCUSD": {"win_rate_pct": 54.0, "profit_factor": 1.34, "payoff_ratio": 1.14, "net_pnl_usd": 20329.85, "max_dd_usd": 366.81, "trades": 2584},
                "ETHUSD": {"win_rate_pct": 55.2, "profit_factor": 1.49, "payoff_ratio": 1.21, "net_pnl_usd": 50658.76, "max_dd_usd": 980.45, "trades": 2976},
            },
        },
        cli_activation_command="bash scripts/start_yield_a_engines.sh",
        notes_and_caveats="Achieved lowest max drawdown across all tested crypto architectures (-$366.81 on BTC = 5.1% max drawdown) due to strict single-position concurrency.",
    ),

    # --------------------------------------------------------------------------
    # 3. High-Conviction ML Meta-Labeled Breakout Engine (>55% WR Candidate)
    # --------------------------------------------------------------------------
    "high_conviction_ml_touch": StrategyProfile(
        id="high_conviction_ml_touch",
        name="High-Conviction ML Breakout Engine (Tight TSL)",
        short_desc="Elevates ML conviction threshold to P >= 0.52–0.55 paired with tightened trailing stops to achieve >55% win rate and 50% drawdown reduction.",
        category="Machine Learning Meta-Labeling",
        status=StrategyStatus.BENCHMARKED_READY,
        risk_archetype=RiskArchetype.FLAT_RISK,
        supported_assets=["ETHUSD", "BTCUSD", "XAUUSD"],
        timeframe="1h",
        default_risk_usd=75.44,
        target_win_rate_pct=55.4,
        technical_rules={
            "trend_filter": "50 EMA (BUY Close > 50 EMA, SELL Close < 50 EMA)",
            "entry_trigger": "1H Donchian 10 Touch + SuperTrend (10, 3.0)",
            "confirmation": "ADX(14) >= 22.0 (suppresses horizontal chop) + Rejection wick guard",
            "initial_stop_loss": "max(1.5 * ATR_1h, min_sl)",
            "stage_1_target": "1.00R (+50% scale-out)",
            "break_even_lock": "BE + offset",
            "stage_2_runner": "Tight Trailing Stop: $6.00 ETH / $200.00 BTC / $3.00 Gold",
            "concurrency": "Uncapped or Max 1 Position",
        },
        ml_parameters={
            "model_type": "LightGBM Purged Triple-Barrier Meta-Labeler",
            "min_probability": 0.55,
            "feature_count": 20,
        },
        benchmark_metrics={
            "3_year_results": {
                "ETHUSD (ML >= 0.55)": {"win_rate_pct": 55.4, "profit_factor": 1.53, "payoff_ratio": 1.23, "net_pnl_usd": 20638.50, "max_dd_usd": 1508.77, "trades": 1115},
                "ETHUSD (ML >= 0.52)": {"win_rate_pct": 54.8, "profit_factor": 1.49, "payoff_ratio": 1.23, "net_pnl_usd": 34162.55, "max_dd_usd": 2867.39, "trades": 1993},
                "BTCUSD (ML >= 0.52)": {"win_rate_pct": 51.6, "profit_factor": 1.23, "payoff_ratio": 1.15, "net_pnl_usd": 16733.91, "max_dd_usd": 2708.01, "trades": 1964},
                "XAUUSD (ML >= 0.52)": {"win_rate_pct": 51.4, "profit_factor": 1.17, "payoff_ratio": 1.10, "net_pnl_usd": 7846.11, "max_dd_usd": 3185.02, "trades": 1193},
            },
        },
        cli_activation_command="python3 -m gold_trader.core.auto_trader --symbol ETHUSD --strategy COMPOUNDING_TOUCH --ml-min-prob 0.55 --trailing-buffer 6.0",
        notes_and_caveats="Surpasses 55% win rate on Ethereum. In exchange for higher win rate and half the drawdown, trade frequency is reduced by ~50% compared to baseline.",
    ),

    # --------------------------------------------------------------------------
    # 4. London / New York Overlap Session Filter Engine (Gold Edge)
    # --------------------------------------------------------------------------
    "gold_london_ny_overlap": StrategyProfile(
        id="gold_london_ny_overlap",
        name="London / NY Overlap Session Filter Engine (Spot Gold)",
        short_desc="Restricts Spot Gold breakout execution exclusively to 12:00–16:00 UTC, eliminating Asian session false-breakout chop.",
        category="Session Liquidity",
        status=StrategyStatus.BENCHMARKED_READY,
        risk_archetype=RiskArchetype.FLAT_RISK,
        supported_assets=["XAUUSD"],
        timeframe="1h",
        default_risk_usd=75.44,
        target_win_rate_pct=53.7,
        technical_rules={
            "session_filter": "Strictly 12:00 – 16:00 UTC (London PM / New York AM overlap)",
            "trend_filter": "50 EMA (Price > 50 EMA for BUY; Price < 50 EMA for SELL)",
            "entry_trigger": "1H Donchian 10 Touch + SuperTrend (10, 3.0)",
            "initial_stop_loss": "max(1.5 * ATR_1h, $8.00)",
            "stage_1_target": "1.00R (+50% volume closed)",
            "break_even_lock": "BE + $0.50 offset",
            "stage_2_runner": "Tight Trailing Stop: $3.00 behind peak excursion",
        },
        ml_parameters={
            "model_type": "LightGBM Meta-Labeler",
            "min_probability": 0.44,
        },
        benchmark_metrics={
            "3_year_results": {
                "XAUUSD (12-16 UTC, TSL $3.00)": {"win_rate_pct": 53.7, "profit_factor": 1.31, "payoff_ratio": 1.14, "net_pnl_usd": 8852.80, "max_dd_usd": 1842.15, "trades": 773},
            },
        },
        cli_activation_command="python3 -m gold_trader.core.auto_trader --symbol XAUUSD --strategy COMPOUNDING_TOUCH --session-start 12 --session-end 16 --trailing-buffer 3.0",
        notes_and_caveats="65% of daily directional volatility on Gold occurs during the London/NY overlap. Fades Asian chop and boosts Gold win rate to 53.7% with a 1.31 Profit Factor.",
    ),

    # --------------------------------------------------------------------------
    # 5. Staged Fast-Track Noise-Adaptive Compounding Ladder ($100 ➔ $5,337)
    # --------------------------------------------------------------------------
    "noise_adaptive_ladder_staged": StrategyProfile(
        id="noise_adaptive_ladder_staged",
        name="37-Step Noise-Adaptive Compounding Ladder ($100 ➔ $5,337)",
        short_desc="37-step noise-immune exponential compounding matrix with tiered ratchet floors ($150, $350, $700, $1,800, $3,000).",
        category="Compounding Progression",
        status=StrategyStatus.STAGED_FOR_ACTIVATION,
        risk_archetype=RiskArchetype.NOISE_ADAPTIVE_LADDER,
        supported_assets=["XAUUSD"],
        timeframe="1h",
        default_risk_usd=10.00,  # Starting level 1 risk
        target_win_rate_pct=53.0,
        technical_rules={
            "starting_seed": "$100.00 (survives 11 consecutive losses on 0.01 micro-lots)",
            "entry_engine": "1H Donchian 10 Touch + SuperTrend (10, 3.0) + 50 EMA",
            "stop_loss": "Dynamic ATR: max(1.5 * ATR_1h, $8.00)",
            "position_sizing": "Strict Planned Risk USD / (SL Distance * 100); never over-risks in high volatility",
            "phased_sprints": "Sprint ($100–$500: 10% risk) -> Expansion ($500–$2,000: 8% risk) -> Preservation ($2,000–$5,337: 5% risk)",
            "ratchet_floors": "Balance >= $250 -> Floor $150 | >= $500 -> Floor $350 | >= $1,000 -> Floor $700 | >= $2,500 -> Floor $1,800 | >= $4,000 -> Floor $3,000",
        },
        ml_parameters={
            "model_type": "LightGBM Meta-Labeler",
            "min_probability": 0.44,
        },
        benchmark_metrics={
            "monte_carlo_audit": {
                "simulations": 60000,
                "completion_rate_pct": 100.0,
                "median_trades_to_target": 332,
                "median_time_to_target": "2.2 months",
            },
        },
        cli_activation_command="bash scripts/start_auto_trader.sh 60 2.2 8.0 1h COMPOUNDING_HYBRID",
        notes_and_caveats="STAGED & READY. Completely eliminates the $20 seed bust trap. Automated ratchet floors prevent drawdowns from cascading back to level 1.",
    ),

    # --------------------------------------------------------------------------
    # 6. Single-Stage 1.0R Fast Target (No Runner Risk)
    # --------------------------------------------------------------------------
    "fast_1r_no_runner": StrategyProfile(
        id="fast_1r_no_runner",
        name="Single-Stage 1.0R Fast Target (No Runner)",
        short_desc="Closes 100% position volume at 1.00R take-profit. Tested to see if removing runner uncertainty improves expectancy.",
        category="Quick Target / Scalping",
        status=StrategyStatus.RESEARCH_BENCHMARKED,
        risk_archetype=RiskArchetype.SINGLE_STAGE_TARGET,
        supported_assets=["XAUUSD", "BTCUSD", "ETHUSD"],
        timeframe="1h",
        default_risk_usd=75.44,
        target_win_rate_pct=52.0,
        technical_rules={
            "entry_trigger": "1H Donchian 10 Touch + 50 EMA trend",
            "initial_stop_loss": "max(1.5 * ATR_1h, min_sl)",
            "take_profit": "1.00R (100% closed, zero runners)",
            "runner_policy": "None (No trailing stop)",
        },
        ml_parameters={
            "min_probability": 0.48,
        },
        benchmark_metrics={
            "3_year_results": {
                "XAUUSD": {"win_rate_pct": 51.8, "profit_factor": 0.97, "payoff_ratio": 0.90, "net_pnl_usd": -2372.33, "max_dd_usd": 6800.37, "trades": 1828},
                "BTCUSD": {"win_rate_pct": 51.8, "profit_factor": 1.00, "payoff_ratio": 0.93, "net_pnl_usd": 507.24, "max_dd_usd": 5795.70, "trades": 3694},
                "ETHUSD": {"win_rate_pct": 52.2, "profit_factor": 1.02, "payoff_ratio": 0.93, "net_pnl_usd": 2025.87, "max_dd_usd": 6005.26, "trades": 3433},
            },
        },
        cli_activation_command="python3 -m gold_trader.core.auto_trader --single-stage --tp-r 1.0",
        notes_and_caveats="DISQUALIFIED FOR LIVE PRODUCTION. While raw hit rate is ~52%, broker spread, slippage, and overnight swaps erode the 1.0R win into 0.90R–0.93R, causing account stagnation or bleed.",
    ),

    # --------------------------------------------------------------------------
    # 7. Smart Money Concepts (SMC) Asian Liquidity Sweep (Judas Swing)
    # --------------------------------------------------------------------------
    "smc_asian_sweep_judas": StrategyProfile(
        id="smc_asian_sweep_judas",
        name="Smart Money Concepts (SMC) Asian Liquidity Sweep",
        short_desc="Fades retail breakout traps outside the 00:00–06:00 UTC Asian session range into the 1H macro trend.",
        category="Smart Money Concepts",
        status=StrategyStatus.RESEARCH_BENCHMARKED,
        risk_archetype=RiskArchetype.FLAT_RISK,
        supported_assets=["XAUUSD", "BTCUSD"],
        timeframe="1h",
        default_risk_usd=75.44,
        target_win_rate_pct=48.0,
        technical_rules={
            "session_setup": "Marks Asian session (00:00–06:00 UTC) High and Low",
            "entry_trigger": "Price sweeps outside Asian high/low during London killzone (07:00–10:00 UTC) but candle closes with rejection wick back inside",
            "trend_confluence": "Must align with 50 EMA (Bearish sweep only if below 50 EMA; Bullish sweep only if above 50 EMA)",
            "stop_loss": "Placed 0.2 * ATR beyond the sweep wick",
            "target": "1.50R target back toward opposite session range",
        },
        ml_parameters={
            "min_probability": 0.44,
        },
        benchmark_metrics={
            "3_year_results": {
                "XAUUSD": {"win_rate_pct": 48.5, "profit_factor": 0.89, "payoff_ratio": 0.95, "net_pnl_usd": -142.10, "max_dd_usd": 503.84, "trades": 33},
                "BTCUSD": {"win_rate_pct": 44.4, "profit_factor": 0.76, "payoff_ratio": 0.95, "net_pnl_usd": -744.30, "max_dd_usd": 939.51, "trades": 72},
            },
        },
        cli_activation_command="python3 scripts/test_advanced_strategies.py",
        notes_and_caveats="Very low trade frequency (only ~1 trade/month). In modern algorithmic crypto and gold markets, false breakout sweeps frequently turn into explosive continuation trends rather than reversing.",
    ),

    # --------------------------------------------------------------------------
    # 8. Dynamic EMA 9/21 Value Zone Pullback
    # --------------------------------------------------------------------------
    "ema_value_zone_pullback": StrategyProfile(
        id="ema_value_zone_pullback",
        name="EMA 9/21 Dynamic Value Zone Pullbacks",
        short_desc="Enters wholesale limit pullbacks between EMA 9 and EMA 21 in the direction of the 50/200 EMA macro trend.",
        category="Trend Pullback & Wholesale Pricing",
        status=StrategyStatus.DISQUALIFIED,
        risk_archetype=RiskArchetype.FLAT_RISK,
        supported_assets=["XAUUSD", "BTCUSD", "ETHUSD"],
        timeframe="1h",
        default_risk_usd=75.44,
        target_win_rate_pct=45.0,
        technical_rules={
            "trend_filter": "EMA 9 > EMA 21 > EMA 50 for BUY; EMA 9 < EMA 21 < EMA 50 for SELL",
            "entry_trigger": "Price retraces into the [EMA 21, EMA 9] value zone and forms an engulfing confirmation candle",
            "stop_loss": "Dynamic ATR: max(1.5 * ATR_1h, min_sl)",
            "target": "Two-stage 1.25R scale-out and 3.0R runner",
        },
        ml_parameters={
            "min_probability": 0.44,
        },
        benchmark_metrics={
            "3_year_results": {
                "XAUUSD": {"win_rate_pct": 45.6, "profit_factor": 0.87, "payoff_ratio": 1.04, "net_pnl_usd": -5246.89, "max_dd_usd": 7295.08, "trades": 912},
                "BTCUSD": {"win_rate_pct": 44.9, "profit_factor": 0.84, "payoff_ratio": 1.03, "net_pnl_usd": -16279.88, "max_dd_usd": 18173.30, "trades": 2368},
                "ETHUSD": {"win_rate_pct": 44.8, "profit_factor": 0.90, "payoff_ratio": 1.11, "net_pnl_usd": -11523.21, "max_dd_usd": 15722.53, "trades": 2718},
            },
        },
        cli_activation_command="python3 scripts/backtest_ema_crossover.py",
        notes_and_caveats="DISQUALIFIED. Tested across 3 years and failed on all 3 assets (win rate 44.8%–45.6%, net loss -$5k to -$16k). Pullbacks into the moving average frequently represent the first leg of a macro trend reversal rather than a continuation.",
    ),

    # --------------------------------------------------------------------------
    # 9. SuperTrend (10, 3.0) + 200 EMA Macro Trend
    # --------------------------------------------------------------------------
    "supertrend_200ema_swing": StrategyProfile(
        id="supertrend_200ema_swing",
        name="SuperTrend (10, 3.0) + 200 EMA Swing Continuation",
        short_desc="Enters exclusively on fresh SuperTrend trend flips aligned with 200 EMA macro filter. Lowest drawdown among non-ML systems.",
        category="Macro Trend Following",
        status=StrategyStatus.RESEARCH_BENCHMARKED,
        risk_archetype=RiskArchetype.FLAT_RISK,
        supported_assets=["XAUUSD", "BTCUSD"],
        timeframe="1h",
        default_risk_usd=75.44,
        target_win_rate_pct=53.0,
        technical_rules={
            "macro_filter": "200 EMA (Price > 200 EMA for BUY; Price < 200 EMA for SELL)",
            "entry_trigger": "SuperTrend (10, 3.0) flips direction on the closed bar in alignment with 200 EMA",
            "stop_loss": "Placed at SuperTrend stop line or 1.5 * ATR_1h",
            "target": "1.20R fast target",
        },
        ml_parameters={},
        benchmark_metrics={
            "6_month_gold_benchmark": {
                "win_rate_pct": 52.9,
                "profit_factor": 1.33,
                "max_drawdown_r": 3.85,
                "note": "Lowest drawdown of all tested non-ML systems (< 4.5%). Zero whipsaw during horizontal chop.",
            },
        },
        cli_activation_command="python3 scripts/tournament_6mo.py",
        notes_and_caveats="Excellent capital preservation, but lower trade frequency (~8–10 trades/month). Best suited as a low-frequency conservative filter.",
    ),
}


# ==============================================================================
# STRATEGY CATALOG ACCESSORS & HELPER FUNCTIONS
# ==============================================================================

def get_strategy_catalog() -> Dict[str, StrategyProfile]:
    """Return the entire dictionary of stored quantitative strategies."""
    return STRATEGY_CATALOG


def get_strategy(strategy_id: str) -> Optional[StrategyProfile]:
    """Retrieve a strategy profile by its ID."""
    return STRATEGY_CATALOG.get(strategy_id)


def list_strategies(
    category: Optional[str] = None,
    status: Optional[StrategyStatus] = None,
    asset: Optional[str] = None,
    min_win_rate: Optional[float] = None,
) -> List[StrategyProfile]:
    """Filter and return strategy profiles based on criteria."""
    results = []
    for strat in STRATEGY_CATALOG.values():
        if category and strat.category.lower() != category.lower():
            continue
        if status and strat.status != status:
            continue
        if asset and asset.upper() not in [a.upper() for a in strat.supported_assets]:
            continue
        if min_win_rate is not None and strat.target_win_rate_pct < min_win_rate:
            continue
        results.append(strat)
    return results


def get_active_strategies() -> List[StrategyProfile]:
    """Return all strategies currently running in live production."""
    return [
        s for s in STRATEGY_CATALOG.values()
        if s.status in (StrategyStatus.ACTIVE_PRODUCTION, StrategyStatus.ACTIVE_YIELD_A)
    ]


def get_high_winrate_candidates(min_win_rate: float = 54.0) -> List[StrategyProfile]:
    """Return strategies meeting or exceeding the high win-rate target threshold."""
    return [
        s for s in STRATEGY_CATALOG.values()
        if s.target_win_rate_pct >= min_win_rate and s.status != StrategyStatus.DISQUALIFIED
    ]


def export_catalog_json() -> str:
    """Export the entire strategy catalog as formatted JSON."""
    data = {k: v.to_dict() for k, v in STRATEGY_CATALOG.items()}
    return json.dumps(data, indent=2)


def get_catalog_summary_table() -> List[Dict[str, Any]]:
    """Return a flat table format suitable for dataframes and UI rendering."""
    rows = []
    for s in STRATEGY_CATALOG.values():
        rows.append({
            "id": s.id,
            "name": s.name,
            "category": s.category,
            "status": s.status.value,
            "assets": ", ".join(s.supported_assets),
            "timeframe": s.timeframe,
            "risk_model": s.risk_archetype.value,
            "risk_usd": f"${s.default_risk_usd:.2f}",
            "target_win_rate": f"{s.target_win_rate_pct:.1f}%",
            "cli_command": s.cli_activation_command,
        })
    return rows
