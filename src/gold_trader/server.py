"""
FastMCP Server for the Institutional Gold Trading System.
Exposes tools for Live Market Data, Technicals, Fibonacci Golden Pocket,
Macro Confluence, Multi-Agent Debate, Strategy Backtesting, and XM/Paper Broker Execution.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Literal, Optional

try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP
    except (ImportError, ModuleNotFoundError):
        class FastMCP:  # type: ignore
            def __init__(self, name: str = "", instructions: str = "", **kwargs: Any):
                self.name = name
                self.instructions = instructions
                self.settings = type("Settings", (), {"host": "127.0.0.1", "port": 8001})()

            def tool(self, *args: Any, **kwargs: Any):
                def decorator(fn: Any) -> Any:
                    return fn
                return decorator

            def resource(self, *args: Any, **kwargs: Any):
                def decorator(fn: Any) -> Any:
                    return fn
                return decorator

            def prompt(self, *args: Any, **kwargs: Any):
                def decorator(fn: Any) -> Any:
                    return fn
                return decorator

            def run(self, *args: Any, **kwargs: Any) -> None:
                pass

from gold_trader.brokers.base import BaseBroker
from gold_trader.brokers.metaapi_broker import MetaApiBroker
from gold_trader.brokers.paper_broker import PaperBroker
from gold_trader.brokers.socket_broker import SocketBroker
from gold_trader.brokers.xm_mt5_broker import XmMT5Broker
from gold_trader.config import config
from gold_trader.core.auto_trader import AutoTraderConfig, GoldAutoTrader
from gold_trader.core.backtester import (
    backtest_ema_crossover_strategy,
    backtest_gold_strategy,
    compare_gold_strategies,
)
from gold_trader.core.compounding_ladder import (
    CompoundingLadderManager,
    get_ladder_level_info,
)
from gold_trader.core.data_feed import fetch_gold_candles, get_live_gold_snapshot
from gold_trader.core.fibonacci import analyze_gold_fibonacci
from gold_trader.core.indicators import extract_all_indicators
from gold_trader.core.macro import analyze_macro_drivers
from gold_trader.core.ml_filter import GoldMLPredictor
from gold_trader.core.multi_agent import run_gold_multi_agent_debate
from gold_trader.core.strategy_engine import (
    generate_compounding_ladder_setup,
    generate_gold_trade_setup,
)


# Initialize FastMCP / MCPServer
mcp = FastMCP(
    name="Gold Trading System MCP",
    instructions=(
        "Institutional Gold Trading MCP Server. "
        "Provides live multi-instrument Gold market data (Spot XAU/USD, COMEX Futures, Gold ETFs), "
        "Macro Dollar/Yield correlation, Fibonacci Golden Pocket detection, Multi-Agent Consensus, "
        "Strategy Backtesting, and Free MT5 Socket Bridge / XM MT5 / Paper Trading Order Execution."
    ),
)


def _get_active_broker(account_type: Optional[str] = None) -> BaseBroker:
    """Factory to retrieve the active broker instance based on configuration."""
    mode = config.BROKER_MODE.lower().strip()
    if mode in ("socket", "bridge", "mql5"):
        return SocketBroker(account_type=account_type)
    elif mode == "xm_mt5":
        return XmMT5Broker(account_type=account_type)
    elif mode == "metaapi":
        return MetaApiBroker(account_type=account_type)
    else:
        return PaperBroker()


# ── Market Data & Snapshot Tools ───────────────────────────────────────────────

@mcp.tool()
def gold_market_snapshot() -> Dict[str, Any]:
    """Get a live synchronized market overview of Spot Gold, COMEX Futures, Gold ETFs, DXY, and US10Y."""
    return get_live_gold_snapshot()


@mcp.tool()
def gold_technical_analysis(symbol: str = "XAUUSD", timeframe: str = "1h", period: str = "1mo") -> Dict[str, Any]:
    """
    Run full technical indicator analysis on Gold (EMAs 20/50/200, Supertrend, Donchian, Bollinger, RSI, MACD, ATR, ADX).
    
    Args:
        symbol: Gold symbol — 'XAUUSD' (Spot), 'GC' (COMEX Futures), 'GOLDBEES' (NSE ETF), 'GLD'
        timeframe: Candle interval — '15m', '1h', '4h', '1d' (default '1h')
        period: History period — '1mo', '3mo', '6mo', '1y' (default '1mo')
    """
    candles = fetch_gold_candles(symbol, period=period, interval=timeframe)
    if not candles:
        return {"error": f"No market data available for {symbol}"}
    ind = extract_all_indicators(candles)
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "period": period,
        "candles_count": len(candles),
        "latest_candle": candles[-1],
        "indicators": ind,
    }


@mcp.tool()
def gold_fibonacci_analysis(symbol: str = "XAUUSD", timeframe: str = "1h", lookback: int = 50) -> Dict[str, Any]:
    """
    Analyze Fibonacci retracements, extensions, and Golden Pocket (0.618 - 0.786) reversal zones for Gold.
    
    Args:
        symbol: 'XAUUSD', 'GC', 'GOLDBEES', 'GLD'
        timeframe: '15m', '1h', '4h', '1d'
        lookback: Number of bars to determine swing high and low (default 50)
    """
    candles = fetch_gold_candles(symbol, period="3mo", interval=timeframe)
    if not candles:
        return {"error": f"No data for {symbol}"}
    return analyze_gold_fibonacci(candles, lookback=lookback)


@mcp.tool()
def gold_macro_analysis() -> Dict[str, Any]:
    """Analyze US Dollar Index (DXY) and US 10-Year Treasury Yields to gauge Macro Gold headwinds/tailwinds."""
    return analyze_macro_drivers()


# ── Trade Generation & Multi-Agent Consensus ──────────────────────────────────

@mcp.tool()
def gold_trade_setup(
    symbol: str = "XAUUSD",
    account_balance: float = 10000.0,
    risk_pct: float = 1.0,
    timeframe: str = "1h",
) -> Dict[str, Any]:
    """
    Generate an institutional 3-Layer trade setup for Gold with precise Entry, Stop-Loss, Targets (T1/T2), and Lot Sizing.
    
    Args:
        symbol: 'XAUUSD', 'GC', 'GOLDBEES'
        account_balance: Capital in USD to calculate risk-adjusted lot size (default $10,000)
        risk_pct: Maximum percentage of capital to risk on this trade (default 1.0%)
        timeframe: Entry trigger timeframe ('15m', '1h', '4h')
    """
    return generate_gold_trade_setup(
        symbol=symbol,
        account_balance=account_balance,
        risk_pct=risk_pct,
        timeframe=timeframe,
    )


@mcp.tool()
def gold_multi_agent_debate(symbol: str = "XAUUSD", timeframe: str = "1h") -> Dict[str, Any]:
    """
    Run a 3-agent debate (Technical Analyst, Macro/Yields Analyst, Risk Manager) to reach a high-conviction trade consensus.
    
    Args:
        symbol: 'XAUUSD', 'GC', 'GOLDBEES', 'GLD'
        timeframe: '15m', '1h', '4h', '1d'
    """
    return run_gold_multi_agent_debate(symbol=symbol, timeframe=timeframe)


# ── Strategy Backtesting Tools ────────────────────────────────────────────────

@mcp.tool()
def gold_backtest(
    symbol: str = "XAUUSD",
    strategy: str = "ema_5_20",
    period: str = "1y",
    interval: str = "1h",
    min_target_points: Optional[float] = None,
    stop_loss_points: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Backtest a Gold trading strategy with realistic slippage and transaction costs.
    
    Args:
        symbol: 'XAUUSD', 'GC', 'GOLDBEES', 'GLD'
        strategy: 'ema_5_20' | 'supertrend' | 'donchian' | 'ema_cross' | 'bollinger'
        period: '1mo', '3mo', '6mo', '1y', '2y'
        interval: '5m', '15m', '1h' (hourly) or '1d' (daily)
        min_target_points: Optional minimum profit target in points (Take Profit)
        stop_loss_points: Optional stop loss in points
    """
    return backtest_gold_strategy(
        symbol=symbol,
        strategy=strategy,
        period=period,
        interval=interval,
        min_target_points=min_target_points,
        stop_loss_points=stop_loss_points,
    )


@mcp.tool()
def gold_backtest_ema_crossover(
    symbol: str = "XAUUSD",
    fast_period: int = 5,
    slow_period: int = 20,
    min_target_points: Optional[float] = None,
    stop_loss_points: Optional[float] = None,
    period: str = "1y",
    interval: str = "1h",
) -> Dict[str, Any]:
    """
    Backtest the EMA crossover strategy for Gold with both LONG and SHORT trades.
    Includes minimum target points capture, stop-loss protection, MFE excursion distribution,
    and target sensitivity analysis.

    Args:
        symbol: Gold ticker (default 'XAUUSD')
        fast_period: Fast EMA period (default 5)
        slow_period: Slow EMA period (default 20)
        min_target_points: Minimum target points to secure profit (e.g. 2, 3, 5, 10 pts)
        stop_loss_points: Stop-loss in points (e.g. 5, 10 pts)
        period: Backtest duration ('1mo', '3mo', '6mo', '1y')
        interval: Timeframe ('5m', '15m', '1h', '1d')
    """
    return backtest_ema_crossover_strategy(
        symbol=symbol,
        fast_period=fast_period,
        slow_period=slow_period,
        min_target_points=min_target_points,
        stop_loss_points=stop_loss_points,
        period=period,
        interval=interval,
    )


@mcp.tool()
def gold_compare_strategies(
    symbol: str = "XAUUSD",
    period: str = "1y",
    interval: str = "1h",
) -> Dict[str, Any]:
    """Run all core Gold strategies and return a ranked leaderboard (Return, Sharpe, Max Drawdown)."""
    return compare_gold_strategies(symbol, period, interval)


# ── Broker Execution Tools (XM MT5 / MetaAPI / Paper Trading) ─────────────────

@mcp.tool()
def broker_account_info(account_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get active broker account details: Balance, Equity, Free Margin, Margin Level, Leverage.
    
    Args:
        account_type: Optional override ('demo' or 'live'). If omitted, uses XM_ACCOUNT_TYPE config.
    """
    broker = _get_active_broker(account_type=account_type)
    return dict(broker.get_account_info())


@mcp.tool()
def broker_open_positions(symbol: str = "", account_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all active open positions and floating PnL on the configured broker.
    
    Args:
        symbol: Optional filter by symbol (e.g. 'XAUUSD', 'GOLD')
        account_type: Optional override ('demo' or 'live'). If omitted, uses XM_ACCOUNT_TYPE config.
    """
    broker = _get_active_broker(account_type=account_type)
    sym = symbol.strip() if symbol else None
    return [dict(p) for p in broker.get_open_positions(sym)]


@mcp.tool()
def broker_place_order(
    symbol: str = "XAUUSD",
    side: str = "BUY",
    lots: float = 0.01,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    comment: str = "Gold-Trader-MCP",
) -> Dict[str, Any]:
    """
    Execute a BUY or SELL market order for Gold with lot size and optional SL/TP on the configured broker (Paper, XM MT5, or MetaAPI).
    
    Args:
        symbol: Symbol to trade (e.g. 'XAUUSD', 'GOLD')
        side: 'BUY' or 'SELL'
        lots: Position volume in lots (e.g. 0.01 micro-lot, 0.1 mini-lot, 1.0 standard lot)
        sl: Stop Loss price level
        tp: Take Profit target price level
        comment: Order comment/tag
    """
    broker = _get_active_broker()
    action: Literal["BUY", "SELL"] = "BUY" if side.upper() == "BUY" else "SELL"
    return dict(broker.place_order(symbol, action, lots, sl, tp, comment))


@mcp.tool()
def broker_modify_position(
    ticket: str,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
) -> Dict[str, Any]:
    """Modify the Stop-Loss or Take-Profit of an open trade by ticket number."""
    broker = _get_active_broker()
    return dict(broker.modify_position(ticket, sl, tp))


@mcp.tool()
def broker_close_position(ticket: str) -> Dict[str, Any]:
    """Close an open position by ticket number on the configured broker."""
    broker = _get_active_broker()
    return dict(broker.close_position(ticket))


@mcp.tool()
def gold_auto_trader_scan(
    symbol: str = "XAUUSD",
    timeframe: str = "15m",
    min_rr: float = 1.5,
    risk_pct: float = 1.0,
    min_quality: int = 80,
    max_positions: int = 0,
    auto_execute: bool = False,
) -> Dict[str, Any]:
    """
    Run an automated live market scan, evaluate 3-layer confluence & Risk-to-Reward ratio.
    If auto_execute=True and a favourable trade is detected (RR >= min_rr and score >= min_quality),
    it executes the trade automatically on the active XM / Paper broker with dynamic ATR SL and TP targets.
    """
    broker = _get_active_broker()
    cfg = AutoTraderConfig(
        symbol=symbol,
        timeframe=timeframe,
        risk_pct_per_trade=risk_pct,
        min_risk_reward=min_rr,
        min_quality_score=min_quality,
        max_open_positions=max_positions,
        auto_execute=auto_execute,
    )
    trader = GoldAutoTrader(broker=broker, config_override=cfg)
    return trader.evaluate_and_trade()


@mcp.tool()
def gold_auto_trader_history(limit: int = 10) -> Dict[str, Any]:
    """Retrieve recent automated market scan logs, R:R decisions, and trade execution records."""
    trader = GoldAutoTrader()
    return trader.get_recent_history(limit=limit)


@mcp.tool()
def gold_auto_trader_service_control(
    action: Literal["status", "start", "stop"] = "status",
    interval_seconds: int = 60,
    min_rr: float = 1.5,
    risk_pct: float = 1.0,
    timeframe: str = "15m",
) -> Dict[str, Any]:
    """
    Control or check the 24/7 background Auto-Trader daemon service (tmux: gold-auto-trader).
    Actions:
        - 'status': Check if the 24/7 scanner daemon is currently active
        - 'start': Launch the 24/7 auto-trader daemon with specified R:R and risk parameters
        - 'stop': Terminate the background auto-trader daemon
    """
    import subprocess
    script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts")

    if action == "status":
        res = subprocess.run(["tmux", "has-session", "-t", "gold-auto-trader"], capture_output=True)
        is_active = (res.returncode == 0)
        log_snippet = ""
        log_path = "/home/mcpuser/MCP/gold-trading-system/auto_trader.log"
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                log_snippet = "".join(lines[-10:])
        return {
            "service": "gold-auto-trader",
            "is_active": is_active,
            "status": "RUNNING 24/7" if is_active else "STOPPED",
            "recent_logs": log_snippet,
        }

    elif action == "start":
        start_script = os.path.join(script_dir, "start_auto_trader.sh")
        proc = subprocess.run(
            [start_script, str(interval_seconds), str(min_rr), str(risk_pct), timeframe],
            capture_output=True,
            text=True,
        )
        return {
            "service": "gold-auto-trader",
            "action": "start",
            "status": "LAUNCHED",
            "interval_seconds": interval_seconds,
            "min_rr": min_rr,
            "risk_pct": risk_pct,
            "timeframe": timeframe,
            "output": proc.stdout,
        }

    elif action == "stop":
        stop_script = os.path.join(script_dir, "stop_auto_trader.sh")
        proc = subprocess.run([stop_script], capture_output=True, text=True)
        return {
            "service": "gold-auto-trader",
            "action": "stop",
            "status": "STOPPED",
            "output": proc.stdout,
        }

    return {"status": "ERROR", "message": f"Invalid action: {action}"}


# ── Compounding Ladder & Machine Learning Tools ───────────────────────────────

@mcp.tool()
def get_compounding_ladder_status() -> Dict[str, Any]:
    """
    Get the active status of the 30-Level Gold Compounding Ladder ($20 -> $52,384).
    Returns current level, balance, win rate, and parameters (lots, TP, SL, risk) for the next stage.
    """
    ladder = CompoundingLadderManager()
    return ladder.get_status()


@mcp.tool()
def reset_compounding_ladder(
    starting_balance: float = 1000.0,
    loss_policy: Literal["RESET_TO_LEVEL_1", "STEP_DOWN_BY_BALANCE", "PRESERVE_MILESTONES", "PROGRESSIVE_RATCHET"] = "PROGRESSIVE_RATCHET",
) -> Dict[str, Any]:
    """
    Reset the Compounding Ladder with specified capital and loss policy.
    Loss policies:
        - 'PROGRESSIVE_RATCHET': Dynamic tiered floors ($150 at $250, $350 at $500, $700 at $1,000, $1,800 at $2,500)
        - 'RESET_TO_LEVEL_1': Full reset on loss (default challenge mode)
        - 'STEP_DOWN_BY_BALANCE': Fall back to the level matching current balance
        - 'PRESERVE_MILESTONES': Preserve milestone floors (Level 10: $212, Level 20: $2923)
    """
    ladder = CompoundingLadderManager(loss_policy=loss_policy)
    return ladder.reset_ladder(starting_balance=starting_balance)


@mcp.tool()
def set_compounding_ladder_level(level: int) -> Dict[str, Any]:
    """Set the active compounding level (1 to 30) manually."""
    ladder = CompoundingLadderManager()
    return ladder.set_level(level)


@mcp.tool()
def generate_compounding_ladder_setup_tool(
    symbol: str = "XAUUSD",
    min_win_probability: float = 0.55,
    timeframe: str = "15m",
    use_pullback_limit: bool = True,
    pullback_discount_pips: float = 8.0,
) -> Dict[str, Any]:
    """
    Evaluate market conditions for the 30-Level Compounding Ladder Plan.
    Integrates the high-frequency 15m EMA 9/21 Value Zone Pullback strategy with 8-pip limit discount,
    20-pip TP, 15.4-pip SL, and +10 pip breakeven lock.
    """
    return generate_compounding_ladder_setup(
        symbol=symbol,
        min_win_probability=min_win_probability,
        timeframe=timeframe,
        use_pullback_limit=use_pullback_limit,
        pullback_discount_pips=pullback_discount_pips,
    )


@mcp.tool()
def train_gold_ml_model(
    symbol: str = "XAUUSD",
    period: str = "1y",
    interval: str = "1h",
    atr_multiplier_sl: float = 1.5,
    rr_ratio: float = 1.25,
    min_sl_pts: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Train or recalibrate the Machine Learning Triple-Barrier Meta-Classifier on historical Gold candles.
    Calculates time-series walk-forward AUC, precision, and high-conviction win rates.
    """
    predictor = GoldMLPredictor(symbol=symbol)
    return predictor.train_model(
        symbol=symbol,
        period=period,
        interval=interval,
        atr_multiplier_sl=atr_multiplier_sl,
        rr_ratio=rr_ratio,
        min_sl_pts=min_sl_pts,
    )


@mcp.tool()
def predict_gold_trade_probability(
    symbol: str = "XAUUSD",
    direction: Literal["BUY", "SELL"] = "BUY",
    timeframe: str = "1h",
    min_probability: float = 0.70,
) -> Dict[str, Any]:
    """
    Query the ML model to predict the probability that a BUY or SELL trade reaches +20 pips ($2.00)
    before hitting -15 pips ($1.50) Stop Loss, given the current microstructure and macro environment.
    """
    candles = fetch_gold_candles(symbol, period="1mo", interval=timeframe)
    if len(candles) < 200:
        return {"error": f"Insufficient candles ({len(candles)}) for ML feature extraction. Need >= 200."}

    predictor = GoldMLPredictor()
    return predictor.predict_trade_probability(
        candles=candles,
        direction=direction,
        min_probability_threshold=min_probability,
    )


@mcp.tool()
def execute_compounding_ladder_trade(
    symbol: str = "XAUUSD",
    min_win_probability: float = 0.55,
    timeframe: str = "15m",
    use_pullback_limit: bool = True,
    pullback_discount_pips: float = 8.0,
    account_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate and execute a trade under the 30-Level Compounding Plan on the active broker
    (XM MT5 Socket Bridge / Native XM / Paper Broker).
    Execution is strictly gated: only executes if ML win probability >= min_win_probability.
    """
    setup = generate_compounding_ladder_setup(
        symbol=symbol,
        min_win_probability=min_win_probability,
        timeframe=timeframe,
        use_pullback_limit=use_pullback_limit,
        pullback_discount_pips=pullback_discount_pips,
    )

    if setup.get("status") != "APPROVED_HIGH_CONVICTION":
        return {
            "status": "REJECTED_OR_WAITING",
            "reason": setup.get("reason", "Setup not approved by technical or ML gate"),
            "setup": setup,
        }

    broker = _get_active_broker(account_type=account_type)
    action = setup["action"]
    lots = setup["lots"]
    tp = setup["take_profit"]
    sl = setup["stop_loss"]

    order_res = broker.place_order(
        symbol="GOLD" if "GOLD" in symbol else symbol,
        side=action,
        lots=lots,
        sl=sl,
        tp=tp,
        comment=f"AI-Compounding-L{setup['level']}",
    )

    return {
        "status": "EXECUTED",
        "level": setup["level"],
        "setup": setup,
        "broker_result": order_res,
    }


# ── Server Entrypoint ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Gold Trading System MCP Server")
    parser.add_argument(
        "transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        nargs="?",
        help="Transport mode (default stdio)",
    )
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8001")))
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
