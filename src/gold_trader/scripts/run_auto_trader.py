"""
24/7 Autonomous Gold Live Market Scanner & Risk-Reward Trading Daemon.
Usage:
    python -m gold_trader.scripts.run_auto_trader --once
    python -m gold_trader.scripts.run_auto_trader --loop --interval-seconds 60
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

from gold_trader.brokers.socket_broker import SocketBroker
from gold_trader.core.auto_trader import AutoTraderConfig, GoldAutoTrader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("AutoTraderRunner")


def main():
    parser = argparse.ArgumentParser(description="24/7 Gold Live Scanner & Autonomous Auto-Trader")
    parser.add_argument("--symbol", default="XAUUSD", help="Trading symbol (XAUUSD / GOLD)")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe (5m, 15m, 1h)")
    parser.add_argument(
        "--mode",
        default="FLAT_RISK_TOUCH",
        choices=["FLAT_RISK_TOUCH", "STANDARD_SWING", "DONCHIAN_BREAKOUT", "DONCHIAN_MOMENTUM_HYBRID", "COMPOUNDING_TOUCH", "COMPOUNDING_LADDER", "COMPOUNDING_HYBRID"],
        help="Trading mode (default: FLAT_RISK_TOUCH)",
    )
    parser.add_argument("--interval-seconds", type=int, default=60, help="Scan interval in seconds")
    parser.add_argument("--active-interval-seconds", type=int, default=5, help="Scan interval when in active trade (default: 5s)")
    parser.add_argument("--adaptive-dual-speed", action=argparse.BooleanOptionalAction, default=True, help="Enable adaptive dual-speed polling (60s idle / 5s in-trade)")
    parser.add_argument("--flat-risk", action=argparse.BooleanOptionalAction, default=True, help="Enforce fixed dollar risk execution (Compounding Removed)")
    parser.add_argument("--fixed-risk-usd", type=float, default=75.44, help="Fixed dollar risk per trade (default: $75.44)")
    parser.add_argument("--starting-capital", "--initial-capital", type=float, default=None, help="[Deprecated/Ignored] Starting capital")
    parser.add_argument("--ladder-level", type=int, default=None, help="[Deprecated/Ignored] Ladder level")
    parser.add_argument("--min-rr", type=float, default=1.25, help="Minimum Risk-Reward ratio (e.g. 1.25 for High-Velocity Touch)")
    parser.add_argument("--risk-pct", type=float, default=8.0, help="Risk percentage per trade (default: 8.0%%)")
    parser.add_argument("--donchian-period", type=int, default=10, help="Donchian channel lookback period (default: 10)")
    parser.add_argument("--trend-filter-ema", type=int, default=50, help="Trend filter EMA period (50 or 200, default: 50)")
    parser.add_argument("--trigger-mode", default="TOUCH", choices=["TOUCH", "CLOSE"], help="Trigger mode: TOUCH or CLOSE (default: TOUCH)")
    parser.add_argument("--min-quality", type=int, default=80, help="Minimum setup quality score (0-100)")
    parser.add_argument("--max-positions", type=int, default=0, help="Max open positions limit (0 for uncapped, default: 0)")
    parser.add_argument("--dry-run", action="store_true", help="Analyze and evaluate without placing real orders")
    parser.add_argument("--once", action="store_true", help="Run a single evaluation cycle and exit")
    parser.add_argument("--loop", action="store_true", help="Run continuously in a 24/7 monitoring loop")
    parser.add_argument("--one-trade-per-bar", action=argparse.BooleanOptionalAction, default=True, help="Enforce maximum 1 trade execution per 15m candle bar")
    parser.add_argument("--loss-cooldown-minutes", type=int, default=60, help="Pause scanning/trading for N minutes after a stop loss (default: 60)")
    parser.add_argument("--use-pullback-limit", action=argparse.BooleanOptionalAction, default=False, help="Execute via Pullback Limit order")
    parser.add_argument("--pullback-discount-pips", type=float, default=0.0, help="Pips discount into EMA Value Zone (default: 0.0)")
    parser.add_argument("--max-daily-trades", type=int, default=6, help="Maximum trades executed per calendar day (default: 6)")
    parser.add_argument("--session-filter", action=argparse.BooleanOptionalAction, default=True, help="Restrict entries to 07:00-19:00 UTC London/NY hours")
    parser.add_argument("--enable-partial-scale-out", action=argparse.BooleanOptionalAction, default=True, help="Enable 50%% scale-out at Target 1 (+1.0R)")
    parser.add_argument("--partial-scale-out-pct", type=float, default=0.50, help="Partial scale-out percentage (default: 0.50)")
    parser.add_argument("--scale-out-r", type=float, default=1.00, help="Target 1 R-multiple for partial scale out (default: 1.00)")
    parser.add_argument("--enable-runner-trailing", action=argparse.BooleanOptionalAction, default=True, help="Enable dynamic runner trailing after Target 1")
    parser.add_argument("--runner-trail-buffer", type=float, default=4.50, help="Runner trailing buffer in USD (default: 4.50 for Gold, 350.00 for BTC)")
    parser.add_argument("--runner-trail-buffer-r", type=float, default=0.50, help="Runner trailing stop buffer in R-multiples (default: 0.50R)")
    parser.add_argument("--runner-target-2-multiplier", type=float, default=3.00, help="Target 2 extension multiplier (default: 3.00)")
    parser.add_argument("--enable-news-catalyst-guard", action=argparse.BooleanOptionalAction, default=True, help="Pause trades ±15m around high-impact US macro events (CPI, NFP, FOMC)")
    parser.add_argument("--news-guard-buffer-minutes", type=int, default=15, help="Minutes blackout buffer before/after high-impact news (default: 15)")
    parser.add_argument("--enable-ml-filter", action=argparse.BooleanOptionalAction, default=True, help="Enable Machine Learning win probability filter gate (default: True)")
    parser.add_argument("--ml-min-prob", type=float, default=0.52, help="Minimum win probability threshold for ML filter (default: 0.52)")
    parser.add_argument("--ml-dynamic-sizing", action=argparse.BooleanOptionalAction, default=False, help="Enable dynamic ML confidence-weighted position sizing (default: False)")
    parser.add_argument("--ml-model-path", default=None, help="Custom path to ML model joblib file")
    parser.add_argument("--ladder-state-file", default=None, help="[Deprecated/Ignored] Path to ladder state JSON file")
    parser.add_argument("--db-path", default=None, help="Path to SQLite database file")

    args = parser.parse_args()

    is_flat_risk = True

    cfg = AutoTraderConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        mode=args.mode,
        flat_risk=is_flat_risk,
        fixed_risk_usd=args.fixed_risk_usd,
        risk_pct_per_trade=args.risk_pct,
        min_risk_reward=args.min_rr,
        min_quality_score=args.min_quality,
        max_open_positions=args.max_positions,
        scan_interval_seconds=args.interval_seconds,
        active_interval_seconds=args.active_interval_seconds,
        enable_adaptive_dual_speed=args.adaptive_dual_speed,
        auto_execute=not args.dry_run,
        trail_to_breakeven_at_t1=True,
        one_trade_per_bar=args.one_trade_per_bar,
        loss_cooldown_minutes=args.loss_cooldown_minutes,
        use_pullback_limit=args.use_pullback_limit,
        pullback_discount_pips=args.pullback_discount_pips,
        max_daily_trades=args.max_daily_trades,
        session_filter=args.session_filter,
        donchian_period=args.donchian_period,
        trend_filter_ema=args.trend_filter_ema,
        trigger_mode=args.trigger_mode,
        enable_partial_scale_out=args.enable_partial_scale_out,
        partial_scale_out_pct=args.partial_scale_out_pct,
        scale_out_r=args.scale_out_r,
        enable_runner_trailing=args.enable_runner_trailing,
        runner_trail_buffer_r=args.runner_trail_buffer_r,
        runner_trail_buffer_usd=(
            350.00 if "BTC" in args.symbol.upper() and args.runner_trail_buffer == 4.50
            else (10.00 if "ETH" in args.symbol.upper() and args.runner_trail_buffer == 4.50 else args.runner_trail_buffer)
        ),
        runner_target_2_multiplier=args.runner_target_2_multiplier,
        enable_news_catalyst_guard=args.enable_news_catalyst_guard,
        news_guard_buffer_minutes=args.news_guard_buffer_minutes,
        enable_ml_filter=args.enable_ml_filter,
        ml_min_probability=args.ml_min_prob,
        ml_dynamic_sizing=args.ml_dynamic_sizing,
        ml_model_path=args.ml_model_path,
        ladder_state_file=args.ladder_state_file,
        db_path=args.db_path,
    )

    trader = GoldAutoTrader(broker=SocketBroker(), config_override=cfg)

    logger.info("=" * 60)
    logger.info(f"🚀 24/7 Autonomous {cfg.symbol} Trading Engine Initialized")
    logger.info(f"   Mode: {cfg.mode} | Symbol: {cfg.symbol} | Timeframe: {cfg.timeframe} | Min R:R: {cfg.min_risk_reward:.2f}R")
    logger.info(f"   Donchian: {cfg.donchian_period} | Filter: {cfg.trend_filter_ema} EMA | Trigger: {cfg.trigger_mode}")
    logger.info(f"   Risk Per Trade: {cfg.risk_pct_per_trade}% | Min Quality: {cfg.min_quality_score}/100")
    logger.info(f"   Bar Lock: {cfg.one_trade_per_bar} | Loss Cooldown: {cfg.loss_cooldown_minutes}m")
    logger.info(f"   Daily Trade Cap: {cfg.max_daily_trades} | Session Filter: {cfg.session_filter}")
    logger.info(f"   News Catalyst Guard (OpenBB): {cfg.enable_news_catalyst_guard} (±{cfg.news_guard_buffer_minutes}m buffer)")
    logger.info(f"   ML Meta-Labeling Filter: {cfg.enable_ml_filter} (Min Win Prob: {cfg.ml_min_probability:.1%}, Dynamic Sizing: {cfg.ml_dynamic_sizing})")
    logger.info(f"   Auto Execute: {cfg.auto_execute} | Max Open: {cfg.max_open_positions}")
    logger.info(f"   Partial Scale-Out: {cfg.enable_partial_scale_out} ({int(cfg.partial_scale_out_pct*100)}% @ {cfg.scale_out_r:.2f}R)")
    logger.info(f"   Runner Trailing: {cfg.enable_runner_trailing} (0.5R Dynamic Buffer: {cfg.runner_trail_buffer_r:.2f}R [~${cfg.runner_trail_buffer_usd:.2f}] | T2: {cfg.runner_target_2_multiplier:.2f}R)")
    logger.info(f"   Polling Cycle: Adaptive Dual-Speed ({cfg.enable_adaptive_dual_speed}) | Idle: {cfg.scan_interval_seconds}s | In-Trade: {cfg.active_interval_seconds}s")
    logger.info(f"🎯 Execution Sizing: FLAT RISK (Compounding Strategy Removed)")
    logger.info(f"   Capital Baseline: $1,000.00 | Fixed Risk: ${cfg.fixed_risk_usd:.2f} per trade")
    logger.info(f"   Expected Target 1 Gain: +${cfg.fixed_risk_usd * cfg.min_risk_reward:.2f} ({cfg.min_risk_reward:.2f}R)")
    logger.info("=" * 60)

    if args.once or not args.loop:
        result = trader.evaluate_and_trade()
        logger.info(f"Scan Decision: {result.get('decision')}")
        logger.info(f"Summary: Price=${result.get('current_price')} | Action={result.get('action')} | Bias={result.get('bias')} | R:R={result.get('risk_reward_ratio')}R")
        if result.get("order_result"):
            logger.info(f"Order Result: {result.get('order_result')}")
        return

    # Loop mode
    logger.info(f"Entering continuous monitoring loop (Idle: {cfg.scan_interval_seconds}s | In-Trade: {cfg.active_interval_seconds}s)...")
    cycle = 1
    in_trade_ticks = 0
    while True:
        try:
            # Check if we currently have an active open position for our symbol
            matching_positions = trader.get_matching_open_positions()
            is_in_trade = len(matching_positions) > 0

            if is_in_trade and cfg.enable_adaptive_dual_speed:
                in_trade_ticks += 1
                res = trader.manage_active_positions()
                managed_actions = res.get("managed_actions", [])

                # If an action was taken (partial scale out, BE move, trailing SL):
                if managed_actions:
                    logger.info(f"🛡️ [In-Trade Fast Tick #{in_trade_ticks}] ACTION: {managed_actions}")
                elif in_trade_ticks % 12 == 1:
                    # Periodic heartbeat log every ~60 seconds (12 ticks * 5s) to avoid log spam
                    p_info = matching_positions[0] if matching_positions else {}
                    cur_p = float(p_info.get("current_price", 0.0))
                    op_p = float(p_info.get("open_price", 0.0))
                    lots = float(p_info.get("lots", 0.0))
                    sl_val = float(p_info.get("sl", 0.0))
                    profit = float(p_info.get("pnl") if p_info.get("pnl") is not None else p_info.get("profit", 0.0))
                    logger.info(
                        f"⚡ [In-Trade Fast Cycle #{in_trade_ticks} | {cfg.symbol}] Price: ${cur_p:.2f} | Entry: ${op_p:.2f} | "
                        f"SL: ${sl_val:.2f} | Lots: {lots:.2f} | Floating PnL: ${profit:+.2f} | Active Polling {cfg.active_interval_seconds}s"
                    )

                # If position closed during this fast tick:
                if res.get("open_positions_count", 0) == 0:
                    logger.info(f"🏁 Active position closed! Transitioning back to normal {cfg.scan_interval_seconds}s market scan.")
                    in_trade_ticks = 0
                    time.sleep(cfg.active_interval_seconds)
                    continue

                time.sleep(cfg.active_interval_seconds)
            else:
                in_trade_ticks = 0
                logger.info(f"\n--- [Cycle #{cycle}] Scanning {cfg.symbol} Market @ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} ---")
                res = trader.evaluate_and_trade()
                logger.info(f"Price: ${res.get('current_price')} | Trend: {res.get('bias')} | Setup: {res.get('setup_name')} | Score: {res.get('quality_score')}")
                logger.info(f"Decision: {res.get('decision')}")
                if res.get("order_result"):
                    logger.info(f"⚡ ORDER EVENT: {res.get('order_result')}")
                if res.get("managed_actions"):
                    logger.info(f"🛡️ POSITION MANAGEMENT: {res.get('managed_actions')}")

                cycle += 1
                order_res = res.get("order_result")
                if order_res and order_res.get("status") == "SUCCESS":
                    logger.info(f"🚀 New position entered! Engaging {cfg.active_interval_seconds}-second fast position management loop.")
                    time.sleep(cfg.active_interval_seconds)
                else:
                    time.sleep(cfg.scan_interval_seconds)
        except KeyboardInterrupt:
            logger.info("\nAuto-Trader stopped by user.")
            break
        except Exception as e:
            logger.error(f"Error in scan cycle: {e}", exc_info=True)
            time.sleep(10)


if __name__ == "__main__":
    main()
