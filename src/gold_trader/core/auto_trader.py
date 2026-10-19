"""
Automated Live Market Scanner & Risk-Reward Trade Execution Engine for Gold.
Continuously scans multi-timeframe candles, evaluates institutional confluences,
calculates strict R:R setups, and executes trades on XM MT5 / Paper Broker with dynamic trade management
and real-time Telegram notifications.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from gold_trader.brokers.base import BaseBroker
from gold_trader.brokers.socket_broker import SocketBroker
from gold_trader.config import config
from gold_trader.core.compounding_ladder import CompoundingLadderManager
from gold_trader.core.data_feed import fetch_gold_candles, get_live_gold_snapshot
from gold_trader.core.strategy_engine import (
    calculate_lot_size,
    generate_compounding_ladder_setup,
    generate_donchian_breakout_setup,
    generate_donchian_momentum_hybrid_setup,
    generate_gold_trade_setup,
)
from gold_trader.core.telegram_notifier import TelegramNotifier

logger = logging.getLogger("GoldAutoTrader")


@dataclass
class AutoTraderConfig:
    symbol: str = "XAUUSD"
    timeframe: str = "1h"
    mode: Literal[
        "FLAT_RISK_TOUCH",
        "COMPOUNDING_TOUCH",
        "COMPOUNDING_LADDER",
        "COMPOUNDING_HYBRID",
        "STANDARD_SWING",
        "DONCHIAN_BREAKOUT",
        "DONCHIAN_MOMENTUM_HYBRID",
    ] = "FLAT_RISK_TOUCH"
    flat_risk: bool = True
    fixed_risk_usd: float = 75.44
    risk_pct_per_trade: float = 1.0
    min_risk_reward: float = 1.25
    min_quality_score: int = 80
    max_open_positions: int = 2
    max_spread_pips: float = 4.0
    scan_interval_seconds: int = 60
    auto_execute: bool = True
    trail_to_breakeven_at_t1: bool = True
    send_telegram_alerts: bool = True
    one_trade_per_bar: bool = True
    loss_cooldown_minutes: int = 60
    use_pullback_limit: bool = True
    pullback_discount_pips: float = 8.0
    max_daily_trades: int = 4
    session_filter: bool = True
    session_start_hour_utc: int = 7
    session_end_hour_utc: int = 19
    donchian_period: int = 10
    trend_filter_ema: int = 50
    trigger_mode: str = "TOUCH"
    enable_partial_scale_out: bool = True
    partial_scale_out_pct: float = 0.50
    scale_out_r: float = 1.00
    enable_runner_trailing: bool = True
    runner_trail_buffer_usd: float = 4.50
    runner_target_2_multiplier: float = 3.00
    enable_news_catalyst_guard: bool = True
    news_guard_buffer_minutes: int = 15
    enable_ml_filter: bool = False
    ml_min_probability: float = 0.52
    ml_dynamic_sizing: bool = False
    ml_model_path: Optional[str] = None
    ladder_state_file: Optional[str] = None
    db_path: Optional[str] = None



class GoldAutoTrader:
    """
    Autonomous 24/7 Gold & Bitcoin Trading Scanner, Execution Controller & Telegram Reporter.
    """

    def __init__(
        self,
        broker: Optional[BaseBroker] = None,
        config_override: Optional[AutoTraderConfig] = None,
        notifier: Optional[TelegramNotifier] = None,
        db_path: Optional[str] = None,
    ):
        self.config = config_override or AutoTraderConfig()
        self.broker = broker or SocketBroker()
        self.notifier = notifier or TelegramNotifier()
        if "BTC" in self.config.symbol.upper():
            default_db = "~/.gold_trading_data/btc_auto_trader.db"
            default_ladder = "~/.gold_trading_data/btc_compounding_ladder_state.json"
        elif "ETH" in self.config.symbol.upper():
            default_db = "~/.gold_trading_data/eth_auto_trader.db"
            default_ladder = "~/.gold_trading_data/eth_compounding_ladder_state.json"
        else:
            default_db = "~/.gold_trading_data/auto_trader.db"
            default_ladder = "~/.gold_trading_data/compounding_ladder_state.json"
        self.db_path = os.path.expanduser(db_path or self.config.db_path or default_db)
        ladder_file = self.config.ladder_state_file or default_ladder
        is_flat_risk = self.config.flat_risk or (self.config.mode == "FLAT_RISK_TOUCH")
        active_ladder_mode = "FLAT_RISK" if is_flat_risk else "NOISE_ADAPTIVE_LADDER"
        active_loss_policy = "ONE_STEP_LADDER" if is_flat_risk else "PROGRESSIVE_RATCHET"
        self.ladder_manager = CompoundingLadderManager(
            state_file=ladder_file,
            ladder_mode=active_ladder_mode,
            loss_policy=active_loss_policy,
            starting_balance=1000.0,
            fixed_risk_usd=self.config.fixed_risk_usd,
        )

        self._ensure_db()
        self.is_running = False
        self._load_active_tracked_positions()
        self.last_trade_bar: Optional[str] = self._get_state("last_trade_bar")
        cooldown_str = self._get_state("cooldown_until_ts")
        self.cooldown_until: Optional[float] = float(cooldown_str) if cooldown_str else None

    @staticmethod
    def _is_matching_symbol(pos_symbol: str, target_symbol: str) -> bool:
        """Check if a broker position symbol matches the trader's target symbol family."""
        p_u = pos_symbol.upper()
        t_u = target_symbol.upper()
        if "BTC" in t_u:
            return "BTC" in p_u
        if "ETH" in t_u:
            return "ETH" in p_u
        if "GOLD" in t_u or "XAU" in t_u:
            return ("GOLD" in p_u or "XAU" in p_u)
        return p_u.startswith(t_u) or t_u.startswith(p_u)

    def _ensure_db(self):
        dname = os.path.dirname(self.db_path)
        if dname:
            os.makedirs(dname, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scan_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    bias TEXT NOT NULL,
                    setup_name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quality_score INTEGER NOT NULL,
                    rr_ratio REAL NOT NULL,
                    decision TEXT NOT NULL,
                    details_json TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticket TEXT NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    lots REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    target_1 REAL,
                    target_2 REAL,
                    rr_ratio REAL,
                    status TEXT NOT NULL,
                    close_price REAL,
                    pnl REAL,
                    close_time TEXT,
                    comment TEXT
                )
            """)
            # Ensure tracking columns exist in existing database
            try:
                cur = conn.cursor()
                cols = [r[1] for r in cur.execute("PRAGMA table_info(trade_executions)").fetchall()]
                if "comment" not in cols:
                    conn.execute("ALTER TABLE trade_executions ADD COLUMN comment TEXT")
                if "tp1_taken" not in cols:
                    conn.execute("ALTER TABLE trade_executions ADD COLUMN tp1_taken INTEGER DEFAULT 0")
                if "peak_price" not in cols:
                    conn.execute("ALTER TABLE trade_executions ADD COLUMN peak_price REAL")
                if "partial_closed_lots" not in cols:
                    conn.execute("ALTER TABLE trade_executions ADD COLUMN partial_closed_lots REAL DEFAULT 0.0")
                if "partial_realized_usd" not in cols:
                    conn.execute("ALTER TABLE trade_executions ADD COLUMN partial_realized_usd REAL DEFAULT 0.0")
                conn.commit()
            except Exception as e:
                logger.debug("Table migration note: %s", e)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS engine_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            conn.commit()

    def _save_state(self, key: str, value: str):
        """Persist engine control state to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO engine_state (key, value) VALUES (?, ?)",
                    (key, value)
                )
                conn.commit()
        except Exception as e:
            logger.error("Failed to save engine state %s: %s", key, e)

    def _get_state(self, key: str) -> Optional[str]:
        """Retrieve engine control state from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                row = cur.execute("SELECT value FROM engine_state WHERE key = ?", (key,)).fetchone()
                return str(row[0]) if row else None
        except Exception:
            return None

    def _load_active_tracked_positions(self):
        """Load currently open tracked trades into memory."""
        self.tracked_positions: Dict[str, Dict[str, Any]] = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT * FROM trade_executions WHERE status = 'OPEN'")
                for row in cur.fetchall():
                    d = dict(row)
                    self.tracked_positions[str(d["ticket"])] = d
        except Exception as e:
            logger.error("Error loading tracked positions: %s", e)

    def log_scan(
        self,
        symbol: str,
        price: float,
        bias: str,
        setup_name: str,
        action: str,
        quality_score: int,
        rr_ratio: float,
        decision: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        now_str = datetime.now(timezone.utc).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO scan_logs 
                    (timestamp, symbol, price, bias, setup_name, action, quality_score, rr_ratio, decision, details_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (now_str, symbol, price, bias, setup_name, action, quality_score, rr_ratio, decision, json.dumps(details or {})),
                )
                conn.commit()
        except Exception as e:
            logger.error("Failed to write scan log: %s", e)

    def evaluate_and_trade(self) -> Dict[str, Any]:
        """
        Execute one complete analysis, position monitoring, and automated trading cycle.
        """
        now = datetime.now(timezone.utc).isoformat()
        symbol = self.config.symbol
        
        # 1. Check account state & open positions
        try:
            acct = self.broker.get_account_info()
            balance = float(acct.get("balance", 10000.0))
            if balance <= 0:
                balance = 10000.0
        except Exception as e:
            balance = 10000.0

        try:
            open_positions = self.broker.get_open_positions()
        except Exception as e:
            open_positions = []

        try:
            pending_orders = self.broker.get_pending_orders()
        except Exception as e:
            pending_orders = []

        # 2. Check for Closed Positions (Exit Alert Detection)
        self._detect_and_notify_exits(open_positions, current_balance=balance)

        # 3. Manage existing open positions (Trailing SL / Breakeven)
        managed_actions = self._manage_open_positions(open_positions)

        # 4. Generate Trade Setup based on mode
        is_touch = (self.config.mode in ("COMPOUNDING_TOUCH", "FLAT_RISK_TOUCH"))
        is_hybrid = (self.config.mode in ("COMPOUNDING_HYBRID", "COMPOUNDING_TOUCH", "FLAT_RISK_TOUCH") or self.config.timeframe == "1h")
        if self.config.mode in ("COMPOUNDING_LADDER", "COMPOUNDING_HYBRID", "COMPOUNDING_TOUCH", "FLAT_RISK_TOUCH"):
            tf = "1h" if is_hybrid else self.config.timeframe
            strat_mode = "HIGH_VELOCITY_TOUCH_1H" if (is_touch or is_hybrid) else "VALUE_ZONE_15M"
            setup = generate_compounding_ladder_setup(
                symbol=symbol,
                timeframe=tf,
                ladder_manager=self.ladder_manager,
                min_win_probability=0.50,
                use_pullback_limit=self.config.use_pullback_limit if not is_hybrid else False,
                pullback_discount_pips=self.config.pullback_discount_pips if not is_hybrid else 0.0,
                strategy_mode=strat_mode,
                donchian_period=self.config.donchian_period,
                rr_multiplier=self.config.min_risk_reward,
                trend_filter_ema=self.config.trend_filter_ema,
                trigger_mode=self.config.trigger_mode,
            )
            if "error" in setup:
                self.log_scan(symbol, 0.0, "ERROR", "None", "HOLD", 0, 0.0, f"Error: {setup.get('error')}", setup)
                return {"status": "ERROR", "message": setup.get("error"), "timestamp": now}

            order_type = str(setup.get("order_type", "MARKET" if is_hybrid else ("LIMIT" if self.config.use_pullback_limit else "MARKET")))
            price = float(setup.get("entry_price", setup.get("market_price", 0.0)))
            bias = str(setup.get("bias", "NEUTRAL"))
            is_flat = (self.ladder_manager.ladder_mode == "FLAT_RISK")
            default_name = (
                f"1H Donchian 10 Touch (Flat Risk ${self.config.fixed_risk_usd:.2f})"
                if (is_touch and is_flat)
                else (
                    "1H Donchian 10 Touch High-Velocity Compounding"
                    if is_touch
                    else ("1H Donchian + Momentum Hybrid Compounding" if is_hybrid else "Compounding Ladder Setup")
                )
            )
            setup_name = str(setup.get("setup_name", default_name))
            action = str(setup.get("action", "HOLD"))
            quality_score = 95 if action in ("BUY", "SELL") else 50
            rr_t1 = float(setup.get("reward_risk_ratio", self.config.min_risk_reward))
            suggested_lots = float(setup.get("lots", 0.01))
            sl = setup.get("stop_loss")
            tp1 = setup.get("take_profit")
            tp2 = tp1
            level = setup.get("level", 1)
        elif self.config.mode in ("DONCHIAN_MOMENTUM_HYBRID", "HYBRID"):
            order_type = "MARKET"
            setup = generate_donchian_momentum_hybrid_setup(
                symbol=symbol,
                account_balance=balance,
                risk_pct=self.config.risk_pct_per_trade,
                timeframe=self.config.timeframe,
            )
            if "error" in setup:
                self.log_scan(symbol, 0.0, "ERROR", "None", "HOLD", 0, 0.0, f"Error: {setup.get('error')}", setup)
                return {"status": "ERROR", "message": setup.get("error"), "timestamp": now}

            price = float(setup.get("entry_price", 0.0))
            bias = str(setup.get("bias", "NEUTRAL"))
            setup_name = str(setup.get("setup_name", "Donchian Momentum Hybrid"))
            action = str(setup.get("action", "HOLD"))
            quality_score = int(setup.get("quality_score", 0))
            rr_t1 = float(setup.get("risk_reward_t1", 1.8))
            suggested_lots = float(setup.get("suggested_lot_size", 0.01))
            sl = setup.get("stop_loss")
            tp1 = setup.get("target_1")
            tp2 = setup.get("target_2")
            level = 1
        elif self.config.mode == "DONCHIAN_BREAKOUT":
            order_type = "MARKET"
            setup = generate_donchian_breakout_setup(
                symbol=symbol,
                account_balance=balance,
                risk_pct=self.config.risk_pct_per_trade,
                timeframe=self.config.timeframe,
            )
            if "error" in setup:
                self.log_scan(symbol, 0.0, "ERROR", "None", "HOLD", 0, 0.0, f"Error: {setup.get('error')}", setup)
                return {"status": "ERROR", "message": setup.get("error"), "timestamp": now}

            price = float(setup.get("entry_price", 0.0))
            bias = str(setup.get("bias", "NEUTRAL"))
            setup_name = str(setup.get("setup_name", "Donchian Breakout"))
            action = str(setup.get("action", "HOLD"))
            quality_score = int(setup.get("quality_score", 0))
            rr_t1 = float(setup.get("risk_reward_t1", 1.2))
            suggested_lots = float(setup.get("suggested_lot_size", 0.01))
            sl = setup.get("stop_loss")
            tp1 = setup.get("target_1")
            tp2 = setup.get("target_2")
            level = 1
        else:
            order_type = "MARKET"
            setup = generate_gold_trade_setup(
                symbol=symbol,
                account_balance=balance,
                risk_pct=self.config.risk_pct_per_trade,
                timeframe=self.config.timeframe,
            )
            if "error" in setup:
                self.log_scan(symbol, 0.0, "ERROR", "None", "HOLD", 0, 0.0, f"Error: {setup.get('error')}", setup)
                return {"status": "ERROR", "message": setup.get("error"), "timestamp": now}

            price = float(setup.get("entry_price", 0.0))
            bias = str(setup.get("bias", "NEUTRAL"))
            setup_name = str(setup.get("setup_name", "None"))
            action = str(setup.get("action", "HOLD"))
            quality_score = int(setup.get("quality_score", 0))
            rr_t1 = float(setup.get("risk_reward_t1", 1.0))
            suggested_lots = float(setup.get("suggested_lot_size", 0.01))
            sl = setup.get("stop_loss")
            tp1 = setup.get("target_1")
            tp2 = setup.get("target_2")
            level = 1

        # Clean up stale pending limit orders if candle has advanced
        candle_time = setup.get("candle_time")
        if pending_orders and candle_time:
            for po in pending_orders:
                po_ticket = str(po.get("ticket"))
                po_comment = str(po.get("comment", ""))
                if ("Compounding" in po_comment or "AI-Auto" in po_comment) and self.last_trade_bar and str(candle_time) != self.last_trade_bar:
                    self.broker.cancel_order(po_ticket)
                    logger.info(f"🧹 Cancelled unfilled pending limit order #{po_ticket} as candle advanced to {candle_time}.")

        # 5. Apply Risk-to-Reward & Quality Gates
        rejection_reasons = []

        if action == "HOLD":
            rejection_reasons.append("No actionable trigger pattern detected.")

        if quality_score < self.config.min_quality_score:
            rejection_reasons.append(f"Quality score {quality_score} < minimum {self.config.min_quality_score}.")

        if rr_t1 < self.config.min_risk_reward:
            rejection_reasons.append(f"R:R ratio {rr_t1:.2f} < minimum {self.config.min_risk_reward:.2f}.")

        sym_open_positions = [p for p in open_positions if self._is_matching_symbol(str(p.get("symbol", "")), symbol)]
        sym_pending_orders = [po for po in pending_orders if self._is_matching_symbol(str(po.get("symbol", "")), symbol)]

        active_exposure = len(sym_open_positions) + len(sym_pending_orders)
        if active_exposure >= self.config.max_open_positions:
            rejection_reasons.append(f"Max active exposure reached ({len(sym_open_positions)} open, {len(sym_pending_orders)} pending >= {self.config.max_open_positions}).")

        # Check if we already have an open position in the same direction
        same_direction = [p for p in sym_open_positions if p.get("side") == action]
        if same_direction:
            rejection_reasons.append(f"Position already open in direction {action} (Ticket #{same_direction[0].get('ticket')}).")

        # Check Bar Lock (Max 1 trade per candle bar)
        if self.config.one_trade_per_bar and candle_time:
            if self.last_trade_bar == str(candle_time):
                rejection_reasons.append(f"Bar Lock: Trade already executed on candle [{candle_time}]. Awaiting next fresh candle.")

        # Check Loss Cooldown (Pause after stop-loss to avoid chop whipsaws)
        now_ts = datetime.now(timezone.utc).timestamp()
        if self.cooldown_until and now_ts < self.cooldown_until:
            rem_mins = int((self.cooldown_until - now_ts) / 60) + 1
            rejection_reasons.append(f"Loss Cooldown: Pausing entries for {rem_mins}m to prevent chop whipsaw.")

        # Check Weekend / Session Filter (Restrict entries to active market and liquid London/NY hours)
        now_dt = datetime.now(timezone.utc)
        is_crypto = "BTC" in symbol.upper() or "ETH" in symbol.upper()
        if self.config.session_filter:
            is_weekend = now_dt.weekday() == 5 or (now_dt.weekday() == 6 and now_dt.hour < 22)
            if is_weekend and not is_crypto:
                rejection_reasons.append("Market Closed: Weekend hours (Gold market reopens Sunday 22:05 UTC).")
            elif not (self.config.session_start_hour_utc <= now_dt.hour < self.config.session_end_hour_utc):
                rejection_reasons.append(
                    f"Session Filter: Current time {now_dt.strftime('%H:%M UTC')} is outside liquid London/NY hours "
                    f"({self.config.session_start_hour_utc:02d}:00 - {self.config.session_end_hour_utc:02d}:00 UTC). Avoiding Asian chop."
                )

        # Check Daily Trade Cap (Max daily trades to preserve capital)
        today_date_str = now_dt.strftime("%Y-%m-%d")
        daily_trades_count = 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                row = cur.execute(
                    "SELECT COUNT(*) FROM trade_executions WHERE (timestamp LIKE ? OR close_time LIKE ?)",
                    (f"{today_date_str}%", f"{today_date_str}%"),
                ).fetchone()
                daily_trades_count = row[0] if row else 0
        except Exception as e:
            logger.debug("Failed to query daily trade count: %s", e)

        if self.config.max_daily_trades > 0 and daily_trades_count >= self.config.max_daily_trades:
            rejection_reasons.append(
                f"Daily Trade Cap Reached: {daily_trades_count} trades executed today >= max {self.config.max_daily_trades}. Capital preserved."
            )

        # Check News Catalyst Guard (Pause entries around high-impact macroeconomic releases via OpenBB)
        if self.config.enable_news_catalyst_guard and action in ("BUY", "SELL"):
            try:
                from gold_trader.core.openbb_feed import check_high_impact_news_window
                news_status = check_high_impact_news_window(buffer_minutes=self.config.news_guard_buffer_minutes)
                if news_status.get("is_in_news_window"):
                    ev_title = news_status.get("event_title", "High-Impact Catalyst")
                    ev_time = news_status.get("event_time_utc", "")
                    diff_m = news_status.get("minutes_to_nearest_event", 0.0)
                    rejection_reasons.append(
                        f"News Catalyst Guard (OpenBB): High-impact release '{ev_title}' at {ev_time} "
                        f"({diff_m:+.1f}m away). Pausing entries within ±{self.config.news_guard_buffer_minutes}m to eliminate spread spikes."
                    )
            except Exception as e:
                logger.debug("News catalyst guard check note: %s", e)

        # Check Machine Learning High-Probability Meta-Labeling Gate
        ml_eval = None
        if self.config.enable_ml_filter and action in ("BUY", "SELL"):
            try:
                from gold_trader.core.ml_filter import GoldMLPredictor
                predictor = GoldMLPredictor(model_path=self.config.ml_model_path, symbol=symbol)
                candles_eval = fetch_gold_candles(symbol, period="1mo", interval=self.config.timeframe)
                if candles_eval and len(candles_eval) >= 200:
                    ml_eval = predictor.predict_trade_probability(
                        candles_eval,
                        direction=action,
                        min_probability_threshold=self.config.ml_min_probability,
                        use_closed_bar=True,
                    )
                    setup["ml_filter"] = ml_eval
                    prob_win = float(ml_eval.get("probability_win", 0.50))
                    is_high_conv = bool(ml_eval.get("is_high_conviction", False))
                    lot_mult = float(ml_eval.get("recommended_lot_multiplier", 1.0))

                    if not is_high_conv:
                        rejection_reasons.append(
                            f"ML Filter Gate: Win probability {prob_win:.1%} < minimum threshold "
                            f"{self.config.ml_min_probability:.1%}. Skipping low-probability setup to preserve capital."
                        )
                    else:
                        logger.info(
                            f"🤖 ML Gate Approved: Win Probability {prob_win:.1%} >= {self.config.ml_min_probability:.1%} "
                            f"({ml_eval.get('recommendation', 'APPROVED')})"
                        )
                        if self.config.ml_dynamic_sizing and lot_mult > 0:
                            old_lots = suggested_lots
                            suggested_lots = max(0.01, round(suggested_lots * lot_mult, 2))
                            logger.info(f"🤖 ML Dynamic Sizing: Scaled lots {old_lots} -> {suggested_lots} (x{lot_mult:.2f})")
            except Exception as ml_err:
                logger.debug("ML Filter gate evaluation note: %s", ml_err)

        # 6. Make Final Decision & Execute
        if rejection_reasons:
            decision = f"SKIP: {'; '.join(rejection_reasons)}"
            order_result = None
        else:
            decision = f"EXECUTE: Favourable {action} Setup on {setup_name} ({order_type} @ ${price:.2f}, RR {rr_t1:.2f} >= {self.config.min_risk_reward:.2f})"
            if self.config.auto_execute:
                try:
                    # Calibrate SL and TP to live broker quotes only for MARKET execution
                    if order_type == "MARKET":
                        try:
                            sym_info = self.broker.get_symbol_info(symbol)
                            if sym_info and sym_info.get("status") == "SUCCESS":
                                b_bid = float(sym_info.get("bid", 0.0))
                                b_ask = float(sym_info.get("ask", 0.0))
                                broker_entry = b_ask if action == "BUY" else b_bid
                                if broker_entry > 0:
                                    is_btc = "BTC" in symbol.upper()
                                    is_eth = "ETH" in symbol.upper()
                                    default_sl_d = 500.0 if is_btc else (20.0 if is_eth else 1.50)
                                    default_tp_d = 625.0 if is_btc else (25.0 if is_eth else 2.00)
                                    sl_dist = abs(price - sl) if sl else default_sl_d
                                    tp_dist = abs(tp1 - price) if tp1 else default_tp_d
                                    price = broker_entry
                                    if action == "BUY":
                                        sl = round(broker_entry - sl_dist, 2)
                                        tp1 = round(broker_entry + tp_dist, 2)
                                    else:
                                        sl = round(broker_entry + sl_dist, 2)
                                        tp1 = round(broker_entry - tp_dist, 2)
                                    tp2 = tp1
                        except Exception as cal_err:
                            logger.debug("Broker price calibration: %s", cal_err)
                    else:
                        # For LIMIT execution: keep precise 8-pip pullback discount and calculated SL/TP
                        price = round(price, 2)
                        if sl is not None:
                            sl = round(float(sl), 2)
                        if tp1 is not None:
                            tp1 = round(float(tp1), 2)
                            tp2 = tp1

                    order_comment = f"AI-FlatRisk-{symbol}"
                    actual_tp = tp2 if (self.config.enable_partial_scale_out or self.config.enable_runner_trailing or self.config.mode in ("DONCHIAN_BREAKOUT", "DONCHIAN_MOMENTUM_HYBRID", "HYBRID")) and tp2 else tp1
                    order_result = self.broker.place_order(
                        symbol="GOLD" if "GOLD" in symbol else symbol,
                        side=action,  # type: ignore
                        lots=suggested_lots,
                        sl=sl,
                        tp=actual_tp,
                        comment=order_comment,
                        price=price if order_type == "LIMIT" else None,
                        order_type="LIMIT" if order_type == "LIMIT" else "MARKET",
                    )
                    # Record to trade executions database & send Telegram alert
                    if order_result.get("status") == "SUCCESS":
                        ticket = str(order_result.get("ticket"))
                        filled_price = float(order_result.get("price", price))
                        
                        if candle_time:
                            self.last_trade_bar = str(candle_time)
                            self._save_state("last_trade_bar", str(candle_time))
                            logger.info(f"🔒 Bar Lock engaged for 15m candle bar: {candle_time}")
                        
                        trade_record = {
                            "ticket": ticket,
                            "symbol": symbol,
                            "action": action,
                            "lots": suggested_lots,
                            "entry_price": filled_price,
                            "stop_loss": sl,
                            "target_1": tp1,
                            "target_2": tp2,
                            "rr_ratio": rr_t1,
                            "comment": order_comment,
                            "status": "PENDING" if order_type == "LIMIT" else "OPEN",
                            "timestamp": now,
                            "tp1_taken": 0,
                            "peak_price": filled_price,
                            "partial_closed_lots": 0.0,
                            "partial_realized_usd": 0.0,
                        }
                        
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute(
                                """
                                INSERT OR REPLACE INTO trade_executions
                                (timestamp, ticket, symbol, action, lots, entry_price, stop_loss, target_1, target_2, rr_ratio, status, comment, tp1_taken, peak_price, partial_closed_lots, partial_realized_usd)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (now, ticket, symbol, action, suggested_lots, filled_price, sl, tp1, tp2, rr_t1, "OPEN", order_comment, 0, filled_price, 0.0, 0.0),
                            )
                            conn.commit()

                        self.tracked_positions[ticket] = trade_record

                        if self.config.send_telegram_alerts:
                            self.notifier.send_trade_entry(order=order_result, setup=setup)
                except Exception as e:
                    order_result = {"status": "ERROR", "message": f"Execution failed: {e}"}
            else:
                order_result = {"status": "DRY_RUN", "message": "Signal identified but auto_execute=False"}

        self.log_scan(
            symbol=symbol,
            price=price,
            bias=bias,
            setup_name=setup_name,
            action=action,
            quality_score=quality_score,
            rr_ratio=rr_t1,
            decision=decision,
            details=setup,
        )

        # Check and send Daily Post-Session Report if hour >= 21 UTC and not yet sent today
        now_dt = datetime.now(timezone.utc)
        today_date_str = now_dt.strftime("%Y-%m-%d")
        if not is_crypto and now_dt.hour >= 21 and self._get_state("daily_report_date") != today_date_str:
            try:
                import subprocess
                subprocess.Popen(
                    ["/usr/bin/python3", "/home/mcpuser/MCP/gold-trading-system/scripts/send_daily_report.py"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._save_state("daily_report_date", today_date_str)
                logger.info(f"📊 Dispatched daily session report for {today_date_str}")
            except Exception as dr_err:
                logger.error(f"Failed to trigger daily report: {dr_err}")

        return {
            "timestamp": now,
            "symbol": symbol,
            "current_price": price,
            "bias": bias,
            "setup_name": setup_name,
            "action": action,
            "quality_score": quality_score,
            "risk_reward_ratio": rr_t1,
            "suggested_lots": suggested_lots,
            "stop_loss": sl,
            "target_1": tp1,
            "target_2": tp2,
            "decision": decision,
            "open_positions_count": len(open_positions),
            "order_result": order_result,
            "managed_actions": managed_actions,
        }

    def _manage_open_positions(self, open_positions: List[Dict[str, Any]]) -> List[str]:
        """
        Institutional Two-Stage Position Management:
        1. Stage 1: Scale out 50% volume at Target 1 (+1.0R) into realized cash.
        2. Stage 2: Move remaining SL to Break-Even + buffer (Risk-Free Runner).
        3. Stage 3: Trail runner with wide ATR-cleared buffer behind peak excursion up to Target 2.
        """
        actions_taken = []
        if not self.config.trail_to_breakeven_at_t1:
            return actions_taken

        for p in open_positions:
            try:
                ticket = str(p.get("ticket"))
                p_sym = str(p.get("symbol", ""))
                if not self._is_matching_symbol(p_sym, self.config.symbol):
                    continue

                open_price = float(p.get("open_price", 0.0))
                current_price = float(p.get("current_price", 0.0))
                cur_lots = float(p.get("lots", 0.01))
                sl = p.get("sl")
                tp = p.get("tp")
                side = str(p.get("side", "BUY")).upper()
                symbol = str(p.get("symbol", self.config.symbol))

                is_btc = "BTC" in symbol.upper()
                is_eth = "ETH" in symbol.upper()
                contract_size = 1.0 if (is_btc or is_eth) else 100.0
                default_sl_dist = 500.00 if is_btc else (20.00 if is_eth else 8.00)
                be_offset = 30.00 if is_btc else (2.00 if is_eth else 0.50)
                if is_btc:
                    trail_buffer = self.config.runner_trail_buffer_usd if self.config.runner_trail_buffer_usd > 50.0 else 350.00
                elif is_eth:
                    trail_buffer = self.config.runner_trail_buffer_usd if self.config.runner_trail_buffer_usd != 4.50 else 10.00
                else:
                    trail_buffer = self.config.runner_trail_buffer_usd

                tracked = self.tracked_positions.get(ticket)
                if not tracked:
                    initial_sl = float(sl or (open_price - default_sl_dist if side == "BUY" else open_price + default_sl_dist))
                    initial_risk_dist = max(abs(open_price - initial_sl), default_sl_dist)
                    calc_t1 = round(open_price + (self.config.scale_out_r * initial_risk_dist), 2) if side == "BUY" else round(open_price - (self.config.scale_out_r * initial_risk_dist), 2)
                    calc_t2 = round(open_price + (self.config.runner_target_2_multiplier * initial_risk_dist), 2) if side == "BUY" else round(open_price - (self.config.runner_target_2_multiplier * initial_risk_dist), 2)
                    tracked = {
                        "ticket": ticket,
                        "symbol": symbol,
                        "action": side,
                        "lots": cur_lots,
                        "entry_price": open_price,
                        "stop_loss": sl,
                        "target_1": calc_t1,
                        "target_2": calc_t2 if (self.config.enable_runner_trailing and calc_t2) else tp,
                        "tp1_taken": 0,
                        "peak_price": open_price,
                        "partial_closed_lots": 0.0,
                        "partial_realized_usd": 0.0,
                    }
                    self.tracked_positions[ticket] = tracked

                # Track peak price excursion
                if side == "BUY":
                    peak_price = max(float(tracked.get("peak_price") or open_price), current_price)
                else:
                    peak_price = min(float(tracked.get("peak_price") or open_price), current_price)
                tracked["peak_price"] = peak_price

                # Determine initial risk and R-multiple targets
                initial_sl = float(tracked.get("stop_loss") or sl or (open_price - default_sl_dist if side == "BUY" else open_price + default_sl_dist))
                initial_risk_dist = max(abs(open_price - initial_sl), default_sl_dist)
                target_1 = float(tracked.get("target_1") or (open_price + (self.config.scale_out_r * initial_risk_dist) if side == "BUY" else open_price - (self.config.scale_out_r * initial_risk_dist)))
                target_2 = float(tracked.get("target_2") or (open_price + (self.config.runner_target_2_multiplier * initial_risk_dist) if side == "BUY" else open_price - (self.config.runner_target_2_multiplier * initial_risk_dist)))
                tp1_taken = bool(tracked.get("tp1_taken", 0))

                # -------------------------------------------------------------
                # STAGE 1 & 2: Target 1 Reached (+1.0R) -> 50% Scale-Out + Move SL to BE + offset
                # -------------------------------------------------------------
                hit_tp1 = (current_price >= target_1) if side == "BUY" else (current_price <= target_1)
                if hit_tp1 and not tp1_taken:
                    if cur_lots >= 0.02 and self.config.enable_partial_scale_out:
                        close_lots = round(cur_lots * self.config.partial_scale_out_pct, 2)
                        close_res = self.broker.close_position(ticket=ticket, lots=close_lots)
                        if close_res.get("status") == "SUCCESS":
                            realized_usd = round(abs(current_price - open_price) * close_lots * contract_size, 2)
                            tracked["partial_closed_lots"] = close_lots
                            tracked["partial_realized_usd"] = realized_usd
                            msg = f"🏆 Scaled out {close_lots} lots (+$ {realized_usd:.2f}) at Target 1 for Ticket #{ticket}"
                            actions_taken.append(msg)
                            logger.info(msg)
                            if self.config.send_telegram_alerts:
                                self.notifier.send_trade_update(
                                    ticket=ticket,
                                    symbol=symbol,
                                    update_type="Target 1 Hit! 50% Scaled Out 💰",
                                    details=f"Price hit Target 1 (${target_1:.2f}). Scaled out {close_lots} lots. Banked +${realized_usd:.2f} guaranteed cash.",
                                )

                    tracked["tp1_taken"] = 1

                    # Shift remaining volume to Break-Even + offset (Risk-Free Runner)
                    new_sl = round(open_price + be_offset, 2) if side == "BUY" else round(open_price - be_offset, 2)
                    new_tp = target_2 if (self.config.enable_runner_trailing and target_2) else p.get("tp")
                    res = self.broker.modify_position(ticket=ticket, sl=new_sl, tp=new_tp)
                    if res.get("status") == "SUCCESS":
                        tracked["stop_loss"] = new_sl
                        tracked["target_2"] = new_tp
                        msg = f"Trailed SL to Break-Even + ${be_offset:.2f} (${new_sl}) for Ticket #{ticket}"
                        actions_taken.append(msg)
                        if self.config.send_telegram_alerts:
                            self.notifier.send_trade_update(
                                ticket=ticket,
                                symbol=symbol,
                                update_type="Risk-Free Runner Activated 🛡️",
                                details=f"Stop-Loss adjusted to ${new_sl:.2f} (Break-Even + spread). Remaining runner trailing toward Target 2 (${new_tp:.2f}).",
                            )

                # -------------------------------------------------------------
                # STAGE 3: Dynamic Runner Trailing with Wide Buffer
                # -------------------------------------------------------------
                elif tracked.get("tp1_taken", 0) and self.config.enable_runner_trailing:
                    current_sl_val = float(sl) if sl is not None else (open_price + be_offset if side == "BUY" else open_price - be_offset)

                    if side == "BUY":
                        trailed_sl = round(peak_price - trail_buffer, 2)
                        if trailed_sl >= current_sl_val + be_offset and trailed_sl >= open_price + be_offset:
                            res = self.broker.modify_position(ticket=ticket, sl=trailed_sl, tp=p.get("tp"))
                            if res.get("status") == "SUCCESS":
                                tracked["stop_loss"] = trailed_sl
                                msg = f"Trailed Runner SL to ${trailed_sl} for Ticket #{ticket}"
                                actions_taken.append(msg)
                                if self.config.send_telegram_alerts:
                                    self.notifier.send_trade_update(
                                        ticket=ticket,
                                        symbol=symbol,
                                        update_type="Runner Trailing SL Advanced 🏃",
                                        details=f"Peak reached ${peak_price:.2f}. Trailed Stop-Loss advanced to ${trailed_sl:.2f} (+$ {trailed_sl - open_price:.2f} profit locked).",
                                    )
                    else:  # SELL
                        trailed_sl = round(peak_price + trail_buffer, 2)
                        if trailed_sl <= current_sl_val - be_offset and trailed_sl <= open_price - be_offset:
                            res = self.broker.modify_position(ticket=ticket, sl=trailed_sl, tp=p.get("tp"))
                            if res.get("status") == "SUCCESS":
                                tracked["stop_loss"] = trailed_sl
                                msg = f"Trailed Runner SL to ${trailed_sl} for Ticket #{ticket}"
                                actions_taken.append(msg)
                                if self.config.send_telegram_alerts:
                                    self.notifier.send_trade_update(
                                        ticket=ticket,
                                        symbol=symbol,
                                        update_type="Runner Trailing SL Advanced 🏃",
                                        details=f"Peak dropped to ${peak_price:.2f}. Trailed Stop-Loss advanced to ${trailed_sl:.2f} (+$ {open_price - trailed_sl:.2f} profit locked).",
                                    )

                # Persist updated tracking state to database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        UPDATE trade_executions 
                        SET stop_loss = ?, tp1_taken = ?, peak_price = ?, partial_closed_lots = ?, partial_realized_usd = ?
                        WHERE ticket = ?
                        """,
                        (tracked.get("stop_loss"), int(tracked.get("tp1_taken", 0)), float(tracked.get("peak_price", open_price)), float(tracked.get("partial_closed_lots", 0.0)), float(tracked.get("partial_realized_usd", 0.0)), ticket),
                    )
                    conn.commit()

            except Exception as e:
                logger.error("Error managing position %s: %s", p, e)

        return actions_taken

    def _detect_and_notify_exits(self, current_open_positions: List[Dict[str, Any]], current_balance: float):
        """
        Detect closed positions by comparing active broker positions against tracked positions,
        and send Telegram Exit notifications with detailed PnL.
        """
        current_tickets = {str(p.get("ticket")) for p in current_open_positions if self._is_matching_symbol(str(p.get("symbol", "")), self.config.symbol)}
        closed_tickets = [t for t in self.tracked_positions.keys() if t not in current_tickets]
        now_str = datetime.now(timezone.utc).isoformat()

        for ticket in closed_tickets:
            trade = self.tracked_positions.pop(ticket, None)
            if not trade:
                continue

            symbol = trade.get("symbol", self.config.symbol)
            is_btc = "BTC" in symbol.upper()
            is_eth = "ETH" in symbol.upper()
            contract_size = 1.0 if (is_btc or is_eth) else 100.0
            side = trade.get("action", "BUY")
            entry_price = float(trade.get("entry_price", 0.0))
            lots = float(trade.get("lots", 0.01))
            comment = str(trade.get("comment", ""))
            
            # Query broker for actual closed deal receipt
            deal_info = None
            try:
                if hasattr(self.broker, "get_deal_history"):
                    deal_resp = self.broker.get_deal_history(ticket=ticket)
                    if deal_resp.get("status") == "SUCCESS":
                        deal_info = deal_resp
            except Exception as d_err:
                logger.debug("Failed to query deal history for #%s: %s", ticket, d_err)

            close_reason = "Take-Profit / Stop-Loss Triggered"
            if deal_info and deal_info.get("close_price") is not None:
                close_price = float(deal_info["close_price"])
                pnl = float(deal_info.get("net_pnl", deal_info.get("total_profit", 0.0)))
                now_str = deal_info.get("close_time") or now_str
                if deal_info.get("close_comment"):
                    close_reason = f"Broker Deal ({deal_info['close_comment']})"
            else:
                # Fallback to estimating exit price if broker deal history is unavailable
                try:
                    sym_info = self.broker.get_symbol_info(symbol)
                    if sym_info and sym_info.get("status") == "SUCCESS":
                        close_price = float(sym_info.get("bid" if side.upper() == "BUY" else "ask", entry_price))
                    else:
                        snapshot = get_live_gold_snapshot()
                        close_price = float(snapshot.get("spot_gold", {}).get("price", entry_price))
                except:
                    close_price = entry_price

                price_diff = (close_price - entry_price) if side.upper() == "BUY" else (entry_price - close_price)
                # Incorporate banked partial scale-out profit into final P&L
                partial_pnl = float(trade.get("partial_realized_usd") or 0.0)
                rem_lots = round(lots - float(trade.get("partial_closed_lots") or 0.0), 2)
                if rem_lots < 0.01:
                    rem_lots = lots
                rem_pnl = round(price_diff * rem_lots * contract_size, 2)
                pnl = round(partial_pnl + rem_pnl, 2)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE trade_executions 
                    SET status = 'CLOSED', close_price = ?, pnl = ?, close_time = ?
                    WHERE ticket = ?
                    """,
                    (close_price, pnl, now_str, ticket),
                )
                conn.commit()

            # If trade closed with a loss, enforce Loss Cooldown to prevent chop whipsaws
            loss_threshold = -1.00 if is_btc else -0.50
            if pnl < loss_threshold and self.config.loss_cooldown_minutes > 0:
                cooldown_secs = self.config.loss_cooldown_minutes * 60
                self.cooldown_until = datetime.now(timezone.utc).timestamp() + cooldown_secs
                self._save_state("cooldown_until_ts", str(self.cooldown_until))
                logger.info(
                    "🛡️ Trade #%s closed with loss ($%.2f). Enforcing %d-minute loss cooldown.",
                    ticket, pnl, self.config.loss_cooldown_minutes
                )

            # If trade was part of the Compounding Ladder or Flat Risk, record trade result
            is_managed_strategy_trade = (
                "compounding" in comment.lower()
                or "ladder" in comment.lower()
                or "flat" in comment.lower()
                or self.config.mode in ("COMPOUNDING_LADDER", "COMPOUNDING_HYBRID", "COMPOUNDING_TOUCH", "FLAT_RISK_TOUCH")
                or self.config.flat_risk
            )
            if is_managed_strategy_trade:
                try:
                    ladder = self.ladder_manager
                    be_cutoff = 1.00 if is_btc else (0.50 if is_eth else 0.35)
                    is_be = abs(pnl) <= be_cutoff
                    is_win = pnl > be_cutoff
                    ladder_update = ladder.record_trade_result(
                        win=is_win,
                        realized_pnl=pnl,
                        exit_price=close_price,
                        is_breakeven=is_be,
                        notes=f"AutoTrader Ticket #{ticket} closed at {close_price}",
                    )
                    mode_lbl = "Flat Risk ($75.44 Risk)" if ladder.ladder_mode == "FLAT_RISK" else f"Compounding Level {ladder_update.get('new_level')}"
                    logger.info(f"Execution logged after Ticket #{ticket} ({mode_lbl}): {ladder_update}")
                except Exception as e:
                    logger.error(f"Failed to update trading execution state on trade exit: {e}")

            if self.config.send_telegram_alerts:
                self.notifier.send_trade_exit(
                    ticket=ticket,
                    symbol=symbol,
                    side=side,
                    lots=lots,
                    entry_price=entry_price,
                    close_price=close_price,
                    pnl=pnl,
                    close_reason=close_reason,
                    account_balance=current_balance,
                )

    def get_recent_history(self, limit: int = 10) -> Dict[str, Any]:
        """Fetch latest scan logs and trade executions."""
        scans = []
        trades = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT * FROM scan_logs ORDER BY id DESC LIMIT ?", (limit,))
                scans = [dict(row) for row in cur.fetchall()]
                cur.execute("SELECT * FROM trade_executions ORDER BY id DESC LIMIT ?", (limit,))
                trades = [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error("Error reading history: %s", e)
            
        return {"scans": scans, "trades": trades}
