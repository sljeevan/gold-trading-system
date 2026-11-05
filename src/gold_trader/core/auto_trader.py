"""
Automated Live Market Scanner & Risk-Reward Trade Execution Engine for Gold.
Continuously scans multi-timeframe candles, evaluates institutional confluences,
calculates strict R:R setups, and executes trades on XM MT5 / Paper Broker with dynamic trade management
and real-time Telegram notifications.
"""
from __future__ import annotations

from contextlib import contextmanager
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
from gold_trader.core.data_feed import fetch_gold_candles, get_live_gold_snapshot
from gold_trader.core.strategy_engine import (
    StrategyConfig,
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
    max_open_positions: Optional[int] = 2  # Max open positions per asset (2 for runner compounding)
    max_unhedged_positions: Optional[int] = 1  # Max open positions with active unhedged risk (default: 1)
    allow_runner_compounding: bool = True  # Permit entering new trade once existing trade has locked Break-Even
    max_spread_pips: float = 4.0
    scan_interval_seconds: int = 60
    active_interval_seconds: int = 5
    enable_adaptive_dual_speed: bool = True
    enable_two_tier_polling: bool = True
    micro_polling_interval_seconds: float = 1.0
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
    runner_trail_buffer_r: Optional[float] = None  # Fixed USD trailing buffer active by default ($4.50 Gold, $350 BTC, $10 ETH)
    runner_target_2_multiplier: float = 3.00
    enable_news_catalyst_guard: bool = True
    news_guard_buffer_minutes: int = 15
    enable_ml_filter: bool = False
    ml_min_probability: float = 0.50
    ml_dynamic_sizing: bool = False
    ml_model_path: Optional[str] = None
    ml_required: bool = False
    ml_failure_policy: Literal["FAIL_CLOSED", "FAIL_OPEN"] = "FAIL_CLOSED"
    require_broker: bool = True
    adx_min_threshold: Optional[float] = None
    require_slope_alignment: bool = True
    allow_live_real: bool = False
    expected_account_id: Optional[str] = None
    trade_cap_failure_policy: Literal["FAIL_CLOSED", "FAIL_OPEN"] = "FAIL_CLOSED"
    news_guard_failure_policy: Literal["FAIL_CLOSED", "FAIL_OPEN"] = "FAIL_CLOSED"
    ladder_state_file: Optional[str] = None
    db_path: Optional[str] = None
    order_tag: Optional[str] = None



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
        elif "ETH" in self.config.symbol.upper():
            default_db = "~/.gold_trading_data/eth_auto_trader.db"
        else:
            default_db = "~/.gold_trading_data/auto_trader.db"
        self.db_path = os.path.expanduser(db_path or self.config.db_path or default_db)

        self._ensure_db()
        self.is_running = False
        self._load_active_tracked_positions()
        self.last_trade_bar: Optional[str] = self._get_state("last_trade_bar")
        cooldown_str = self._get_state("cooldown_until_ts")
        self.cooldown_until: Optional[float] = float(cooldown_str) if cooldown_str else None

        # Validate effective strategy configuration at daemon startup (Fail Fast)
        init_strat_cfg = StrategyConfig(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            strategy_mode=self.config.mode,
            donchian_period=self.config.donchian_period,
            rr_multiplier=self.config.min_risk_reward,
            trend_filter_ema=self.config.trend_filter_ema,
            trigger_mode=self.config.trigger_mode,
            adx_min_threshold=self.config.adx_min_threshold,
            require_slope_alignment=self.config.require_slope_alignment,
            pullback_discount_pips=self.config.pullback_discount_pips if self.config.use_pullback_limit else 0.0,
            use_pullback_limit=self.config.use_pullback_limit,
            min_win_probability=self.config.ml_min_probability,
            runner_t2_multiplier=self.config.runner_target_2_multiplier,
            broker=self.broker,
            broker_port=getattr(self.broker, "port", None),
        )
        init_strat_cfg.validate()
        self.strategy_hash = init_strat_cfg.get_hash()

        # Validate Unified Run Configuration with cryptographic hash & Git SHA
        from gold_trader.core.unified_config import UnifiedRunConfig
        self.run_config = UnifiedRunConfig(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            strategy_mode=self.config.mode,
            trigger_mode=self.config.trigger_mode,
            execution_mode="LIVE" if self.config.auto_execute else "PAPER",
            donchian_period=self.config.donchian_period,
            rr_multiplier=self.config.min_risk_reward,
            trend_filter_ema=self.config.trend_filter_ema,
            require_slope_alignment=self.config.require_slope_alignment,
            adx_min_threshold=self.config.adx_min_threshold,
            use_pullback_limit=self.config.use_pullback_limit,
            pullback_discount_pips=self.config.pullback_discount_pips if self.config.use_pullback_limit else 0.0,
            fixed_risk_usd=self.config.fixed_risk_usd,
            risk_pct_per_trade=self.config.risk_pct_per_trade,
            enable_partial_scale_out=self.config.enable_partial_scale_out,
            partial_scale_out_pct=self.config.partial_scale_out_pct,
            scale_out_r=self.config.scale_out_r,
            enable_runner_trailing=self.config.enable_runner_trailing,
            runner_trail_buffer_usd=self.config.runner_trail_buffer_usd,
            runner_trail_buffer_r=self.config.runner_trail_buffer_r,
            runner_target_2_multiplier=self.config.runner_target_2_multiplier,
            trail_to_breakeven_at_t1=self.config.trail_to_breakeven_at_t1,
            enable_news_catalyst_guard=self.config.enable_news_catalyst_guard,
            news_guard_buffer_minutes=self.config.news_guard_buffer_minutes,
            enable_ml_filter=self.config.enable_ml_filter,
            ml_min_probability=self.config.ml_min_probability,
            ml_model_path=self.config.ml_model_path,
            ml_required=self.config.ml_required,
            ml_failure_policy=self.config.ml_failure_policy,
            require_broker=self.config.require_broker,
            session_filter=self.config.session_filter,
            session_start_hour_utc=self.config.session_start_hour_utc,
            session_end_hour_utc=self.config.session_end_hour_utc,
            max_daily_trades=self.config.max_daily_trades,
            loss_cooldown_minutes=self.config.loss_cooldown_minutes,
        )
        self.run_config.validate()
        self.config_hash = self.run_config.config_hash
        self.git_sha = self.run_config.git_sha
        self.schema_version = self.run_config.schema_version

        logger.info(
            "Strategy Configuration validated: mode=%s, tf=%s, donchian=%d, trigger=%s, hash=%s, git_sha=%s",
            init_strat_cfg.strategy_mode,
            init_strat_cfg.timeframe,
            init_strat_cfg.donchian_period,
            init_strat_cfg.trigger_mode,
            self.config_hash,
            self.git_sha[:8],
        )

    @property
    def ladder_manager(self):
        """Flat Risk manager stub ensuring complete independence from compounding ladder."""
        fixed_risk = self.config.fixed_risk_usd
        class FlatRiskManagerStub:
            def __init__(self, fixed_risk_val: float):
                self.ladder_mode = "FLAT_RISK"
                self.fixed_risk_usd = fixed_risk_val
            def get_status(self):
                return {
                    "ladder_mode": "FLAT_RISK",
                    "compounding_enabled": False,
                    "starting_balance": 1000.0,
                    "current_balance": 1000.0,
                    "fixed_risk_usd": self.fixed_risk_usd,
                    "current_level": 18,
                }
            def record_trade_result(self, *args, **kwargs):
                return {"status": "SUCCESS", "mode": "FLAT_RISK"}
            def calculate_bracket_order(self, current_price: float, direction: str = "BUY", symbol: str = "XAUUSD", atr: Optional[float] = None, rr_ratio: float = 1.25, **kwargs):
                is_crypto = "BTC" in symbol.upper() or "ETH" in symbol.upper()
                contract_size = 1.0 if is_crypto else 100.0
                min_sl = 500.0 if "BTC" in symbol.upper() else (20.0 if "ETH" in symbol.upper() else 8.0)
                sl_dist = max(round(1.5 * (atr or min_sl), 2), min_sl)
                raw_lots = self.fixed_risk_usd / (sl_dist * contract_size)
                lots = max(0.01, round(raw_lots, 2))
                if "ETH" in symbol.upper():
                    lots = max(0.02, lots)

                actual_risk_usd = round(lots * sl_dist * contract_size, 2)
                if actual_risk_usd > self.fixed_risk_usd * 1.25:
                    logger.warning(
                        "Risk budget inflation note for %s: %s lots gives actual risk $%.2f vs target $%.2f",
                        symbol, lots, actual_risk_usd, self.fixed_risk_usd
                    )

                tp_dist = round(rr_ratio * sl_dist, 2)
                sl_price = round(current_price - sl_dist, 2) if direction == "BUY" else round(current_price + sl_dist, 2)
                tp_price = round(current_price + tp_dist, 2) if direction == "BUY" else round(current_price - tp_dist, 2)
                return {
                    "order_type": "MARKET",
                    "market_price": current_price,
                    "entry_price": current_price,
                    "lots": lots,
                    "stop_loss": sl_price,
                    "take_profit": tp_price,
                    "breakeven_trigger_price": tp_price,
                    "pullback_discount_pips": 0.0,
                    "sl_pips": sl_dist * 10,
                    "tp_pips": tp_dist * 10,
                    "expected_loss_usd": actual_risk_usd,
                    "expected_profit_usd": round(actual_risk_usd * rr_ratio, 2),
                    "reward_risk_ratio": rr_ratio,
                }
        return FlatRiskManagerStub(fixed_risk)

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

    def get_matching_open_positions(self) -> List[Dict[str, Any]]:
        """Fetch active broker positions matching this trader's symbol family."""
        try:
            positions = self.broker.get_open_positions()
            matched = [p for p in positions if self._is_matching_symbol(str(p.get("symbol", "")), self.config.symbol)]
            if self.config.order_tag:
                matched = [
                    p for p in matched
                    if str(p.get("ticket")) in self.tracked_positions or str(p.get("comment", "")).startswith(self.config.order_tag)
                ]
            else:
                matched = [
                    p for p in matched
                    if not str(p.get("comment", "")).startswith("YieldA")
                ]
            return matched
        except Exception as e:
            logger.debug(f"Failed to fetch open positions: {e}")
            return []

    def get_unhedged_open_positions(self) -> List[Dict[str, Any]]:
        """Fetch open positions matching this trader's symbol that have active downside risk (SL not at Break-Even)."""
        matching = self.get_matching_open_positions()
        unhedged = []
        for p in matching:
            side = str(p.get("side", "BUY")).upper()
            open_p = float(p.get("open_price", 0.0))
            sl = float(p.get("sl", 0.0) or 0.0)
            if side == "BUY":
                if sl < open_p or sl == 0.0:
                    unhedged.append(p)
            elif side == "SELL":
                if sl > open_p or sl == 0.0:
                    unhedged.append(p)
        return unhedged

    def has_open_position(self) -> bool:
        """Check if there is currently an active open position for this trader's symbol family."""
        return len(self.get_matching_open_positions()) > 0

    def manage_active_positions(self) -> Dict[str, Any]:
        """
        Fast-tick position management cycle (Adaptive 5-second interval):
        - Evaluates peak price excursion
        - Executes 50% scale-out at Target 1 (+1.0R) into banked cash
        - Adjusts Stop-Loss to Break-Even + offset (Risk-Free Runner)
        - Trails runner Stop-Loss with wide buffer behind peak excursion
        - Detects broker exit deal receipts and dispatches Telegram notifications
        - Bypasses 1H indicator calculations and database scan logging (zero DB bloat)
        """
        now = datetime.now(timezone.utc).isoformat()
        symbol = self.config.symbol
        try:
            acct = self.broker.get_account_info()
            balance = float(acct.get("balance", 0.0)) if acct else 0.0
            if balance <= 0:
                logger.warning("Broker returned invalid or non-positive balance (%s) in fast cycle. Failing closed.", balance)
                return {
                    "status": "FAST_MANAGE_SKIPPED",
                    "timestamp": now,
                    "symbol": symbol,
                    "reason": "Broker balance unavailable or invalid",
                }
        except Exception as e:
            logger.warning("Broker account info query failed in fast cycle: %s. Failing closed.", e)
            return {
                "status": "FAST_MANAGE_SKIPPED",
                "timestamp": now,
                "symbol": symbol,
                "reason": f"Broker account unavailable: {e}",
            }

        try:
            open_positions = self.broker.get_open_positions()
        except Exception as e:
            logger.debug("Failed to fetch open positions in fast cycle: %s", e)
            open_positions = None

        if open_positions is None:
            return {
                "status": "FAST_MANAGE_SKIPPED",
                "timestamp": now,
                "symbol": symbol,
                "reason": "Broker positions unavailable",
            }

        sym_open_positions = [p for p in open_positions if self._is_matching_symbol(str(p.get("symbol", "")), symbol)]
        if self.config.order_tag:
            sym_open_positions = [
                p for p in sym_open_positions
                if str(p.get("ticket")) in self.tracked_positions or str(p.get("comment", "")).startswith(self.config.order_tag)
            ]

        # 1. Detect closed positions (exits) and dispatch alerts / update state
        self._detect_and_notify_exits(open_positions, current_balance=balance)

        # 2. Manage existing open positions (Trailing SL / Breakeven / Partial scale out)
        managed_actions = self._manage_open_positions(open_positions)

        return {
            "status": "FAST_MANAGE",
            "timestamp": now,
            "symbol": symbol,
            "open_positions": sym_open_positions,
            "open_positions_count": len(sym_open_positions),
            "managed_actions": managed_actions,
            "decision": "ACTIVE_POSITION_MANAGED" if managed_actions else "MONITORING_ACTIVE_POSITION",
        }

    @contextmanager
    def _db_connection(self):
        """Context manager that yields an active SQLite connection and guarantees it is closed."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _ensure_db(self):
        dname = os.path.dirname(self.db_path)
        if dname:
            os.makedirs(dname, exist_ok=True)
        with self._db_connection() as conn:
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
                if "initial_stop_loss" not in cols:
                    conn.execute("ALTER TABLE trade_executions ADD COLUMN initial_stop_loss REAL")
                if "initial_risk_distance" not in cols:
                    conn.execute("ALTER TABLE trade_executions ADD COLUMN initial_risk_distance REAL")
                if "initial_lots" not in cols:
                    conn.execute("ALTER TABLE trade_executions ADD COLUMN initial_lots REAL")
                if "remaining_lots" not in cols:
                    conn.execute("ALTER TABLE trade_executions ADD COLUMN remaining_lots REAL")
                if "config_hash" not in cols:
                    conn.execute("ALTER TABLE trade_executions ADD COLUMN config_hash TEXT")
                if "git_sha" not in cols:
                    conn.execute("ALTER TABLE trade_executions ADD COLUMN git_sha TEXT")
                if "schema_version" not in cols:
                    conn.execute("ALTER TABLE trade_executions ADD COLUMN schema_version TEXT")

                # Scan logs columns for provenance
                cur.execute("PRAGMA table_info(scan_logs)")
                scan_cols = [c[1] for c in cur.fetchall()]
                if "config_hash" not in scan_cols:
                    conn.execute("ALTER TABLE scan_logs ADD COLUMN config_hash TEXT")
                if "git_sha" not in scan_cols:
                    conn.execute("ALTER TABLE scan_logs ADD COLUMN git_sha TEXT")

                # Data migration for existing trades
                conn.execute("UPDATE trade_executions SET initial_stop_loss = stop_loss WHERE initial_stop_loss IS NULL")
                conn.execute("UPDATE trade_executions SET initial_risk_distance = abs(entry_price - initial_stop_loss) WHERE initial_risk_distance IS NULL AND initial_stop_loss IS NOT NULL")
                conn.execute("UPDATE trade_executions SET initial_lots = lots WHERE initial_lots IS NULL")
                conn.execute("UPDATE trade_executions SET remaining_lots = round(lots - COALESCE(partial_closed_lots, 0.0), 2) WHERE remaining_lots IS NULL")
                conn.commit()
            except Exception as e:
                logger.debug("Table migration note: %s", e)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS engine_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS close_intents (
                    intent_id TEXT PRIMARY KEY,
                    ticket TEXT NOT NULL,
                    intended_lots REAL NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL,
                    closed_lots REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def _get_close_intent(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve persisted close intent by idempotency key."""
        try:
            with self._db_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                row = cur.execute("SELECT * FROM close_intents WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.debug("Close intent query note: %s", e)
            return None

    def _create_close_intent(self, ticket: str, intended_lots: float, trigger_reason: str, idempotency_key: str):
        """Record intent to close position before sending command to broker."""
        now_ts = datetime.now(timezone.utc).isoformat()
        intent_id = f"INTENT_{idempotency_key}"
        try:
            with self._db_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO close_intents
                    (intent_id, ticket, intended_lots, trigger_reason, idempotency_key, status, closed_lots, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (intent_id, str(ticket), float(intended_lots), trigger_reason, idempotency_key, "PENDING", 0.0, now_ts, now_ts)
                )
                conn.commit()
        except Exception as e:
            logger.error("Failed to create close intent %s: %s", idempotency_key, e)

    def _update_close_intent(self, idempotency_key: str, status: str, closed_lots: float = 0.0):
        """Update lifecycle state of close intent."""
        now_ts = datetime.now(timezone.utc).isoformat()
        try:
            with self._db_connection() as conn:
                conn.execute(
                    "UPDATE close_intents SET status = ?, closed_lots = ?, updated_at = ? WHERE idempotency_key = ?",
                    (status, float(closed_lots), now_ts, idempotency_key)
                )
                conn.commit()
        except Exception as e:
            logger.error("Failed to update close intent %s: %s", idempotency_key, e)

    def _reconcile_partial_close(self, ticket: str, intended_lots: float, initial_lots: float) -> Tuple[bool, float]:
        """Reconcile partial close against broker truth: deal history and remaining volume."""
        try:
            # 1. Query broker deal history
            deals_resp = self.broker.get_deal_history(ticket=ticket)
            if deals_resp and deals_resp.get("status") == "SUCCESS" and deals_resp.get("deals"):
                for d in deals_resp.get("deals", []):
                    d_vol = float(d.get("volume", 0.0))
                    if d_vol > 0 and abs(d_vol - intended_lots) < 0.01:
                        logger.info("Partial close verified via broker deal history (ticket #%s, volume %.2f)", ticket, d_vol)
                        return True, d_vol

            # 2. Query open positions remaining volume
            open_positions = self.broker.get_open_positions()
            for p in open_positions:
                if str(p.get("ticket")) == str(ticket):
                    current_lots = float(p.get("lots", 0.0))
                    expected_max = float(initial_lots) - intended_lots + 0.005
                    if current_lots <= expected_max:
                        closed_vol = round(float(initial_lots) - current_lots, 2)
                        logger.info("Partial close verified via volume reduction (initial: %.2f, current: %.2f)", initial_lots, current_lots)
                        return True, closed_vol if closed_vol > 0 else intended_lots
        except Exception as recon_err:
            logger.warning("Error reconciling partial close for #%s: %s", ticket, recon_err)

        return False, 0.0

    def _save_state(self, key: str, value: str):
        """Persist engine control state to database."""
        try:
            with self._db_connection() as conn:
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
            with self._db_connection() as conn:
                cur = conn.cursor()
                row = cur.execute("SELECT value FROM engine_state WHERE key = ?", (key,)).fetchone()
                return str(row[0]) if row else None
        except Exception:
            return None

    def _load_active_tracked_positions(self):
        """Load currently open tracked trades into memory."""
        self.tracked_positions: Dict[str, Dict[str, Any]] = {}
        try:
            with self._db_connection() as conn:
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
        cfg_hash = getattr(self, "config_hash", "")
        git_sha = getattr(self, "git_sha", "")
        try:
            with self._db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO scan_logs 
                    (timestamp, symbol, price, bias, setup_name, action, quality_score, rr_ratio, decision, details_json, config_hash, git_sha)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (now_str, symbol, price, bias, setup_name, action, quality_score, rr_ratio, decision, json.dumps(details or {}), cfg_hash, git_sha),
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
        
        # 0. Safety Lockout Check (Block further entries if previous trade had unverified SL/TP)
        if self._get_state("lockout") == "true" or getattr(self, "lockout", False):
            logger.critical("SAFETY LOCKOUT ENGAGED: Cannot trade due to unverified protective SL/TP failure. Manual intervention required.")
            self.log_scan(symbol, 0.0, "LOCKED_OUT", "None", "HOLD", 0, 0.0, "Safety lockout active: protective SL/TP was unverified", {})
            return {"status": "LOCKED_OUT", "reason": "Safety lockout active: protective SL/TP unverified", "timestamp": now}

        # 1. Check account state & open positions (Fail Closed)
        try:
            acct = self.broker.get_account_info()
            balance = float(acct.get("balance", 0.0)) if acct else 0.0
            if balance <= 0:
                logger.error("Broker account info returned invalid or non-positive balance (%s). Failing closed.", balance)
                self.log_scan(symbol, 0.0, "ERROR_BROKER_UNAVAILABLE", "None", "HOLD", 0, 0.0, f"Broker balance invalid ({balance})", {})
                return {"status": "BROKER_UNAVAILABLE", "reason": "Invalid or zero account balance", "timestamp": now}
        except Exception as e:
            logger.error("Broker account query failed: %s. Failing closed.", e)
            self.log_scan(symbol, 0.0, "ERROR_BROKER_UNAVAILABLE", "None", "HOLD", 0, 0.0, f"Broker account error: {e}", {})
            return {"status": "BROKER_UNAVAILABLE", "reason": f"Account query failed: {e}", "timestamp": now}

        # 1b. Live Real Account & Account Identity Execution Guard (P1-17)
        trade_mode = str(acct.get("trade_mode", "DEMO")).upper()
        if "LIVE" in trade_mode and not self.config.allow_live_real:
            logger.critical("LIVE REAL ACCOUNT DETECTED (%s) but allow_live_real is False! Blocking automated execution.", trade_mode)
            self.log_scan(symbol, 0.0, "REAL_ACCOUNT_BLOCKED", "None", "HOLD", 0, 0.0, "Live real execution blocked: allow_live_real=False", {})
            return {"status": "BLOCKED", "reason": "Live real account execution blocked without explicit opt-in (allow_live_real=False)", "timestamp": now}

        expected_acc = self.config.expected_account_id
        if expected_acc and str(acct.get("account_id", "")) != str(expected_acc):
            logger.critical("ACCOUNT ID MISMATCH: Connected to %s, expected %s! Blocking execution.", acct.get("account_id"), expected_acc)
            self.log_scan(symbol, 0.0, "ACCOUNT_MISMATCH", "None", "HOLD", 0, 0.0, f"Account mismatch: connected {acct.get('account_id')} != expected {expected_acc}", {})
            return {"status": "BLOCKED", "reason": f"Account ID mismatch (connected {acct.get('account_id')} != expected {expected_acc})", "timestamp": now}

        try:
            open_positions = self.broker.get_open_positions()
        except Exception as e:
            logger.error("Could not fetch open positions from broker: %s. Failing closed.", e)
            open_positions = None

        if open_positions is None:
            logger.error("Open positions retrieval failed. Failing closed to prevent unverified duplicate orders.")
            self.log_scan(symbol, 0.0, "ERROR_POSITIONS_UNAVAILABLE", "None", "HOLD", 0, 0.0, "Broker open positions unverified", {})
            return {"status": "BROKER_UNAVAILABLE", "reason": "Open positions unverified", "timestamp": now}

        if hasattr(self.broker, "get_pending_orders"):
            try:
                pending_orders = self.broker.get_pending_orders()
            except Exception as e:
                logger.error("Could not fetch pending orders from broker: %s. Failing closed.", e)
                pending_orders = None

            if pending_orders is None:
                logger.error("Pending orders retrieval failed. Failing closed to prevent unverified duplicate orders.")
                self.log_scan(symbol, 0.0, "ERROR_ORDERS_UNAVAILABLE", "None", "HOLD", 0, 0.0, "Broker pending orders unverified", {})
                return {"status": "BROKER_UNAVAILABLE", "reason": "Pending orders unverified", "timestamp": now}
        else:
            pending_orders = []

        # 2. Check for Closed Positions (Exit Alert Detection)
        self._detect_and_notify_exits(open_positions, current_balance=balance)
        # 3. Manage existing open positions (Trailing SL / Breakeven)
        managed_actions = self._manage_open_positions(open_positions)

        # 4. Generate Trade Setup based on mode & validated StrategyConfig
        strat_cfg = StrategyConfig(
            symbol=symbol,
            timeframe=self.config.timeframe,  # Propagates exact configured timeframe (15m, 1h, etc.)
            strategy_mode=self.config.mode,
            donchian_period=self.config.donchian_period,
            rr_multiplier=self.config.min_risk_reward,
            trend_filter_ema=self.config.trend_filter_ema,
            trigger_mode=self.config.trigger_mode,
            adx_min_threshold=self.config.adx_min_threshold,
            require_slope_alignment=self.config.require_slope_alignment,
            pullback_discount_pips=self.config.pullback_discount_pips if self.config.use_pullback_limit else 0.0,
            use_pullback_limit=self.config.use_pullback_limit,
            min_win_probability=self.config.ml_min_probability,
            runner_t2_multiplier=self.config.runner_target_2_multiplier,
            broker=self.broker,
            broker_port=getattr(self.broker, "port", None),
        )
        strat_cfg.validate()

        is_touch = self.config.mode in ("FLAT_RISK_TOUCH", "COMPOUNDING_TOUCH", "HIGH_VELOCITY_TOUCH")
        is_compounding_legacy = self.config.mode in ("COMPOUNDING_LADDER", "COMPOUNDING_HYBRID")
        if is_touch or is_compounding_legacy:
            order_type = "MARKET"
            setup = generate_compounding_ladder_setup(
                symbol=symbol,
                timeframe=self.config.timeframe,
                ladder_manager=self.ladder_manager,
                strategy_config=strat_cfg,
                require_broker=self.config.require_broker,
            )
            if "error" in setup:
                self.log_scan(symbol, 0.0, "ERROR", "None", "HOLD", 0, 0.0, f"Error: {setup.get('error')}", setup)
                return {"status": "ERROR", "message": setup.get("error"), "timestamp": now}

            order_type = str(setup.get("order_type", "MARKET"))
            price = float(setup.get("entry_price") or setup.get("market_price") or setup.get("setup_details", {}).get("entry_price") or setup.get("bracket_order_specs", {}).get("market_price") or 0.0)
            bias = str(setup.get("bias") or setup.get("setup_details", {}).get("bias", "NEUTRAL"))
            default_name = f"{self.config.timeframe.upper()} Donchian {self.config.donchian_period} {self.config.trigger_mode} (Flat Risk ${self.config.fixed_risk_usd:.2f})"
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
                require_broker=self.config.require_broker,
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
                require_broker=self.config.require_broker,
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
        elif action == "PASS":
            rejection_reasons.append(str(setup.get("reason", "Filtered by ML gatekeeper or minimum probability threshold.")))

        if quality_score < self.config.min_quality_score:
            rejection_reasons.append(f"Quality score {quality_score} < minimum {self.config.min_quality_score}.")

        if rr_t1 < self.config.min_risk_reward:
            rejection_reasons.append(f"R:R ratio {rr_t1:.2f} < minimum {self.config.min_risk_reward:.2f}.")

        sym_open_positions = [p for p in open_positions if self._is_matching_symbol(str(p.get("symbol", "")), symbol)]
        if self.config.order_tag:
            sym_open_positions = [
                p for p in sym_open_positions
                if str(p.get("ticket")) in self.tracked_positions or str(p.get("comment", "")).startswith(self.config.order_tag)
            ]
        else:
            sym_open_positions = [
                p for p in sym_open_positions
                if not str(p.get("comment", "")).startswith("YieldA")
            ]
        sym_pending_orders = [po for po in pending_orders if self._is_matching_symbol(str(po.get("symbol", "")), symbol)]
        if self.config.order_tag:
            sym_pending_orders = [
                po for po in sym_pending_orders
                if str(po.get("comment", "")).startswith(self.config.order_tag)
            ]
        else:
            sym_pending_orders = [
                po for po in sym_pending_orders
                if not str(po.get("comment", "")).startswith("YieldA")
            ]

        active_exposure = len(sym_open_positions) + len(sym_pending_orders)
        if self.config.max_open_positions and self.config.max_open_positions > 0:
            if active_exposure >= self.config.max_open_positions:
                rejection_reasons.append(f"Max active exposure reached ({len(sym_open_positions)} open, {len(sym_pending_orders)} pending >= {self.config.max_open_positions}).")

            # Check if we already have an open position in the same direction (unless runner compounding is enabled)
            if not self.config.allow_runner_compounding:
                same_direction = [p for p in sym_open_positions if p.get("side") == action]
                if same_direction:
                    rejection_reasons.append(f"Position already open in direction {action} (Ticket #{same_direction[0].get('ticket')}).")

        # Check Unhedged Active Risk Gate (Risk-Free Runner Compounding)
        if self.config.max_unhedged_positions and self.config.max_unhedged_positions > 0:
            unhedged_positions = []
            for p in sym_open_positions:
                p_side = str(p.get("side", "BUY")).upper()
                p_open = float(p.get("open_price", 0.0))
                p_sl = float(p.get("sl", 0.0) or 0.0)
                # Position is unhedged if stop-loss has not yet been advanced to or beyond entry price
                if p_side == "BUY":
                    if p_sl < p_open or p_sl == 0.0:
                        unhedged_positions.append(p)
                elif p_side == "SELL":
                    if p_sl > p_open or p_sl == 0.0:
                        unhedged_positions.append(p)

            if len(unhedged_positions) >= self.config.max_unhedged_positions:
                rejection_reasons.append(
                    f"Unhedged Risk Gate: Active position #{unhedged_positions[0].get('ticket')} has not locked Break-Even yet. "
                    f"Awaiting Target 1 scale-out before permitting runner compounding."
                )

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

        if not is_crypto:
            is_weekend = (
                (now_dt.weekday() == 4 and now_dt.hour >= 21)
                or now_dt.weekday() == 5
                or (now_dt.weekday() == 6 and (now_dt.hour < 22 or (now_dt.hour == 22 and now_dt.minute < 5)))
            )
            if is_weekend:
                rejection_reasons.append("Market Closed: Weekend hours (Gold market reopens Sunday 22:05 UTC).")

        if self.config.session_filter:
            if not (self.config.session_start_hour_utc <= now_dt.hour < self.config.session_end_hour_utc):
                rejection_reasons.append(
                    f"Session Filter: Current time {now_dt.strftime('%H:%M UTC')} is outside liquid London/NY hours "
                    f"({self.config.session_start_hour_utc:02d}:00 - {self.config.session_end_hour_utc:02d}:00 UTC). Avoiding Asian chop."
                )

        # Check Daily Trade Cap (Max daily trades to preserve capital)
        today_date_str = now_dt.strftime("%Y-%m-%d")
        daily_trades_count = 0
        trade_cap_query_failed = False
        try:
            with self._db_connection() as conn:
                cur = conn.cursor()
                row = cur.execute(
                    "SELECT COUNT(*) FROM trade_executions WHERE (timestamp LIKE ? OR close_time LIKE ?)",
                    (f"{today_date_str}%", f"{today_date_str}%"),
                ).fetchone()
                daily_trades_count = row[0] if row else 0
        except Exception as e:
            logger.error("Failed to query daily trade count: %s", e)
            trade_cap_query_failed = True

        if trade_cap_query_failed and self.config.trade_cap_failure_policy == "FAIL_CLOSED":
            rejection_reasons.append("Guard Unavailable: Failed to verify daily trade cap from database (trade_cap_failure_policy=FAIL_CLOSED).")
        elif self.config.max_daily_trades > 0 and daily_trades_count >= self.config.max_daily_trades:
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
                logger.error("Failed to evaluate news catalyst guard: %s", e)
                if self.config.news_guard_failure_policy == "FAIL_CLOSED":
                    rejection_reasons.append(f"Guard Unavailable: News catalyst calendar query failed ({e}). Fail-closed policy active.")

        # Check Machine Learning High-Probability Meta-Labeling Gate
        ml_eval = None
        if (self.config.enable_ml_filter or self.config.ml_required) and action in ("BUY", "SELL"):
            ml_failed = False
            ml_failure_reason = ""
            try:
                from gold_trader.core.ml_filter import GoldMLPredictor
                predictor = GoldMLPredictor(model_path=self.config.ml_model_path, symbol=symbol)
                candles_eval = fetch_gold_candles(symbol, period="1mo", interval=self.config.timeframe, require_broker=self.config.require_broker)
                if not candles_eval or len(candles_eval) < 200:
                    ml_failed = True
                    ml_failure_reason = f"ML_UNAVAILABLE: Insufficient history ({len(candles_eval) if candles_eval else 0} bars < 200 required)."
                else:
                    ml_eval = predictor.predict_trade_probability(
                        candles_eval,
                        direction=action,
                        min_probability_threshold=self.config.ml_min_probability,
                        use_closed_bar=True,
                    )
                    if "error" in ml_eval or ml_eval.get("status") == "ERROR":
                        ml_failed = True
                        ml_failure_reason = f"ML_UNAVAILABLE: {ml_eval.get('error', 'Prediction error')}"
                    else:
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
                ml_failed = True
                ml_failure_reason = f"ML_UNAVAILABLE: Prediction exception: {ml_err}"
                logger.error("ML Filter gate evaluation error: %s", ml_err)

            if ml_failed:
                if self.config.ml_required or self.config.ml_failure_policy == "FAIL_CLOSED":
                    rejection_reasons.append(ml_failure_reason)
                    logger.warning("🛡️ ML Filter FAIL_CLOSED engaged: %s. Order blocked.", ml_failure_reason)
                else:
                    logger.warning("ML Filter FAIL_OPEN engaged: %s. Permitting trade under optional mode.", ml_failure_reason)

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
                                    tp2_dist = abs(tp2 - price) if tp2 else round(self.config.runner_target_2_multiplier * sl_dist, 2)
                                    price = broker_entry
                                    if action == "BUY":
                                        sl = round(broker_entry - sl_dist, 2)
                                        tp1 = round(broker_entry + tp_dist, 2)
                                        tp2 = round(broker_entry + tp2_dist, 2) if self.config.enable_runner_trailing else tp1
                                    else:
                                        sl = round(broker_entry + sl_dist, 2)
                                        tp1 = round(broker_entry - tp_dist, 2)
                                        tp2 = round(broker_entry - tp2_dist, 2) if self.config.enable_runner_trailing else tp1
                        except Exception as cal_err:
                            logger.debug("Broker price calibration: %s", cal_err)
                    else:
                        # For LIMIT execution: keep precise 8-pip pullback discount and calculated SL/TP
                        price = round(price, 2)
                        if sl is not None:
                            sl = round(float(sl), 2)
                        if tp1 is not None:
                            tp1 = round(float(tp1), 2)
                        if tp2 is not None:
                            tp2 = round(float(tp2), 2)
                        elif tp1 is not None:
                            limit_sl_d = abs(price - sl) if sl else (500.0 if "BTC" in symbol.upper() else (20.0 if "ETH" in symbol.upper() else 8.0))
                            tp2 = round(price + (self.config.runner_target_2_multiplier * limit_sl_d), 2) if action == "BUY" else round(price - (self.config.runner_target_2_multiplier * limit_sl_d), 2)
                        else:
                            tp2 = tp1

                    order_comment = f"{self.config.order_tag}-{symbol}" if self.config.order_tag else f"AI-FlatRisk-{symbol}"
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
                    res_status = order_result.get("status")
                    if res_status == "SUCCESS":
                        # Verify that protective SL was confirmed if sl was requested
                        order_sl = order_result.get("sl")
                        if sl is not None and float(sl) > 0 and (order_sl is None or float(order_sl) <= 0):
                            logger.critical("🚨 Protective SL missing from confirmed trade! Triggering emergency lockout.")
                            self.lockout = True
                            self._save_state("lockout", "true")
                            if self.config.send_telegram_alerts:
                                self.notifier.send_error_alert(f"🚨 CRITICAL SAFETY BREACH: Trade on {symbol} lacks verified SL! Emergency lockout active.")
                            return {"status": "EMERGENCY_PROTECTION_FAILED", "message": "Position lacks verified SL", "timestamp": now}

                        ticket = str(order_result.get("ticket"))
                        filled_price = float(order_result.get("price", price))
                        executed_lots = float(order_result.get("actual_lots") or order_result.get("lots") or suggested_lots)
                        
                        if candle_time:
                            self.last_trade_bar = str(candle_time)
                            self._save_state("last_trade_bar", str(candle_time))
                            logger.info(f"🔒 Bar Lock engaged for 15m candle bar: {candle_time}")
                        
                        is_btc_t = "BTC" in symbol.upper()
                        is_eth_t = "ETH" in symbol.upper()
                        default_sl_d = 500.0 if is_btc_t else (20.0 if is_eth_t else 8.00)
                        trade_risk_d = max(abs(filled_price - sl), default_sl_d) if sl else default_sl_d
                        calc_stage1_tp = round(filled_price + (self.config.scale_out_r * trade_risk_d), 2) if action == "BUY" else round(filled_price - (self.config.scale_out_r * trade_risk_d), 2)
                        calc_stage2_tp = round(filled_price + (self.config.runner_target_2_multiplier * trade_risk_d), 2) if action == "BUY" else round(filled_price - (self.config.runner_target_2_multiplier * trade_risk_d), 2)

                        trade_record = {
                            "ticket": ticket,
                            "symbol": symbol,
                            "action": action,
                            "lots": executed_lots,
                            "entry_price": filled_price,
                            "stop_loss": sl,
                            "target_1": calc_stage1_tp,
                            "target_2": calc_stage2_tp if self.config.enable_runner_trailing else tp2,
                            "rr_ratio": rr_t1,
                            "comment": order_comment,
                            "status": "PENDING" if order_type == "LIMIT" else "OPEN",
                            "timestamp": now,
                            "tp1_taken": 0,
                            "peak_price": filled_price,
                            "partial_closed_lots": 0.0,
                            "partial_realized_usd": 0.0,
                            "initial_stop_loss": sl,
                            "initial_risk_distance": trade_risk_d,
                            "initial_lots": executed_lots,
                            "remaining_lots": executed_lots,
                            "config_hash": getattr(self, "config_hash", ""),
                            "git_sha": getattr(self, "git_sha", ""),
                            "schema_version": getattr(self, "schema_version", "2.0.0"),
                        }
                        
                        with self._db_connection() as conn:
                            conn.execute(
                                """
                                INSERT OR REPLACE INTO trade_executions
                                (timestamp, ticket, symbol, action, lots, entry_price, stop_loss, target_1, target_2, rr_ratio, status, comment, tp1_taken, peak_price, partial_closed_lots, partial_realized_usd, initial_stop_loss, initial_risk_distance, initial_lots, remaining_lots, config_hash, git_sha, schema_version)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (now, ticket, symbol, action, executed_lots, filled_price, sl, calc_stage1_tp, calc_stage2_tp if self.config.enable_runner_trailing else tp2, rr_t1, "OPEN", order_comment, 0, filled_price, 0.0, 0.0, sl, trade_risk_d, executed_lots, executed_lots, getattr(self, "config_hash", ""), getattr(self, "git_sha", ""), getattr(self, "schema_version", "2.0.0")),
                            )
                            conn.commit()

                        self.tracked_positions[ticket] = trade_record

                        if self.config.send_telegram_alerts:
                            self.notifier.send_trade_entry(order=order_result, setup=setup)
                    elif res_status == "EMERGENCY_PROTECTION_FAILED" or order_result.get("lockout"):
                        logger.critical("🚨 EMERGENCY: Order failed protection verification! Lockout engaged. Details: %s", order_result.get("message"))
                        self.lockout = True
                        self._save_state("lockout", "true")
                        if self.config.send_telegram_alerts:
                            self.notifier.send_error_alert(f"🚨 CRITICAL SAFETY BREACH: {order_result.get('message')}. System LOCKED OUT.")
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
                import sys
                subprocess.Popen(
                    [sys.executable, "/home/mcpuser/MCP/gold-trading-system/scripts/send_daily_report.py"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._save_state("daily_report_date", today_date_str)
                logger.info(f"📊 Dispatched daily session report for {today_date_str}")
            except Exception as dr_err:
                logger.error(f"Failed to trigger daily report: {dr_err}")

        armed_trigger = None
        if not order_result and action == "HOLD":
            armed_trigger = self._check_and_arm_setup(setup, balance)
            if armed_trigger:
                act = armed_trigger["action"]
                tp = armed_trigger["trigger_price"]
                p_win = armed_trigger.get("ml_prob", 0.50)
                logger.info(
                    f"🎯 [Tier 1 Armed] Ready for 1-Second Touch Trigger: {symbol} {act} @ ${tp:.2f} "
                    f"(SL Dist: ${armed_trigger['sl_dist']:.2f}, Lots: {armed_trigger['lots']}, ML: {p_win:.1%})"
                )

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
            "armed_trigger": armed_trigger,
        }

    def _check_and_arm_setup(self, setup: Dict[str, Any], balance: float) -> Optional[Dict[str, Any]]:
        """
        Evaluate if a pending Donchian breakout touch setup can be armed for Tier 2 1-second micro-loop execution.
        Pre-screens: 50 EMA trend, SuperTrend, 50 EMA slope alignment, ADX regime, session filter, daily cap,
        bar lock, loss cooldown, news guard, and ML probability threshold.
        """
        if not getattr(self.config, "enable_two_tier_polling", True):
            return None

        is_touch = self.config.mode in ("FLAT_RISK_TOUCH", "COMPOUNDING_TOUCH", "HIGH_VELOCITY_TOUCH")
        if not is_touch:
            return None

        setup_details = setup.get("setup_details", {})
        d_up = setup_details.get("donchian_upper")
        d_low = setup_details.get("donchian_lower")
        st_dir = setup_details.get("supertrend_direction", 0)
        bias = str(setup_details.get("bias", "NEUTRAL")).upper()
        candle_time = setup.get("candle_time")

        if d_up is None or d_low is None or not candle_time:
            return None

        d_up = float(d_up)
        d_low = float(d_low)

        cand_action = None
        trig_price = 0.0
        if bias == "BULLISH" and st_dir == 1:
            cand_action = "BUY"
            trig_price = d_up
        elif bias == "BEARISH" and st_dir == -1:
            cand_action = "SELL"
            trig_price = d_low
        else:
            return None

        symbol = self.config.symbol
        now_dt = datetime.now(timezone.utc)
        now_ts = now_dt.timestamp()

        # 1. Bar Lock
        if self.config.one_trade_per_bar and candle_time:
            if self.last_trade_bar == str(candle_time):
                return None

        # 2. Loss Cooldown
        if self.cooldown_until and now_ts < self.cooldown_until:
            return None

        # 3. Weekend / Session Filter
        is_crypto = "BTC" in symbol.upper() or "ETH" in symbol.upper()
        if not is_crypto:
            is_weekend = (
                (now_dt.weekday() == 4 and now_dt.hour >= 21)
                or now_dt.weekday() == 5
                or (now_dt.weekday() == 6 and (now_dt.hour < 22 or (now_dt.hour == 22 and now_dt.minute < 5)))
            )
            if is_weekend:
                return None

        if self.config.session_filter:
            if not (self.config.session_start_hour_utc <= now_dt.hour < self.config.session_end_hour_utc):
                return None

        # 4. Daily Trade Cap
        today_date_str = now_dt.strftime("%Y-%m-%d")
        daily_trades_count = 0
        try:
            with self._db_connection() as conn:
                cur = conn.cursor()
                row = cur.execute(
                    "SELECT COUNT(*) FROM trade_executions WHERE (timestamp LIKE ? OR close_time LIKE ?)",
                    (f"{today_date_str}%", f"{today_date_str}%"),
                ).fetchone()
                daily_trades_count = row[0] if row else 0
        except Exception:
            daily_trades_count = 0

        if self.config.max_daily_trades > 0 and daily_trades_count >= self.config.max_daily_trades:
            return None

        # 5. News Catalyst Guard
        if self.config.enable_news_catalyst_guard:
            try:
                from gold_trader.core.openbb_feed import check_high_impact_news_window
                news_status = check_high_impact_news_window(buffer_minutes=self.config.news_guard_buffer_minutes)
                if news_status.get("is_in_news_window"):
                    return None
            except Exception:
                if self.config.news_guard_failure_policy == "FAIL_CLOSED":
                    return None

        # 6. 50 EMA Slope Alignment Guard
        if self.config.require_slope_alignment:
            ema50_slope = float(setup_details.get("ema50_slope", 0.0))
            if cand_action == "BUY" and ema50_slope < 0.0:
                return None
            elif cand_action == "SELL" and ema50_slope > 0.0:
                return None

        # 7. ADX Regime Guard
        if self.config.adx_min_threshold is not None:
            adx_val = float(setup_details.get("adx", 25.0))
            if adx_val < self.config.adx_min_threshold:
                return None

        # 8. ML Filter Gatekeeper
        ml_prob = 0.50
        if self.config.enable_ml_filter or self.config.ml_required:
            try:
                from gold_trader.core.ml_filter import GoldMLPredictor
                predictor = GoldMLPredictor(model_path=self.config.ml_model_path, symbol=symbol)
                candles_eval = fetch_gold_candles(symbol, period="1mo", interval=self.config.timeframe, require_broker=self.config.require_broker)
                if not candles_eval or len(candles_eval) < 200:
                    return None
                ml_eval = predictor.predict_trade_probability(
                    candles_eval,
                    direction=cand_action,
                    min_probability_threshold=self.config.ml_min_probability,
                    use_closed_bar=True,
                )
                if ml_eval.get("status") == "ERROR" or not ml_eval.get("is_high_conviction"):
                    return None
                ml_prob = float(ml_eval.get("probability_win", 0.50))
            except Exception:
                if self.config.ml_required or self.config.ml_failure_policy == "FAIL_CLOSED":
                    return None

        # 9. Stop Loss & Target Distance
        is_btc = "BTC" in symbol.upper()
        is_eth = "ETH" in symbol.upper()
        min_pts = 500.0 if is_btc else (20.0 if is_eth else 8.0)
        atr_val = float(setup_details.get("atr") or min_pts)
        sl_dist = max(1.5 * atr_val, min_pts)
        tp1_dist = sl_dist * self.config.min_risk_reward
        tp2_dist = sl_dist * self.config.runner_target_2_multiplier

        # 10. Position Sizing
        contract_size = 1.0 if (is_btc or is_eth) else 100.0
        if self.config.flat_risk:
            planned_risk = self.config.fixed_risk_usd
            lots = round(planned_risk / (sl_dist * contract_size), 2)
            lots = max(0.01, lots)
        else:
            planned_risk = balance * (self.config.risk_pct_per_trade / 100.0)
            lots = calculate_lot_size(
                account_balance=balance,
                risk_pct=self.config.risk_pct_per_trade,
                entry_price=trig_price,
                stop_loss=trig_price - sl_dist if cand_action == "BUY" else trig_price + sl_dist,
                contract_size=contract_size,
            )
        if is_eth:
            lots = max(0.02, lots)

        armed = {
            "action": cand_action,
            "trigger_price": trig_price,
            "trigger_condition": "BID_GTE" if cand_action == "BUY" else "ASK_LTE",
            "sl_dist": sl_dist,
            "tp1_dist": tp1_dist,
            "tp2_dist": tp2_dist,
            "lots": lots,
            "candle_time": candle_time,
            "setup_name": f"{self.config.timeframe.upper()} Donchian {self.config.donchian_period} Touch (Flat Risk ${self.config.fixed_risk_usd:.2f})",
            "rr_t1": self.config.min_risk_reward,
            "ml_prob": ml_prob,
        }
        return armed

    def execute_armed_order(self, armed_trigger: Dict[str, Any], sym_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an armed intrabar trigger immediately via Tier 2 1-second high-velocity execution.
        """
        now = datetime.now(timezone.utc).isoformat()
        symbol = self.config.symbol
        action = armed_trigger["action"]
        suggested_lots = armed_trigger["lots"]
        sl_dist = armed_trigger["sl_dist"]
        tp1_dist = armed_trigger["tp1_dist"]
        tp2_dist = armed_trigger["tp2_dist"]
        rr_t1 = armed_trigger.get("rr_t1", self.config.min_risk_reward)
        candle_time = armed_trigger.get("candle_time")
        setup_name = armed_trigger.get("setup_name", f"{self.config.timeframe.upper()} Donchian 10 Touch")

        b_bid = float(sym_info.get("bid", 0.0))
        b_ask = float(sym_info.get("ask", 0.0))
        broker_entry = b_ask if action == "BUY" else b_bid

        if action == "BUY":
            sl = round(broker_entry - sl_dist, 2)
            tp1 = round(broker_entry + tp1_dist, 2)
            tp2 = round(broker_entry + tp2_dist, 2) if self.config.enable_runner_trailing else tp1
        else:
            sl = round(broker_entry + sl_dist, 2)
            tp1 = round(broker_entry - tp1_dist, 2)
            tp2 = round(broker_entry - tp2_dist, 2) if self.config.enable_runner_trailing else tp1

        order_comment = f"{self.config.order_tag}-{symbol}" if self.config.order_tag else f"AI-FlatRisk-{symbol}"
        actual_tp = tp2 if (self.config.enable_partial_scale_out or self.config.enable_runner_trailing) and tp2 else tp1

        try:
            order_result = self.broker.place_order(
                symbol="GOLD" if "GOLD" in symbol else symbol,
                side=action,
                lots=suggested_lots,
                sl=sl,
                tp=actual_tp,
                comment=order_comment,
                order_type="MARKET",
            )
            res_status = order_result.get("status")
            if res_status == "SUCCESS":
                order_sl = order_result.get("sl")
                if sl is not None and float(sl) > 0 and (order_sl is None or float(order_sl) <= 0):
                    logger.critical("🚨 Protective SL missing from confirmed trade! Triggering emergency lockout.")
                    self.lockout = True
                    self._save_state("lockout", "true")
                    if self.config.send_telegram_alerts:
                        self.notifier.send_error_alert(f"🚨 CRITICAL SAFETY BREACH: Trade on {symbol} lacks verified SL! Emergency lockout active.")
                    return {"status": "EMERGENCY_PROTECTION_FAILED", "message": "Position lacks verified SL", "timestamp": now}

                ticket = str(order_result.get("ticket"))
                filled_price = float(order_result.get("price", broker_entry))
                executed_lots = float(order_result.get("actual_lots") or order_result.get("lots") or suggested_lots)

                if candle_time:
                    self.last_trade_bar = str(candle_time)
                    self._save_state("last_trade_bar", str(candle_time))
                    logger.info(f"🔒 Bar Lock engaged for candle bar: {candle_time}")

                is_btc_t = "BTC" in symbol.upper()
                is_eth_t = "ETH" in symbol.upper()
                default_sl_d = 500.0 if is_btc_t else (20.0 if is_eth_t else 8.00)
                trade_risk_d = max(abs(filled_price - sl), default_sl_d) if sl else default_sl_d
                calc_stage1_tp = round(filled_price + (self.config.scale_out_r * trade_risk_d), 2) if action == "BUY" else round(filled_price - (self.config.scale_out_r * trade_risk_d), 2)
                calc_stage2_tp = round(filled_price + (self.config.runner_target_2_multiplier * trade_risk_d), 2) if action == "BUY" else round(filled_price - (self.config.runner_target_2_multiplier * trade_risk_d), 2)

                trade_record = {
                    "ticket": ticket,
                    "symbol": symbol,
                    "action": action,
                    "lots": executed_lots,
                    "entry_price": filled_price,
                    "stop_loss": sl,
                    "target_1": calc_stage1_tp,
                    "target_2": calc_stage2_tp if self.config.enable_runner_trailing else tp2,
                    "rr_ratio": rr_t1,
                    "comment": order_comment,
                    "status": "OPEN",
                    "timestamp": now,
                    "tp1_taken": 0,
                    "peak_price": filled_price,
                    "partial_closed_lots": 0.0,
                    "partial_realized_usd": 0.0,
                    "initial_stop_loss": sl,
                    "initial_risk_distance": trade_risk_d,
                    "initial_lots": executed_lots,
                    "remaining_lots": executed_lots,
                    "config_hash": getattr(self, "config_hash", ""),
                    "git_sha": getattr(self, "git_sha", ""),
                    "schema_version": getattr(self, "schema_version", "2.0.0"),
                }

                with self._db_connection() as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO trade_executions
                        (timestamp, ticket, symbol, action, lots, entry_price, stop_loss, target_1, target_2, rr_ratio, status, comment, tp1_taken, peak_price, partial_closed_lots, partial_realized_usd, initial_stop_loss, initial_risk_distance, initial_lots, remaining_lots, config_hash, git_sha, schema_version)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (now, ticket, symbol, action, executed_lots, filled_price, sl, calc_stage1_tp, calc_stage2_tp if self.config.enable_runner_trailing else tp2, rr_t1, "OPEN", order_comment, 0, filled_price, 0.0, 0.0, sl, trade_risk_d, executed_lots, executed_lots, getattr(self, "config_hash", ""), getattr(self, "git_sha", ""), getattr(self, "schema_version", "2.0.0")),
                    )
                    conn.commit()

                self.tracked_positions[ticket] = trade_record

                if self.config.send_telegram_alerts:
                    setup_alert_mock = {
                        "symbol": symbol,
                        "action": action,
                        "entry_price": filled_price,
                        "stop_loss": sl,
                        "target_1": calc_stage1_tp,
                        "target_2": calc_stage2_tp if self.config.enable_runner_trailing else tp2,
                        "setup_name": setup_name,
                        "lots": executed_lots,
                    }
                    self.notifier.send_trade_entry(order=order_result, setup=setup_alert_mock)

                return {"status": "SUCCESS", "order_result": order_result, "trade_record": trade_record}
            else:
                return {"status": "ERROR", "order_result": order_result, "message": order_result.get("message")}
        except Exception as e:
            logger.error("Failed to execute armed order: %s", e)
            return {"status": "ERROR", "message": str(e)}

    def _manage_open_positions(self, open_positions: List[Dict[str, Any]]) -> List[str]:
        """
        Institutional Two-Stage Position Management:
        1. Stage 1: Scale out 50% volume at Target 1 (+1.0R) into realized cash.
        2. Stage 2: Move remaining SL to Break-Even + buffer (Risk-Free Runner).
        3. Stage 3: Trail runner with wide ATR-cleared buffer behind peak excursion up to Target 2.
        """
        actions_taken = []
        if not (self.config.enable_partial_scale_out or self.config.trail_to_breakeven_at_t1 or self.config.enable_runner_trailing):
            return actions_taken

        for p in open_positions:
            try:
                ticket = str(p.get("ticket"))
                p_sym = str(p.get("symbol", ""))
                if not self._is_matching_symbol(p_sym, self.config.symbol):
                    continue
                if self.config.order_tag:
                    if ticket not in self.tracked_positions and not str(p.get("comment", "")).startswith(self.config.order_tag):
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
                        "initial_stop_loss": initial_sl,
                        "initial_risk_distance": initial_risk_dist,
                        "initial_lots": cur_lots,
                        "remaining_lots": cur_lots,
                    }
                    self.tracked_positions[ticket] = tracked

                # Track peak price excursion
                if side == "BUY":
                    peak_price = max(float(tracked.get("peak_price") or open_price), current_price)
                else:
                    peak_price = min(float(tracked.get("peak_price") or open_price), current_price)
                tracked["peak_price"] = peak_price

                # Determine immutable initial risk and R-multiple targets
                initial_sl = float(tracked.get("initial_stop_loss") or tracked.get("stop_loss") or sl or (open_price - default_sl_dist if side == "BUY" else open_price + default_sl_dist))
                initial_risk_dist = float(tracked.get("initial_risk_distance") or max(abs(open_price - initial_sl), default_sl_dist))

                # Dynamic 0.5R Trailing Stop Buffer calculation
                if self.config.runner_trail_buffer_r is not None and self.config.runner_trail_buffer_r > 0:
                    trail_buffer = round(self.config.runner_trail_buffer_r * initial_risk_dist, 2)
                elif is_btc:
                    trail_buffer = self.config.runner_trail_buffer_usd if self.config.runner_trail_buffer_usd > 50.0 else 350.00
                elif is_eth:
                    trail_buffer = self.config.runner_trail_buffer_usd if self.config.runner_trail_buffer_usd != 4.50 else 10.00
                else:
                    trail_buffer = self.config.runner_trail_buffer_usd

                # Target 1 is strictly calibrated to scale_out_r (default 1.0R) for institutional partial exit & BE
                calc_t1 = round(open_price + (self.config.scale_out_r * initial_risk_dist), 2) if side == "BUY" else round(open_price - (self.config.scale_out_r * initial_risk_dist), 2)
                calc_t2 = round(open_price + (self.config.runner_target_2_multiplier * initial_risk_dist), 2) if side == "BUY" else round(open_price - (self.config.runner_target_2_multiplier * initial_risk_dist), 2)

                target_1 = calc_t1
                target_2 = calc_t2 if self.config.enable_runner_trailing else float(tracked.get("target_2") or tp or calc_t2)
                tracked["target_1"] = target_1
                tracked["target_2"] = target_2
                tp1_taken = bool(tracked.get("tp1_taken", 0))

                # -------------------------------------------------------------
                # STAGE 1 & 2: Target 1 Reached (+1.0R) -> 50% Scale-Out + Move SL to BE + offset
                # -------------------------------------------------------------
                is_above_be = (current_price >= open_price + be_offset) if side == "BUY" else (current_price <= open_price - be_offset)
                hit_tp1 = (current_price >= target_1 or (peak_price >= target_1 and is_above_be)) if side == "BUY" else (current_price <= target_1 or (peak_price <= target_1 and is_above_be))
                if hit_tp1 and not tp1_taken:
                    scale_out_success = True
                    scaled_out_lots = 0.0
                    scaled_out_usd = 0.0
                    if cur_lots >= 0.02 and self.config.enable_partial_scale_out:
                        close_lots = round(cur_lots * self.config.partial_scale_out_pct, 2)
                        idempotency_key = f"SCALE_OUT_{ticket}_{close_lots:.2f}"
                        intent = self._get_close_intent(idempotency_key)
                        if intent and intent.get("status") in ("COMPLETED", "RECONCILED"):
                            scale_out_success = True
                            scaled_out_lots = float(intent.get("closed_lots", close_lots))
                            tracked["partial_closed_lots"] = scaled_out_lots
                            tracked["tp1_taken"] = 1
                        else:
                            if not intent:
                                self._create_close_intent(
                                    ticket=ticket,
                                    intended_lots=close_lots,
                                    trigger_reason="TARGET_1_SCALE_OUT",
                                    idempotency_key=idempotency_key,
                                )
                            close_res = self.broker.close_position(ticket=ticket, lots=close_lots)
                            if close_res.get("status") == "SUCCESS":
                                scaled_out_lots = float(close_res.get("actual_lots") or close_res.get("lots") or close_lots)
                                if close_res.get("realized_pnl") is not None:
                                    scaled_out_usd = round(float(close_res["realized_pnl"]), 2)
                                else:
                                    scaled_out_usd = round(abs(current_price - open_price) * scaled_out_lots * contract_size, 2)
                                self._update_close_intent(idempotency_key, status="COMPLETED", closed_lots=scaled_out_lots)
                                tracked["partial_closed_lots"] = scaled_out_lots
                                tracked["partial_realized_usd"] = scaled_out_usd
                                msg = f"🏆 Scaled out {scaled_out_lots} lots (+$ {scaled_out_usd:.2f}) at Target 1 for Ticket #{ticket}"
                                actions_taken.append(msg)
                                logger.info(msg)
                            else:
                                # Reconcile against broker deal history & remaining position volume
                                recon_success, recon_lots = self._reconcile_partial_close(ticket, close_lots, initial_lots=float(tracked.get("lots", cur_lots)))
                                if recon_success:
                                    scaled_out_lots = recon_lots
                                    deal_pnl = None
                                    if hasattr(self.broker, "get_deal_history"):
                                        try:
                                            deals = self.broker.get_deal_history(ticket=ticket)
                                            if deals.get("status") == "SUCCESS" and deals.get("deals"):
                                                deal_pnl = sum(float(d.get("profit", 0.0)) for d in deals["deals"])
                                        except Exception:
                                            pass
                                    scaled_out_usd = round(deal_pnl, 2) if deal_pnl is not None else round(abs(current_price - open_price) * scaled_out_lots * contract_size, 2)
                                    self._update_close_intent(idempotency_key, status="RECONCILED", closed_lots=scaled_out_lots)
                                    tracked["partial_closed_lots"] = scaled_out_lots
                                    tracked["partial_realized_usd"] = scaled_out_usd
                                    scale_out_success = True
                                    msg = f"🏆 Reconciled partial close {scaled_out_lots} lots (+$ {scaled_out_usd:.2f}) at Target 1 for Ticket #{ticket}"
                                    actions_taken.append(msg)
                                    logger.info(msg)
                                else:
                                    scale_out_success = False
                                    logger.warning("Partial scale-out failed for #%s: %s. Pending reconciliation.", ticket, close_res.get("message"))

                    if scale_out_success:
                        tracked["tp1_taken"] = 1
                        init_vol = float(tracked.get("initial_lots") or tracked.get("lots") or cur_lots)
                        tracked["remaining_lots"] = max(0.0, round(init_vol - float(tracked.get("partial_closed_lots", 0.0)), 2))

                        # Shift remaining volume to Break-Even + offset (Risk-Free Runner) if enabled
                        res = {"status": "SKIPPED"}
                        if self.config.trail_to_breakeven_at_t1:
                            new_sl = round(open_price + be_offset, 2) if side == "BUY" else round(open_price - be_offset, 2)
                            new_tp = target_2 if (self.config.enable_runner_trailing and target_2) else p.get("tp")
                            res = self.broker.modify_position(ticket=ticket, sl=new_sl, tp=new_tp)
                            if res.get("status") == "SUCCESS":
                                tracked["stop_loss"] = new_sl
                                tracked["target_2"] = new_tp
                                msg = f"Trailed SL to Break-Even + ${be_offset:.2f} (${new_sl}) for Ticket #{ticket}"
                                actions_taken.append(msg)
                                logger.info(msg)
                            else:
                                logger.warning("Failed to modify SL to Break-Even for #%s: %s", ticket, res.get("message"))

                        # Dispatch Telegram alerts AFTER broker order actions have completed
                        if self.config.send_telegram_alerts:
                            if scaled_out_lots > 0:
                                self.notifier.send_trade_update(
                                    ticket=ticket,
                                    symbol=symbol,
                                    update_type="Target 1 Hit! 50% Scaled Out 💰",
                                    details=f"Price hit Target 1 (${target_1:.2f}). Scaled out {scaled_out_lots} lots. Banked +${scaled_out_usd:.2f} guaranteed cash.",
                                )
                            if res.get("status") == "SUCCESS":
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
                    target_2_val = target_2 or tracked.get("target_2") or p.get("tp")

                    if side == "BUY":
                        trailed_sl = round(peak_price - trail_buffer, 2)
                        if trailed_sl >= current_sl_val + be_offset and trailed_sl >= open_price + be_offset:
                            res = self.broker.modify_position(ticket=ticket, sl=trailed_sl, tp=target_2_val)
                            if res.get("status") == "SUCCESS":
                                tracked["stop_loss"] = trailed_sl
                                msg = f"Trailed Runner SL to ${trailed_sl} for Ticket #{ticket}"
                                actions_taken.append(msg)
                                logger.info(msg)
                                if self.config.send_telegram_alerts:
                                    self.notifier.send_trade_update(
                                        ticket=ticket,
                                        symbol=symbol,
                                        update_type="Runner Trailing SL Advanced 🏃",
                                        details=f"Peak reached ${peak_price:.2f}. Trailed Stop-Loss advanced to ${trailed_sl:.2f} (+$ {trailed_sl - open_price:.2f} profit locked).",
                                    )
                            else:
                                logger.warning("Runner trailing modification failed for #%s: %s", ticket, res.get("message"))
                    else:  # SELL
                        trailed_sl = round(peak_price + trail_buffer, 2)
                        if trailed_sl <= current_sl_val - be_offset and trailed_sl <= open_price - be_offset:
                            res = self.broker.modify_position(ticket=ticket, sl=trailed_sl, tp=target_2_val)
                            if res.get("status") == "SUCCESS":
                                tracked["stop_loss"] = trailed_sl
                                msg = f"Trailed Runner SL to ${trailed_sl} for Ticket #{ticket}"
                                actions_taken.append(msg)
                                logger.info(msg)
                                if self.config.send_telegram_alerts:
                                    self.notifier.send_trade_update(
                                        ticket=ticket,
                                        symbol=symbol,
                                        update_type="Runner Trailing SL Advanced 🏃",
                                        details=f"Peak dropped to ${peak_price:.2f}. Trailed Stop-Loss advanced to ${trailed_sl:.2f} (+$ {open_price - trailed_sl:.2f} profit locked).",
                                    )
                            else:
                                logger.warning("Runner trailing modification failed for #%s: %s", ticket, res.get("message"))

                # Persist updated tracking state to database
                with self._db_connection() as conn:
                    conn.execute(
                        """
                        UPDATE trade_executions 
                        SET stop_loss = ?, tp1_taken = ?, peak_price = ?, partial_closed_lots = ?, partial_realized_usd = ?, remaining_lots = ?
                        WHERE ticket = ?
                        """,
                        (tracked.get("stop_loss"), int(tracked.get("tp1_taken", 0)), float(tracked.get("peak_price", open_price)), float(tracked.get("partial_closed_lots", 0.0)), float(tracked.get("partial_realized_usd", 0.0)), float(tracked.get("remaining_lots", cur_lots)), ticket),
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
        if current_open_positions is None:
            return
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
                        if is_btc or is_eth:
                            close_price = entry_price
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

            with self._db_connection() as conn:
                conn.execute(
                    """
                    UPDATE trade_executions 
                    SET status = 'CLOSED', close_price = ?, pnl = ?, close_time = ?, remaining_lots = 0.0
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

            logger.info("📊 Trade #%s closed at $%.2f (Net PnL: $%.2f). Recorded in trade_executions DB.", ticket, close_price, pnl)

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
            with self._db_connection() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT * FROM scan_logs ORDER BY id DESC LIMIT ?", (limit,))
                scans = [dict(row) for row in cur.fetchall()]
                cur.execute("SELECT * FROM trade_executions ORDER BY id DESC LIMIT ?", (limit,))
                trades = [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error("Error reading history: %s", e)
            
        return {"scans": scans, "trades": trades}
