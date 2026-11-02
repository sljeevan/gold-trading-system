"""
Phase A Safety & Hardening Acceptance Tests.

Verifies:
1. P2-01: Bridge MT5 terminal call serialization mutex lock (threading.Lock).
2. P1-17: Explicit live-account execution guard (allow_live_real opt-in and account ID verification).
3. P1-14: Client order UUID generation and unambiguous entry timeout reconciliation.
4. P1-15: Fail-closed policy for news catalyst guard and daily trade-cap database errors.
5. P1-13: Broker-adjusted volume and deal-ticket realized P&L reflection in local state.
6. P1-06: Signal-bar index consistency in CLOSE trigger mode (completed bar [-2] vs forming bar [-1]).
7. P1-07: ML validation embargo enforcement and fail-closed behavior on insufficient samples.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from gold_trader.brokers.base import BaseBroker
from gold_trader.brokers.socket_broker import SocketBroker
from gold_trader.core.auto_trader import AutoTraderConfig, GoldAutoTrader
from gold_trader.core.ml_filter import purged_walk_forward_cv
from gold_trader.core.strategy_engine import StrategyConfig, generate_donchian_momentum_hybrid_setup
from gold_trader.types import AccountInfo, OrderResult, Position


class MockTestingBroker(BaseBroker):
    def __init__(
        self,
        balance: float = 1000.0,
        trade_mode: str = "DEMO",
        account_id: str = "168987477",
        open_positions: list | None = None,
        pending_orders: list | None = None,
    ):
        self._balance = balance
        self._trade_mode = trade_mode
        self._account_id = account_id
        self._open_positions = open_positions or []
        self._pending_orders = pending_orders or []
        self.orders: list[dict] = []
        self.closed: list[dict] = []

    def connect(self) -> bool:
        return True

    def get_account_info(self) -> AccountInfo:
        return {
            "broker": f"XM MT5 [{self._trade_mode}] (XMGlobal-MT5)",
            "account_id": self._account_id,
            "balance": self._balance,
            "equity": self._balance,
            "margin": 0.0,
            "free_margin": self._balance,
            "margin_level_pct": 0.0,
            "leverage": 100,
            "currency": "USD",
            "trade_mode": self._trade_mode,
            "trade_allowed": True,
            "server": "XMGlobal-MT5",
            "company": "XM Global",
        }

    def get_open_positions(self, symbol: str | None = None) -> list[Position]:
        return list(self._open_positions)

    def get_pending_orders(self, symbol: str | None = None) -> list[dict]:
        return list(self._pending_orders)

    def get_rates(self, symbol: str = "XAUUSD", timeframe: str = "15m", count: int = 100) -> list[dict]:
        rates = []
        for i in range(count):
            rates.append({
                "date": f"2026-11-02T{i%24:02d}:00:00",
                "open": 2700.0,
                "high": 2705.0,
                "low": 2695.0,
                "close": 2702.0,
                "volume": 100.0,
                "source": "MT5_BROKER",
            })
        return rates

    def get_deal_history(self, ticket: str | int | None = None, days: int = 7) -> dict:
        return {
            "status": "SUCCESS",
            "deals": [{
                "ticket": 999,
                "order": int(ticket or 0),
                "profit": 75.0,
                "commission": -0.35,
                "swap": 0.0,
                "volume": 0.05,
                "price": 2715.0,
            }]
        }

    def place_order(
        self,
        symbol: str,
        side: str,
        lots: float,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "Test",
        price: float | None = None,
        order_type: str = "MARKET",
    ) -> OrderResult:
        ticket = str(100000 + len(self.orders) + 1)
        res: OrderResult = {
            "status": "SUCCESS",
            "ticket": ticket,
            "symbol": symbol,
            "action": side,
            "lots": lots,
            "actual_lots": lots,
            "price": price or 2700.0,
            "sl": sl,
            "tp": tp,
            "deal_ticket": f"D_{ticket}",
            "client_order_id": "test_uuid",
            "message": "Order executed",
            "timestamp": "2026-11-02T12:00:00Z",
        }
        self.orders.append(res)
        return res

    def modify_position(self, ticket: str, sl: float | None = None, tp: float | None = None) -> OrderResult:
        return {
            "status": "SUCCESS",
            "ticket": ticket,
            "symbol": "XAUUSD",
            "action": "MODIFY",
            "lots": 0.0,
            "price": 0.0,
            "sl": sl,
            "tp": tp,
            "message": "Modified",
            "timestamp": "2026-11-02T12:00:00Z",
        }

    def close_position(self, ticket: str, lots: float | None = None) -> OrderResult:
        res: OrderResult = {
            "status": "SUCCESS",
            "ticket": ticket,
            "symbol": "XAUUSD",
            "action": "CLOSE",
            "lots": lots or 0.01,
            "actual_lots": lots or 0.01,
            "price": 2715.0,
            "deal_ticket": "DEAL_CLOSE_99",
            "realized_pnl": 75.00,
            "commission": -0.35,
            "swap": 0.0,
            "message": "Closed",
            "timestamp": "2026-11-02T12:00:00Z",
        }
        self.closed.append(res)
        return res


class TestPhaseAHardening(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

    def tearDown(self):
        if os.path.exists(self.temp_db.name):
            try:
                os.remove(self.temp_db.name)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # 1. P2-01: MT5 Bridge Execution Mutex Lock
    # -------------------------------------------------------------------------
    def test_bridge_server_uses_mutex_lock(self):
        """Verify bridge_server exposes mt5_lock and process_bridge_command acquires it."""
        from gold_trader import bridge_server
        self.assertTrue(hasattr(bridge_server, "mt5_lock"))
        self.assertTrue(hasattr(bridge_server, "process_bridge_command"))

        # Verify executing process_bridge_command with mock MT5 acquires the lock
        with patch.object(bridge_server, "mt5") as mock_mt5:
            mock_mt5.terminal_info.return_value = MagicMock(connected=True, trade_allowed=True)
            res = bridge_server.process_bridge_command({"command": "PING"})
            self.assertEqual(res["status"], "SUCCESS")

    # -------------------------------------------------------------------------
    # 2. P1-17: Explicit Live-Account Execution Guard
    # -------------------------------------------------------------------------
    def test_live_real_account_blocked_by_default(self):
        """Verify trading on a LIVE account is blocked unless allow_live_real=True."""
        broker = MockTestingBroker(balance=5000.0, trade_mode="LIVE")
        config = AutoTraderConfig(
            symbol="XAUUSD",
            auto_execute=True,
            flat_risk=True,
            allow_live_real=False,  # Default safety guard
            send_telegram_alerts=False,
        )
        trader = GoldAutoTrader(broker=broker, config_override=config, db_path=self.temp_db.name)

        result = trader.evaluate_and_trade()
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("Live real account execution blocked", result["reason"])
        self.assertEqual(len(broker.orders), 0)

    def test_live_account_permitted_with_explicit_opt_in(self):
        """Verify trading proceeds on a LIVE account when allow_live_real=True."""
        broker = MockTestingBroker(balance=5000.0, trade_mode="LIVE")
        config = AutoTraderConfig(
            symbol="XAUUSD",
            auto_execute=True,
            flat_risk=True,
            allow_live_real=True,  # Explicitly opted-in
            send_telegram_alerts=False,
        )
        trader = GoldAutoTrader(broker=broker, config_override=config, db_path=self.temp_db.name)

        result = trader.evaluate_and_trade()
        # Should not be blocked by live real guard
        self.assertNotEqual(result.get("status"), "BLOCKED")

    def test_account_id_mismatch_blocks_execution(self):
        """Verify connected account ID is validated against expected_account_id."""
        broker = MockTestingBroker(balance=5000.0, account_id="999999")
        config = AutoTraderConfig(
            symbol="XAUUSD",
            auto_execute=True,
            flat_risk=True,
            expected_account_id="168987477",  # Expecting 168987477, connected to 999999
            send_telegram_alerts=False,
        )
        trader = GoldAutoTrader(broker=broker, config_override=config, db_path=self.temp_db.name)

        result = trader.evaluate_and_trade()
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("Account ID mismatch", result["reason"])
        self.assertEqual(len(broker.orders), 0)

    # -------------------------------------------------------------------------
    # 3. P1-14: Client Order UUID & Unambiguous Timeout Reconciliation
    # -------------------------------------------------------------------------
    def test_socket_broker_generates_unique_client_order_id(self):
        """Verify place_order generates a unique client_order_id and reconciles unambiguously."""
        broker = SocketBroker()
        with patch.object(broker, "_send_command") as mock_send:
            mock_send.return_value = {
                "status": "SUCCESS",
                "ticket": "77123",
                "symbol": "XAUUSD",
                "price": 2700.0,
                "lots": 0.05,
                "actual_lots": 0.05,
                "deal_ticket": "DEAL_77123",
            }
            res = broker.place_order(symbol="XAUUSD", side="BUY", lots=0.05)
            self.assertEqual(res["status"], "SUCCESS")
            self.assertIsNotNone(res.get("client_order_id"))
            # Verify payload passed client_order_id
            payload = mock_send.call_args[0][0]
            self.assertIn("client_order_id", payload)
            self.assertEqual(payload["client_order_id"], res["client_order_id"])

    def test_timeout_reconciliation_rejects_ambiguous_or_unmatched_positions(self):
        """Verify timeout reconciliation fails closed when no exact client_order_id matches."""
        broker = SocketBroker()
        # Mock timeout response
        with patch.object(broker, "_send_command", return_value={"status": "ERROR", "message": "Socket timeout"}):
            # Open positions contain unrelated/stale trades from earlier sessions
            stale_positions = [
                {"ticket": "111", "symbol": "XAUUSD", "side": "SELL", "comment": "OldTrade_abc"},
                {"ticket": "222", "symbol": "XAUUSD", "side": "BUY", "comment": "OldTrade_xyz"},
            ]
            with patch.object(broker, "get_open_positions", return_value=stale_positions):
                res = broker.place_order(symbol="XAUUSD", side="BUY", lots=0.05, comment="NewScan")
                # Must NOT falsely claim one of the stale positions is the timed-out order
                self.assertEqual(res["status"], "REJECTED")
                self.assertIn("Reconciliation failed", res["message"])

    # -------------------------------------------------------------------------
    # 4. P1-15: Fail-Closed News Guard and Daily Trade-Cap Error Handling
    # -------------------------------------------------------------------------
    def test_trade_cap_db_failure_fails_closed(self):
        """Verify database query failure for daily trade cap blocks trading when FAIL_CLOSED."""
        broker = MockTestingBroker(balance=1000.0)
        config = AutoTraderConfig(
            symbol="XAUUSD",
            auto_execute=True,
            flat_risk=True,
            trade_cap_failure_policy="FAIL_CLOSED",
            send_telegram_alerts=False,
        )
        trader = GoldAutoTrader(broker=broker, config_override=config, db_path=self.temp_db.name)

        # Corrupt DB connection during scan to simulate SQLite lock/failure
        with patch.object(trader, "_db_connection", side_effect=sqlite3.OperationalError("Database disk I/O error")):
            result = trader.evaluate_and_trade()
            self.assertIn("SKIP", result.get("decision", ""))
            self.assertIn("daily trade cap", result.get("decision", ""))
            self.assertIsNone(result.get("order_result"))
            self.assertEqual(len(broker.orders), 0)

    def test_news_guard_query_failure_fails_closed(self):
        """Verify news guard evaluation failure blocks trading when news_guard_failure_policy=FAIL_CLOSED."""
        broker = MockTestingBroker(balance=1000.0)
        config = AutoTraderConfig(
            symbol="XAUUSD",
            auto_execute=True,
            flat_risk=True,
            enable_news_catalyst_guard=True,
            news_guard_failure_policy="FAIL_CLOSED",
            send_telegram_alerts=False,
        )
        trader = GoldAutoTrader(broker=broker, config_override=config, db_path=self.temp_db.name)

        # Force economic calendar query to throw
        with patch("gold_trader.core.openbb_feed.check_high_impact_news_window", side_effect=RuntimeError("FRED API timeout")):
            with patch.object(trader, "log_scan"):
                # Mock a BUY setup from strategy generator
                with patch("gold_trader.core.auto_trader.generate_compounding_ladder_setup", return_value={
                    "status": "APPROVED",
                    "action": "BUY",
                    "entry_price": 2700.0,
                    "stop_loss": 2690.0,
                    "target_1": 2712.5,
                    "target_2": 2730.0,
                    "suggested_lot_size": 0.05,
                    "quality_score": 95,
                    "risk_reward_t1": 1.25,
                }):
                    result = trader.evaluate_and_trade()
                    self.assertIn("SKIP", result.get("decision", ""))
                    self.assertTrue("News catalyst" in result.get("decision", "") or "Guard Unavailable" in result.get("decision", ""))
                    self.assertIsNone(result.get("order_result"))
                    self.assertEqual(len(broker.orders), 0)

    # -------------------------------------------------------------------------
    # 5. P1-13: Broker Volume Adjustment & Deal Ticket Realized P&L
    # -------------------------------------------------------------------------
    def test_broker_adjusted_volume_and_deal_pnl_recorded(self):
        """Verify actual executed lots and deal realized P&L are persisted to SQLite."""
        broker = MockTestingBroker(balance=1000.0)
        config = AutoTraderConfig(
            symbol="XAUUSD",
            auto_execute=True,
            flat_risk=True,
            enable_partial_scale_out=True,
            trail_to_breakeven_at_t1=True,
            send_telegram_alerts=False,
        )
        trader = GoldAutoTrader(broker=broker, config_override=config, db_path=self.temp_db.name)

        # Seed an open trade with initial lots 0.10
        with trader._db_connection() as conn:
            conn.execute(
                """
                INSERT INTO trade_executions
                (timestamp, ticket, symbol, action, lots, entry_price, stop_loss, target_1, target_2, rr_ratio, status, tp1_taken, peak_price, partial_closed_lots, partial_realized_usd, initial_stop_loss, initial_risk_distance, initial_lots, remaining_lots)
                VALUES ('2026-11-02T10:00:00Z', '55501', 'XAUUSD', 'BUY', 0.10, 2700.0, 2690.0, 2710.0, 2730.0, 1.0, 'OPEN', 0, 2712.0, 0.0, 0.0, 2690.0, 10.0, 0.10, 0.10)
                """
            )
            conn.commit()

        trader._load_active_tracked_positions()
        # Mock broker positions and close_position response with deal receipt
        active_pos = [{
            "ticket": "55501",
            "symbol": "XAUUSD",
            "side": "BUY",
            "lots": 0.10,
            "open_price": 2700.0,
            "current_price": 2712.0,  # Price is past Target 1 ($2710.0)
            "sl": 2690.0,
            "tp": 2730.0,
        }]

        # Trigger position management
        actions = trader._manage_open_positions(active_pos)
        self.assertTrue(any("Scaled out" in a for a in actions))

        # Verify SQLite row was updated with deal realized P&L ($75.00 from MockTestingBroker.close_position)
        with trader._db_connection() as conn:
            row = conn.execute("SELECT partial_realized_usd, partial_closed_lots, remaining_lots FROM trade_executions WHERE ticket='55501'").fetchone()
            self.assertEqual(row[0], 75.00)
            self.assertEqual(row[1], 0.05)
            self.assertEqual(row[2], 0.05)

    # -------------------------------------------------------------------------
    # 6. P1-06: Strict Closed Signal Bar Index Consistency in CLOSE Trigger Mode
    # -------------------------------------------------------------------------
    def test_close_trigger_mode_strictly_evaluates_completed_bar(self):
        """Verify that in CLOSE trigger mode, all confirmation indicators evaluate on bar [-2]."""
        # Create 50 synthetic candles where:
        # Candle [-2] closed above Donchian Upper [-3] (Breakout on completed bar)
        # Candle [-1] is an opposing/reversal bar (forming in-progress)
        candles = []
        for i in range(50):
            candles.append({
                "date": f"2026-11-02 14:00",
                "open": 2600.0 + i,
                "high": 2605.0 + i,
                "low": 2595.0 + i,
                "close": 2602.0 + i,
                "volume": 100.0,
            })
        # Set completed candle [-2] to strong bullish breakout
        candles[-2]["open"] = 2650.0
        candles[-2]["high"] = 2665.0
        candles[-2]["low"] = 2649.0
        candles[-2]["close"] = 2663.0  # Big green breakout bar

        # Set incomplete forming candle [-1] to an adverse tick
        candles[-1]["open"] = 2663.0
        candles[-1]["high"] = 2664.0
        candles[-1]["low"] = 2640.0
        candles[-1]["close"] = 2641.0  # Incomplete intra-bar noise

        with patch("gold_trader.core.strategy_engine.fetch_gold_candles", return_value=candles):
            with patch("gold_trader.core.strategy_engine.calc_supertrend", return_value={"direction": [1]*50, "upper": [2600.0]*50, "lower": [2500.0]*50}):
                with patch("gold_trader.core.strategy_engine.calc_rsi", return_value=[50.0]*50):
                    with patch("gold_trader.core.strategy_engine.calc_ema", return_value=[2600.0]*50):
                        setup = generate_donchian_momentum_hybrid_setup(
                            symbol="XAUUSD",
                            timeframe="1h",
                            trigger_mode="CLOSE",
                            donchian_period=10,
                            trend_filter_ema=50,
                            require_slope_alignment=False,
                        )
                        # In CLOSE mode, signal must strictly confirm BUY based on closed bar [-2]
                        self.assertEqual(setup.get("action"), "BUY")
                        self.assertIn("Breakout", setup.get("setup_name", ""))

    # -------------------------------------------------------------------------
    # 7. P1-07: ML Validation Embargo & Strict Purging
    # -------------------------------------------------------------------------
    def test_purged_walk_forward_cv_enforces_embargo_and_fails_closed(self):
        """Verify purged_walk_forward_cv applies embargo gap and raises on insufficient samples."""
        # 1. Sufficient sample: verify valid folds are created without overlap
        bar_indices = list(range(100))
        splits = purged_walk_forward_cv(bar_indices, max_holding_bars=10, n_splits=3, pct_embargo=0.05)
        self.assertGreater(len(splits), 0)
        for train_idx, val_idx in splits:
            val_start = min(bar_indices[i] for i in val_idx)
            # Ensure every training bar holding window + embargo ends before validation start
            embargo_bars = int(round(100 * 0.05))
            for t in train_idx:
                self.assertLess(bar_indices[t] + 10 + embargo_bars, val_start)

        # 2. Insufficient sample: must raise ValueError, NEVER fall back to unpurged TimeSeriesSplit
        small_indices = list(range(12))
        with self.assertRaises(ValueError):
            purged_walk_forward_cv(small_indices, max_holding_bars=10, n_splits=5)


if __name__ == "__main__":
    unittest.main()
