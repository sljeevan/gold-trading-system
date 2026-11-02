"""
Unit & Integration Tests verifying P2 High-Priority Remediation:
1. Standardised Backtest and Live Configuration Schema (UnifiedRunConfig).
2. Cryptographic configuration hashing and Git SHA provenance tracking.
3. Strict rejection of unknown / unapproved configuration overrides.
4. Broker-to-Database Reconciliation (unmatched positions, ghost trades, volume mismatches, missing SL).
5. Multi-component dashboard health matrix and fail-closed feed monitoring.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from gold_trader.core.unified_config import (
    CURRENT_SCHEMA_VERSION,
    CURRENT_STRATEGY_VERSION,
    UnifiedRunConfig,
    resolve_git_sha,
)
from gold_trader.core.reconciliation import ReconciliationEngine
from gold_trader.core.auto_trader import AutoTraderConfig, GoldAutoTrader
from gold_trader.brokers.socket_broker import BrokerPositionsUnavailableError
from tests.test_auto_trader import DummyMockBroker


class TestP2Remediation(unittest.TestCase):

    def test_unified_config_validation_and_fail_fast(self):
        """
        Verify that UnifiedRunConfig validates required bounds and rejects invalid inputs fail-fast.
        """
        # Valid config
        cfg = UnifiedRunConfig(symbol="XAUUSD", timeframe="1h", fixed_risk_usd=75.44)
        cfg.validate()
        self.assertEqual(cfg.schema_version, CURRENT_SCHEMA_VERSION)
        self.assertEqual(cfg.strategy_version, CURRENT_STRATEGY_VERSION)
        self.assertGreater(len(cfg.config_hash), 0)

        # Invalid timeframe
        with self.assertRaises(ValueError) as ctx:
            bad_tf = UnifiedRunConfig(timeframe="2h")
            bad_tf.validate()
        self.assertIn("Invalid timeframe", str(ctx.exception))

        # Invalid fixed risk
        with self.assertRaises(ValueError) as ctx:
            bad_risk = UnifiedRunConfig(fixed_risk_usd=-10.0)
            bad_risk.validate()
        self.assertIn("fixed_risk_usd must be > 0", str(ctx.exception))

        # Invalid trigger mode
        with self.assertRaises(ValueError) as ctx:
            bad_trig = UnifiedRunConfig(trigger_mode="INVALID_MODE")
            bad_trig.validate()
        self.assertIn("Invalid trigger_mode", str(ctx.exception))

    def test_unified_config_deterministic_hash_and_rejection_of_unknown_keys(self):
        """
        Verify that identical configurations produce identical hashes, and unknown keys are rejected.
        """
        cfg1 = UnifiedRunConfig(symbol="XAUUSD", timeframe="1h", donchian_period=10, fixed_risk_usd=75.44)
        cfg2 = UnifiedRunConfig(symbol="XAUUSD", timeframe="1h", donchian_period=10, fixed_risk_usd=75.44)
        cfg3 = UnifiedRunConfig(symbol="XAUUSD", timeframe="15m", donchian_period=10, fixed_risk_usd=75.44)

        self.assertEqual(cfg1.config_hash, cfg2.config_hash, "Identical configurations must yield identical hashes")
        self.assertNotEqual(cfg1.config_hash, cfg3.config_hash, "Different timeframes must produce different hashes")

        # Unknown parameter rejection
        raw_dict = cfg1.to_dict()
        raw_dict["unapproved_secret_lever"] = 999.0
        with self.assertRaises(ValueError) as ctx:
            UnifiedRunConfig.from_dict(raw_dict)
        self.assertIn("Unapproved configuration parameter(s) rejected", str(ctx.exception))

    def test_unified_config_json_roundtrip(self):
        """
        Verify that save_json and load_json reproduce the exact configuration with complete parity.
        """
        temp_cfg = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        temp_cfg.close()
        original = UnifiedRunConfig(
            symbol="BTCUSD",
            timeframe="15m",
            donchian_period=20,
            rr_multiplier=1.5,
            spread_usd=5.0,
            commission_per_lot_usd=0.0,
        )
        original.save_json(temp_cfg.name)

        loaded = UnifiedRunConfig.load_json(temp_cfg.name)
        self.assertEqual(original.config_hash, loaded.config_hash)
        self.assertEqual(original.symbol, loaded.symbol)
        self.assertEqual(original.donchian_period, loaded.donchian_period)
        self.assertEqual(original.spread_usd, loaded.spread_usd)

    def test_auto_trader_stores_provenance_metadata(self):
        """
        Verify that GoldAutoTrader writes config_hash, git_sha, and schema_version into trade_executions and scan_logs.
        """
        broker = DummyMockBroker()
        config = AutoTraderConfig(
            symbol="XAUUSD",
            auto_execute=True,
            flat_risk=True,
            fixed_risk_usd=75.44,
            send_telegram_alerts=False,
        )
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()
        trader = GoldAutoTrader(broker=broker, config_override=config, db_path=temp_db.name)

        self.assertIsNotNone(trader.config_hash)
        self.assertIsNotNone(trader.git_sha)

        # Place a mock scan log
        trader.log_scan(
            symbol="XAUUSD",
            price=2700.0,
            bias="BULLISH",
            setup_name="Test",
            action="HOLD",
            quality_score=85,
            rr_ratio=1.25,
            decision="WAIT",
        )

        with trader._db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT config_hash, git_sha FROM scan_logs ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], trader.config_hash)
            self.assertEqual(row[1], trader.git_sha)

    def test_reconciliation_missing_protective_sl(self):
        """
        Verify that ReconciliationEngine flags any broker open position lacking active Stop-Loss protection as CRITICAL.
        """
        mock_broker = MagicMock()
        mock_broker.get_open_positions.return_value = [{
            "ticket": 55511,
            "symbol": "XAUUSD",
            "side": "BUY",
            "lots": 0.05,
            "open_price": 2700.0,
            "sl": 0.0,  # Missing SL!
            "tp": 2720.0,
        }]

        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()

        engine = ReconciliationEngine(broker=mock_broker, db_paths={"XAUUSD": temp_db.name})
        report = engine.reconcile_positions(symbol="XAUUSD")

        self.assertEqual(report["status"], "DISCREPANCY_DETECTED")
        types = [d["type"] for d in report["discrepancies"]]
        self.assertIn("MISSING_PROTECTIVE_SL", types)

    def test_reconciliation_unmatched_and_ghost_trades(self):
        """
        Verify that ReconciliationEngine detects:
        1. UNMATCHED_BROKER_POSITION (active on broker, missing locally)
        2. LOCAL_GHOST_TRADE (marked OPEN locally, missing on broker)
        """
        mock_broker = MagicMock()
        mock_broker.get_open_positions.return_value = [{
            "ticket": 99911,
            "symbol": "XAUUSD",
            "side": "BUY",
            "lots": 0.04,
            "open_price": 2700.0,
            "sl": 2690.0,
            "tp": 2720.0,
        }]

        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()

        # Insert a ghost trade into local database that broker does not possess
        import sqlite3
        conn = sqlite3.connect(temp_db.name)
        conn.execute("""
            CREATE TABLE trade_executions (
                id INTEGER PRIMARY KEY, ticket TEXT UNIQUE, symbol TEXT, action TEXT, lots REAL,
                entry_price REAL, stop_loss REAL, target_1 REAL, target_2 REAL, rr_ratio REAL,
                status TEXT, remaining_lots REAL
            )
        """)
        conn.execute("""
            INSERT INTO trade_executions (ticket, symbol, action, lots, entry_price, stop_loss, target_1, target_2, rr_ratio, status, remaining_lots)
            VALUES ('88822', 'XAUUSD', 'BUY', 0.03, 2695.0, 2685.0, 2710.0, 2730.0, 1.5, 'OPEN', 0.03)
        """)
        conn.commit()
        conn.close()

        engine = ReconciliationEngine(broker=mock_broker, db_paths={"XAUUSD": temp_db.name})
        report = engine.reconcile_positions(symbol="XAUUSD")

        self.assertEqual(report["status"], "DISCREPANCY_DETECTED")
        types = [d["type"] for d in report["discrepancies"]]
        self.assertIn("UNMATCHED_BROKER_POSITION", types)
        self.assertIn("LOCAL_GHOST_TRADE", types)

    def test_reconciliation_volume_mismatch(self):
        """
        Verify that ReconciliationEngine detects discrepancies between local remaining volume and broker lots.
        """
        mock_broker = MagicMock()
        mock_broker.get_open_positions.return_value = [{
            "ticket": 77733,
            "symbol": "XAUUSD",
            "side": "BUY",
            "lots": 0.02,  # Broker scaled out to 0.02
            "open_price": 2700.0,
            "sl": 2700.50,
            "tp": 2730.0,
        }]

        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()
        import sqlite3
        conn = sqlite3.connect(temp_db.name)
        conn.execute("""
            CREATE TABLE trade_executions (
                id INTEGER PRIMARY KEY, ticket TEXT UNIQUE, symbol TEXT, action TEXT, lots REAL,
                entry_price REAL, stop_loss REAL, target_1 REAL, target_2 REAL, rr_ratio REAL,
                status TEXT, remaining_lots REAL
            )
        """)
        # Local still expects 0.04 lots
        conn.execute("""
            INSERT INTO trade_executions (ticket, symbol, action, lots, entry_price, stop_loss, target_1, target_2, rr_ratio, status, remaining_lots)
            VALUES ('77733', 'XAUUSD', 'BUY', 0.04, 2700.0, 2700.50, 2712.5, 2730.0, 1.25, 'OPEN', 0.04)
        """)
        conn.commit()
        conn.close()

        engine = ReconciliationEngine(broker=mock_broker, db_paths={"XAUUSD": temp_db.name})
        report = engine.reconcile_positions(symbol="XAUUSD")

        self.assertEqual(report["status"], "DISCREPANCY_DETECTED")
        types = [d["type"] for d in report["discrepancies"]]
        self.assertIn("VOLUME_MISMATCH", types)
        vol_disc = next(d for d in report["discrepancies"] if d["type"] == "VOLUME_MISMATCH")
        self.assertEqual(vol_disc["broker_lots"], 0.02)
        self.assertEqual(vol_disc["local_lots"], 0.04)

    def test_reconciliation_broker_unavailable_fails_closed(self):
        """
        Verify that when broker positions query raises an error, ReconciliationEngine returns
        BROKER_UNAVAILABLE rather than falsely reporting 0 positions.
        """
        mock_broker = MagicMock()
        mock_broker.get_open_positions.side_effect = BrokerPositionsUnavailableError("MT5 query timed out")

        engine = ReconciliationEngine(broker=mock_broker)
        report = engine.reconcile_positions()

        self.assertEqual(report["status"], "BROKER_UNAVAILABLE")
        self.assertIsNone(report["broker_positions"])
        self.assertIn("MT5 query timed out", report["error"])

    def test_dashboard_health_matrix_structure(self):
        """
        Verify that dashboard_api.health() returns a complete 9-component status matrix.
        """
        try:
            from gold_trader.api.dashboard_api import health
        except ModuleNotFoundError:
            self.skipTest("fastapi not installed in current python interpreter; run with .venv")

        with patch("gold_trader.api.dashboard_api.get_broker") as mock_get_broker:
            mock_broker = MagicMock()
            mock_broker.host = "127.0.0.1"
            mock_broker.port = 8888
            mock_broker.get_health.return_value = {
                "status": "ONLINE",
                "mt5_connected": True,
                "mt5_status": "CONNECTED",
                "mt5_terminal_info": {"login": 12345},
            }
            mock_broker.get_account_info.return_value = {"balance": 1000.0, "equity": 1000.0}
            mock_broker.get_open_positions.return_value = []
            mock_broker.get_pending_orders.return_value = []
            mock_get_broker.return_value = mock_broker

            res = health()
            self.assertIn("status", res)
            self.assertIn("components", res)
            comps = res["components"]
            self.assertIn("socket_bridge", comps)
            self.assertIn("mt5_terminal", comps)
            self.assertIn("account_margin", comps)
            self.assertIn("positions_feed", comps)
            self.assertIn("pending_orders_feed", comps)
            self.assertIn("market_data", comps)
            self.assertIn("ml_predictor", comps)
            self.assertIn("database", comps)
            self.assertEqual(comps["socket_bridge"]["status"], "ONLINE")
            self.assertEqual(comps["mt5_terminal"]["connected"], True)


if __name__ == "__main__":
    unittest.main()
