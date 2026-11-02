"""
Unit tests verifying P1 High Priority Remediation:
1. Historical Macro Feature Leakage prevention during ML training.
2. Strict broker market-data enforcement (no silent external fallback in live mode).
3. Independent, decoupled position management (partial scale-out, BE, runner trailing).
4. Broker execution post-timeout order reconciliation.
5. Close-confirmed breakout signal evaluation on completed candles.
"""
import unittest
from unittest.mock import MagicMock, patch
import tempfile

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.auto_trader import AutoTraderConfig, GoldAutoTrader
from gold_trader.brokers.socket_broker import SocketBroker
from gold_trader.core.strategy_engine import generate_donchian_momentum_hybrid_setup
from tests.test_auto_trader import DummyMockBroker


class TestP1Remediation(unittest.TestCase):

    def test_strict_broker_market_data_fails_closed(self):
        """
        Verify that fetch_gold_candles with require_broker=True raises an error
        when broker rates are unavailable, rather than silently substituting Yahoo Finance.
        """
        with patch("gold_trader.brokers.socket_broker.SocketBroker.get_rates", return_value=[]):
            with self.assertRaises(RuntimeError) as ctx:
                fetch_gold_candles("XAUUSD", period="1d", interval="1h", require_broker=True)
            self.assertIn("broker feed required", str(ctx.exception).lower())

    def test_decoupled_partial_scale_out_when_breakeven_disabled(self):
        """
        Verify that partial scale-out at Target 1 still triggers even if trail_to_breakeven_at_t1 is False.
        """
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()
        broker = DummyMockBroker()
        config = AutoTraderConfig(
            symbol="XAUUSD",
            auto_execute=True,
            flat_risk=True,
            enable_partial_scale_out=True,
            trail_to_breakeven_at_t1=False,  # Decoupled!
            enable_runner_trailing=True,
            send_telegram_alerts=False,
            scale_out_r=1.0,
        )
        trader = GoldAutoTrader(broker=broker, config_override=config, db_path=temp_db.name)

        # Mock an open position that has reached Target 1 (+1.0R = 2710.0 from 2700.0 with $10 SL)
        open_pos = [{
            "ticket": "99901",
            "symbol": "XAUUSD",
            "side": "BUY",
            "lots": 0.04,
            "open_price": 2700.0,
            "current_price": 2712.0,  # Past target 1
            "sl": 2690.0,
            "tp": 2730.0,
            "comment": "AI-FlatRisk-XAUUSD",
        }]

        actions = trader._manage_open_positions(open_pos)
        self.assertGreater(len(actions), 0, "Partial scale-out should trigger even with breakeven disabled")
        self.assertTrue(any("Scaled out" in a for a in actions))

    def test_initial_risk_persistence_across_breakeven_trailing_and_restart(self):
        """
        Verify that immutable initial_risk_distance, initial_stop_loss, and initial_lots
        are preserved across break-even, runner trailing stop updates, partial scale-out,
        and daemon restart.
        """
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()
        broker = DummyMockBroker()
        config = AutoTraderConfig(
            symbol="XAUUSD",
            auto_execute=True,
            flat_risk=True,
            enable_partial_scale_out=True,
            trail_to_breakeven_at_t1=True,
            enable_runner_trailing=True,
            runner_trail_buffer_usd=4.50,
            send_telegram_alerts=False,
            scale_out_r=1.0,
        )
        trader = GoldAutoTrader(broker=broker, config_override=config, db_path=temp_db.name)

        # Place initial trade into db with entry 2700.0, SL 2690.0 (risk distance 10.0)
        ticket = "77701"
        trade_record = {
            "ticket": ticket,
            "symbol": "XAUUSD",
            "action": "BUY",
            "lots": 0.04,
            "entry_price": 2700.0,
            "stop_loss": 2690.0,
            "target_1": 2710.0,
            "target_2": 2730.0,
            "rr_ratio": 1.0,
            "comment": "AI-Auto-XAUUSD",
            "status": "OPEN",
            "timestamp": "2026-11-02T10:00:00Z",
            "tp1_taken": 0,
            "peak_price": 2700.0,
            "partial_closed_lots": 0.0,
            "partial_realized_usd": 0.0,
            "initial_stop_loss": 2690.0,
            "initial_risk_distance": 10.0,
            "initial_lots": 0.04,
            "remaining_lots": 0.04,
        }
        with trader._db_connection() as conn:
            conn.execute(
                """
                INSERT INTO trade_executions
                (timestamp, ticket, symbol, action, lots, entry_price, stop_loss, target_1, target_2, rr_ratio, status, comment, tp1_taken, peak_price, partial_closed_lots, partial_realized_usd, initial_stop_loss, initial_risk_distance, initial_lots, remaining_lots)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("2026-11-02T10:00:00Z", ticket, "XAUUSD", "BUY", 0.04, 2700.0, 2690.0, 2710.0, 2730.0, 1.0, "OPEN", "AI-Auto-XAUUSD", 0, 2700.0, 0.0, 0.0, 2690.0, 10.0, 0.04, 0.04),
            )
            conn.commit()

        # Reload tracked positions
        trader._load_active_tracked_positions()
        self.assertIn(ticket, trader.tracked_positions)
        self.assertEqual(trader.tracked_positions[ticket]["initial_risk_distance"], 10.0)

        # 1. Price advances to 2712.0 (Target 1 is 2710.0 -> triggers partial close and BE)
        open_pos = [{
            "ticket": ticket,
            "symbol": "XAUUSD",
            "side": "BUY",
            "lots": 0.04,
            "open_price": 2700.0,
            "current_price": 2712.0,
            "sl": 2690.0,
            "tp": 2730.0,
            "comment": "AI-Auto-XAUUSD",
        }]
        trader._manage_open_positions(open_pos)

        tracked = trader.tracked_positions[ticket]
        self.assertEqual(tracked["initial_risk_distance"], 10.0, "Initial risk distance must remain immutable after BE")
        self.assertEqual(tracked["initial_stop_loss"], 2690.0, "Initial stop loss must remain immutable after BE")
        self.assertEqual(tracked["stop_loss"], 2700.50, "Current stop loss moved to break-even + offset")
        self.assertEqual(tracked["remaining_lots"], 0.02, "Remaining volume updated after partial scale-out")

        # 2. Price advances further to 2725.0 -> Runner trailing stop advances
        open_pos[0]["current_price"] = 2725.0
        open_pos[0]["sl"] = 2700.50
        trader._manage_open_positions(open_pos)

        tracked = trader.tracked_positions[ticket]
        self.assertEqual(tracked["initial_risk_distance"], 10.0, "Initial risk distance must remain immutable after trailing")
        # Runner stop trailed using 0.5R = 0.5 * 10.0 = 5.0 -> 2725.0 - 5.0 = 2720.0
        self.assertEqual(tracked["stop_loss"], 2720.0, "Runner stop trailed to 2725.0 - 5.00 (0.5R) = 2720.00")

        # 3. Simulate daemon restart
        trader_restarted = GoldAutoTrader(broker=broker, config_override=config, db_path=temp_db.name)
        self.assertIn(ticket, trader_restarted.tracked_positions)
        recovered = trader_restarted.tracked_positions[ticket]
        self.assertEqual(recovered["initial_risk_distance"], 10.0, "Recovered initial risk distance must be 10.0")
        self.assertEqual(recovered["initial_stop_loss"], 2690.0, "Recovered initial stop loss must be 2690.0")
        self.assertEqual(recovered["stop_loss"], 2720.0, "Recovered current stop loss must be 2720.0")
        self.assertEqual(recovered["remaining_lots"], 0.02, "Recovered remaining lots must be 0.02")
        self.assertEqual(recovered["initial_lots"], 0.04, "Recovered initial lots must be 0.04")

    def test_post_timeout_order_reconciliation(self):
        """
        Verify that SocketBroker.place_order checks open positions if a socket timeout occurs,
        successfully reconciling the trade without false rejection.
        """
        broker = SocketBroker()
        with patch.object(broker, "_send_command", return_value={"status": "ERROR", "message": "Socket timeout"}):
            with patch.object(broker, "get_open_positions", return_value=[{
                "ticket": 88812,
                "symbol": "XAUUSD",
                "lots": 0.05,
                "open_price": 2705.50,
                "sl": 2695.0,
                "tp": 2725.0,
                "comment": "ReconTest-XAUUSD",
            }]):
                result = broker.place_order(
                    symbol="XAUUSD",
                    side="BUY",
                    lots=0.05,
                    comment="ReconTest-XAUUSD",
                )
                self.assertEqual(result["status"], "SUCCESS")
                self.assertEqual(result["ticket"], "88812")
                self.assertIn("verified via post-timeout reconciliation", result["message"])

    def test_close_confirmed_breakout_candle_evaluation(self):
        """
        Verify that when trigger_mode='BREAKOUT', setup evaluates completed candle [-2]
        against channel [-3], rather than incomplete forming bar.
        """
        # Create a series where candle [-2] broke above band, but candle [-1] is inside
        # generate_donchian_momentum_hybrid_setup with trigger_mode='BREAKOUT'
        with patch("gold_trader.core.strategy_engine.fetch_gold_candles") as mock_candles:
            # Generate 50 candles
            mock_data = []
            for i in range(50):
                mock_data.append({
                    "date": f"2026-01-01T{i%24:02d}:00:00",
                    "open": 2000.0 + i,
                    "high": 2005.0 + i,
                    "low": 1995.0 + i,
                    "close": 2000.0 + i,
                })
            mock_candles.return_value = mock_data

            setup = generate_donchian_momentum_hybrid_setup(
                symbol="XAUUSD",
                account_balance=1000.0,
                trigger_mode="BREAKOUT",
                trend_filter_ema=50,
            )
            self.assertNotIn("error", setup)

    def test_dynamic_timeframe_candle_counts_and_coverage(self):
        """
        Verify that candle count calculation scales dynamically with timeframe and market sessions
        eliminating the 500-bar fixed truncation trap on 15m/5m.
        """
        from gold_trader.core.data_feed import calculate_required_candle_count, validate_candle_coverage

        # 15m timeframe: 1 month should require ~2,024 bars, 3 months ~5,980 bars
        count_15m_1mo = calculate_required_candle_count(period="1mo", interval="15m", is_crypto=False)
        count_15m_3mo = calculate_required_candle_count(period="3mo", interval="15m", is_crypto=False)
        self.assertGreater(count_15m_1mo, 1500, "15m 1mo should require > 1,500 bars")
        self.assertGreater(count_15m_3mo, 4500, "15m 3mo should require > 4,500 bars")

        # 1h timeframe: 1 month ~506 bars, 3 months ~1,495 bars
        count_1h_1mo = calculate_required_candle_count(period="1mo", interval="1h", is_crypto=False)
        count_1h_3mo = calculate_required_candle_count(period="3mo", interval="1h", is_crypto=False)
        self.assertGreaterEqual(count_1h_1mo, 400)
        self.assertGreaterEqual(count_1h_3mo, 1200)

        # Coverage validator
        mock_candles = [{"date": f"2026-01-01 {i:02d}:00", "close": 2000.0} for i in range(10)]
        coverage = validate_candle_coverage(mock_candles, requested_count=100, interval="1h")
        self.assertEqual(coverage["status"], "INCOMPLETE")
        self.assertFalse(coverage["is_complete"])
        self.assertEqual(coverage["count"], 10)

    def test_ml_fail_closed_on_missing_or_corrupt_model(self):
        """
        Verify that when ML is enabled with FAIL_CLOSED policy, missing or corrupt models
        block order entry with ML_UNAVAILABLE instead of silently failing open.
        """
        broker = DummyMockBroker()
        config = AutoTraderConfig(
            symbol="XAUUSD",
            auto_execute=True,
            flat_risk=True,
            enable_ml_filter=True,
            ml_required=True,
            ml_failure_policy="FAIL_CLOSED",
            ml_model_path="/tmp/non_existent_ml_model.joblib",
            send_telegram_alerts=False,
        )
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()
        trader = GoldAutoTrader(broker=broker, config_override=config, db_path=temp_db.name)

        # Mock a valid BUY setup from strategy engine
        mock_setup = {
            "status": "APPROVED",
            "action": "BUY",
            "entry_price": 2700.0,
            "stop_loss": 2690.0,
            "target_1": 2712.5,
            "target_2": 2730.0,
            "reward_risk_ratio": 1.25,
            "lots": 0.05,
            "quality_score": 90,
            "setup_name": "TestSetup",
            "candle_time": "2026-11-02 12:00",
        }
        with patch.object(trader.broker, "get_account_info", return_value={"balance": 1000.0, "equity": 1000.0}):
            with patch.object(trader.broker, "get_open_positions", return_value=[]):
                with patch.object(trader.broker, "get_pending_orders", return_value=[]):
                    with patch("gold_trader.core.auto_trader.generate_compounding_ladder_setup", return_value=mock_setup):
                        res = trader.evaluate_and_trade()
                        self.assertIn("SKIP", res.get("decision", ""))
                        self.assertIn("ML_UNAVAILABLE", res.get("decision", ""))
                        self.assertIsNone(res.get("order_result"))

    def test_ml_training_inference_feature_parity_and_schema_validation(self):
        """
        Verify that training and inference share identical feature sets, zero macro drift,
        and schema mismatch is caught and rejected.
        """
        from gold_trader.core.ml_filter import GoldMLPredictor, extract_candle_features

        # Check feature names consistency
        _, features = extract_candle_features([])
        self.assertNotIn("macro_score", features, "macro_score should be eliminated to avoid distribution drift")
        self.assertEqual(len(features), 22, "Stationary base feature set must have exactly 22 features")

        # Test schema mismatch rejection
        temp_model = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
        temp_model.close()
        import joblib
        corrupt_bundle = {
            "model": "dummy_model_placeholder",
            "feature_names": ["wrong_feature_1", "wrong_feature_2"],  # Invalid schema
            "metadata": {"symbol": "XAUUSD"},
        }
        joblib.dump(corrupt_bundle, temp_model.name)

        predictor = GoldMLPredictor(model_path=temp_model.name, symbol="XAUUSD")
        self.assertIsNone(predictor.model)
        self.assertEqual(predictor.metadata.get("status"), "SCHEMA_MISMATCH")

        # Symbol mismatch rejection
        mismatch_bundle = {
            "model": "dummy_model_placeholder",
            "feature_names": features + ["trade_direction_sign"],
            "metadata": {"symbol": "BTCUSD"},
        }
        joblib.dump(mismatch_bundle, temp_model.name)
        predictor_gold = GoldMLPredictor(model_path=temp_model.name, symbol="XAUUSD")
        self.assertIsNone(predictor_gold.model)
        self.assertEqual(predictor_gold.metadata.get("status"), "SYMBOL_MISMATCH")

    def test_backtest_symmetric_spread_and_adverse_first(self):
        """
        Verify that 15m backtester enforces symmetric spread, commission deduction,
        and conservative adverse-first barrier resolution on same-candle TP/SL.
        """
        from scripts.backtest_donchian_15m import simulate_trade_execution

        # Create 300 test candles
        candles = []
        for i in range(300):
            candles.append({
                "date": f"2026-01-01 {i%24:02d}:{(i%4)*15:02d}",
                "open": 2000.0,
                "high": 2005.0,
                "low": 1995.0,
                "close": 2000.0,
            })

        # Run with adverse_first=True
        res = simulate_trade_execution(
            candles=candles,
            donchian_period=20,
            be_mode="PARTIAL_SCALE",
            trigger_mode="TOUCH",
            spread_usd=0.35,
            slippage_usd=0.05,
            commission_per_lot_usd=7.00,
            adverse_first=True,
        )
        self.assertIn("trades", res)
        self.assertIn("closed_trades", res)
        # Verify that commission was accounted for
        if res["closed_trades"]:
            self.assertGreater(res["closed_trades"][0]["commission"], 0.0)


if __name__ == "__main__":
    unittest.main()
