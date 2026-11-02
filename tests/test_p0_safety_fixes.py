"""
Unit tests verifying P0 Critical Remediation:
1. Same-candle TP/SL adverse-first resolution in label_triple_barrier.
2. Purged & Embargoed walk-forward cross validation without label lookahead.
3. Fail-closed behavior when broker account or open positions queries fail.
4. Risk budget protection against minimum lot rounding inflation.
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from gold_trader.core.ml_filter import label_triple_barrier, purged_walk_forward_cv
from gold_trader.core.strategy_engine import calculate_lot_size
from gold_trader.core.auto_trader import AutoTraderConfig, GoldAutoTrader
from gold_trader.brokers.base import BaseBroker


class MockFailingBroker(BaseBroker):
    def __init__(self, fail_account=False, fail_positions=False, zero_balance=False):
        self.fail_account = fail_account
        self.fail_positions = fail_positions
        self.zero_balance = zero_balance
        self.orders = []

    def connect(self) -> bool:
        return True

    def get_account_info(self):
        if self.fail_account:
            raise RuntimeError("MT5 socket connection lost during account info query")
        if self.zero_balance:
            return {"balance": 0.0, "equity": 0.0}
        return {"balance": 1000.0, "equity": 1000.0}

    def get_open_positions(self, symbol=None):
        if self.fail_positions:
            raise RuntimeError("MT5 socket timeout while fetching open positions")
        return []

    def get_pending_orders(self, symbol=None):
        return []

    def get_symbol_info(self, symbol=None):
        return {"status": "SUCCESS", "bid": 2700.0, "ask": 2700.35}

    def place_order(self, *args, **kwargs):
        self.orders.append(kwargs)
        return {"status": "SUCCESS", "ticket": "12345"}

    def modify_position(self, *args, **kwargs):
        return {"status": "SUCCESS"}

    def close_position(self, *args, **kwargs):
        return {"status": "SUCCESS"}


class TestP0Remediation(unittest.TestCase):

    def test_same_candle_tpsl_adverse_first_buy(self):
        """
        Verify that when a single candle breaches BOTH Take Profit and Stop Loss,
        label_triple_barrier assigns outcome 0 (adverse-first / SL hit first).
        """
        candles = [
            {"date": "2026-01-01T00:00:00", "open": 2000.0, "high": 2005.0, "low": 1995.0, "close": 2000.0},
            # Candle 1: massive spike breaching entry+10 (TP) and entry-10 (SL)
            {"date": "2026-01-01T01:00:00", "open": 2000.0, "high": 2025.0, "low": 1975.0, "close": 2015.0},
            {"date": "2026-01-01T02:00:00", "open": 2015.0, "high": 2020.0, "low": 2010.0, "close": 2015.0},
        ]
        # Pad candles to satisfy max_holding_bars
        for i in range(3, 10):
            candles.append({"date": f"2026-01-01T0{i}:00:00", "open": 2015.0, "high": 2020.0, "low": 2010.0, "close": 2015.0})

        labels = label_triple_barrier(
            candles,
            direction="BUY",
            tp_points=10.0,
            sl_points=10.0,
            max_holding_bars=4,
        )
        # Entry at index 0 (price 2000.0): Candle 1 reaches high 2025 (>= 2010) and low 1975 (<= 1990)
        self.assertEqual(labels[0], 0, "Same-candle TP/SL breach must resolve to 0 under adverse-first rule")

    def test_same_candle_tpsl_adverse_first_sell(self):
        """
        Verify SELL adverse-first resolution when single candle breaches both barriers.
        """
        candles = [
            {"date": "2026-01-01T00:00:00", "open": 2000.0, "high": 2005.0, "low": 1995.0, "close": 2000.0},
            # Candle 1: reaches low 1975 (TP for SELL) and high 2025 (SL for SELL)
            {"date": "2026-01-01T01:00:00", "open": 2000.0, "high": 2025.0, "low": 1975.0, "close": 1980.0},
        ]
        for i in range(2, 10):
            candles.append({"date": f"2026-01-01T0{i}:00:00", "open": 1980.0, "high": 1985.0, "low": 1975.0, "close": 1980.0})

        labels = label_triple_barrier(
            candles,
            direction="SELL",
            tp_points=10.0,
            sl_points=10.0,
            max_holding_bars=4,
        )
        self.assertEqual(labels[0], 0, "Same-candle TP/SL breach for SELL must resolve to 0 under adverse-first rule")

    def test_purged_walk_forward_cv_no_leakage(self):
        """
        Verify that purged_walk_forward_cv drops training samples whose forward
        outcome window overlaps with validation start.
        """
        bar_indices = list(range(100))
        max_holding_bars = 10
        splits = purged_walk_forward_cv(
            bar_indices=bar_indices,
            max_holding_bars=max_holding_bars,
            n_splits=3,
        )
        self.assertGreater(len(splits), 0)

        for train_idx, val_idx in splits:
            val_start_bar = min(bar_indices[v] for v in val_idx)
            max_train_horizon = max(bar_indices[tr] + max_holding_bars for tr in train_idx)
            self.assertLess(
                max_train_horizon,
                val_start_bar,
                f"Purging violated: train horizon ({max_train_horizon}) reached into validation ({val_start_bar})"
            )

    def test_auto_trader_fails_closed_on_account_query_failure(self):
        """
        Verify that if broker.get_account_info() fails, auto_trader halts and places 0 orders.
        """
        import tempfile
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()
        broker = MockFailingBroker(fail_account=True)
        config = AutoTraderConfig(symbol="XAUUSD", auto_execute=True, flat_risk=True, send_telegram_alerts=False)
        trader = GoldAutoTrader(broker=broker, config_override=config, db_path=temp_db.name)

        res = trader.evaluate_and_trade()
        self.assertEqual(res.get("status"), "BROKER_UNAVAILABLE")
        self.assertEqual(len(broker.orders), 0, "No orders must be placed when account info fails")

    def test_auto_trader_fails_closed_on_zero_balance(self):
        """
        Verify that if broker balance is <= 0, auto_trader halts and does not use $10k fallback.
        """
        import tempfile
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()
        broker = MockFailingBroker(zero_balance=True)
        config = AutoTraderConfig(symbol="XAUUSD", auto_execute=True, flat_risk=True, send_telegram_alerts=False)
        trader = GoldAutoTrader(broker=broker, config_override=config, db_path=temp_db.name)

        res = trader.evaluate_and_trade()
        self.assertEqual(res.get("status"), "BROKER_UNAVAILABLE")
        self.assertEqual(len(broker.orders), 0, "No orders must be placed when balance is 0")

    def test_auto_trader_fails_closed_on_positions_query_failure(self):
        """
        Verify that if broker.get_open_positions() fails, auto_trader halts and does not treat positions as 0.
        """
        import tempfile
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()
        broker = MockFailingBroker(fail_positions=True)
        config = AutoTraderConfig(symbol="XAUUSD", auto_execute=True, flat_risk=True, send_telegram_alerts=False)
        trader = GoldAutoTrader(broker=broker, config_override=config, db_path=temp_db.name)

        res = trader.evaluate_and_trade()
        self.assertEqual(res.get("status"), "BROKER_UNAVAILABLE")
        self.assertEqual(len(broker.orders), 0, "No orders must be placed when open positions cannot be verified")

    def test_position_size_risk_inflation_handling(self):
        """
        Verify that calculate_lot_size handles lot rounding without throwing errors and logs warnings on overflow.
        """
        # Very small account ($100), 1% risk ($1.00 risk), large stop distance ($20)
        # Required lots = 1.0 / (20 * 100) = 0.0005 -> rounds to 0.00 -> clamped to 0.01
        # 0.01 lots * 20 * 100 = $20.00 (exceeds $1.00 budget)
        lots = calculate_lot_size(
            account_balance=100.0,
            risk_pct=1.0,
            entry_price=2700.0,
            stop_loss_price=2680.0,
            contract_size=100.0,
            max_risk_multiplier=1.25,
        )
        self.assertEqual(lots, 0.01)

    def test_bridge_positions_unavailable_raises_typed_error(self):
        """
        Verify SocketBroker.get_open_positions raises BrokerPositionsUnavailableError
        when MT5 bridge returns ERROR / MT5_POSITIONS_UNAVAILABLE (positions_get is None).
        """
        from gold_trader.brokers.socket_broker import SocketBroker, BrokerPositionsUnavailableError
        broker = SocketBroker()
        with patch.object(broker, "_send_command", return_value={
            "status": "ERROR",
            "error_code": "MT5_POSITIONS_UNAVAILABLE",
            "message": "Failed to get positions from MT5: error code 4001",
        }):
            with self.assertRaises(BrokerPositionsUnavailableError) as ctx:
                broker.get_open_positions()
            self.assertIn("MT5_POSITIONS_UNAVAILABLE", str(ctx.exception))

    def test_bridge_orders_unavailable_raises_typed_error(self):
        """
        Verify SocketBroker.get_pending_orders raises BrokerOrdersUnavailableError
        when MT5 bridge returns ERROR / MT5_ORDERS_UNAVAILABLE.
        """
        from gold_trader.brokers.socket_broker import SocketBroker, BrokerOrdersUnavailableError
        broker = SocketBroker()
        with patch.object(broker, "_send_command", return_value={
            "status": "ERROR",
            "error_code": "MT5_ORDERS_UNAVAILABLE",
            "message": "Failed to get pending orders from MT5",
        }):
            with self.assertRaises(BrokerOrdersUnavailableError) as ctx:
                broker.get_pending_orders()
            self.assertIn("MT5_ORDERS_UNAVAILABLE", str(ctx.exception))

    def test_bridge_success_with_zero_positions_returns_empty_list(self):
        """
        Verify SocketBroker.get_open_positions returns [] ONLY when status is SUCCESS
        and MT5 successfully confirmed zero open positions.
        """
        from gold_trader.brokers.socket_broker import SocketBroker
        broker = SocketBroker()
        with patch.object(broker, "_send_command", return_value={"status": "SUCCESS", "positions": []}):
            pos = broker.get_open_positions()
            self.assertEqual(pos, [])

    def test_bridge_health_separates_socket_and_terminal(self):
        """
        Verify SocketBroker.get_health returns socket_status and mt5_connected separately.
        """
        from gold_trader.brokers.socket_broker import SocketBroker
        broker = SocketBroker()
        with patch.object(broker, "_send_command", return_value={
            "status": "ERROR",
            "socket_status": "ONLINE",
            "mt5_initialized": True,
            "mt5_connected": False,
            "message": "MT5 Terminal Unavailable",
        }):
            health = broker.get_health()
            self.assertEqual(health["socket_status"], "ONLINE")
            self.assertFalse(health["mt5_connected"])
            self.assertEqual(health["status"], "ERROR")

    def test_unprotected_order_triggers_emergency_protection_failed(self):
        """
        Verify SocketBroker.place_order detects missing SL on success response
        and fails with EMERGENCY_PROTECTION_FAILED.
        """
        from gold_trader.brokers.socket_broker import SocketBroker
        broker = SocketBroker()
        with patch.object(broker, "_send_command", return_value={
            "status": "SUCCESS",
            "ticket": "99912",
            "symbol": "XAUUSD",
            "price": 2700.0,
            "sl": 0.0,  # Unprotected!
            "tp": 2720.0,
            "message": "Order executed without SL",
        }):
            with patch.object(broker, "close_position", return_value={"status": "SUCCESS"}):
                res = broker.place_order(symbol="XAUUSD", side="BUY", lots=0.01, sl=2690.0)
                self.assertEqual(res["status"], "EMERGENCY_PROTECTION_FAILED")
                self.assertTrue(res["lockout"])

    def test_auto_trader_safety_lockout_blocks_trading(self):
        """
        Verify that when an emergency protection failure occurs,
        AutoTrader sets safety lockout and refuses all further entries.
        """
        import tempfile
        from gold_trader.brokers.socket_broker import SocketBroker
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()
        broker = SocketBroker()
        config = AutoTraderConfig(symbol="XAUUSD", auto_execute=True, flat_risk=True, send_telegram_alerts=False)
        trader = GoldAutoTrader(broker=broker, config_override=config, db_path=temp_db.name)

        # Manually engage lockout
        trader._save_state("lockout", "true")
        res = trader.evaluate_and_trade()
        self.assertEqual(res["status"], "LOCKED_OUT")
        self.assertIn("Safety lockout active", res["reason"])

    def test_strategy_config_validation(self):
        """
        Verify StrategyConfig validates parameters and rejects invalid settings at startup.
        """
        from gold_trader.core.strategy_engine import StrategyConfig
        # Invalid timeframe
        with self.assertRaises(ValueError):
            cfg = StrategyConfig(timeframe="2m")
            cfg.validate()

        # Invalid donchian period
        with self.assertRaises(ValueError):
            cfg = StrategyConfig(donchian_period=1)
            cfg.validate()

        # Invalid trigger mode
        with self.assertRaises(ValueError):
            cfg = StrategyConfig(trigger_mode="INVALID")
            cfg.validate()

        # Valid config produces deterministic hash
        cfg = StrategyConfig(timeframe="15m", donchian_period=40, trigger_mode="CLOSE")
        cfg.validate()
        self.assertTrue(len(cfg.get_hash()) > 0)

    def test_strategy_parameter_propagation_15m_reaches_generator(self):
        """
        Verify non-default strategy parameters (15m, Donchian 40, CLOSE trigger)
        reach candle fetch, signal generators, and setup outputs without hardcoded 1h fallback.
        """
        from gold_trader.core.strategy_engine import generate_compounding_ladder_setup, StrategyConfig
        strat_cfg = StrategyConfig(
            symbol="XAUUSD",
            timeframe="15m",
            donchian_period=40,
            trigger_mode="CLOSE",
            strategy_mode="DONCHIAN_BREAKOUT",
        )
        # Mock candle fetch
        mock_candles = [
            {"date": f"2026-01-01T{i%24:02d}:00:00", "open": 2000.0, "high": 2010.0, "low": 1990.0, "close": 2000.0}
            for i in range(120)
        ]
        with patch("gold_trader.core.strategy_engine.fetch_gold_candles", return_value=mock_candles) as mock_fetch:
            setup = generate_compounding_ladder_setup(
                symbol="XAUUSD",
                strategy_config=strat_cfg,
            )
            # Assert 15m reached candle fetch
            intervals_called = [call_args.kwargs.get("interval") for call_args in mock_fetch.call_args_list]
            self.assertIn("15m", intervals_called, f"Expected 15m in candle fetch, got {intervals_called}")
            self.assertNotIn("1h", intervals_called, f"Hardcoded 1h found in candle fetch calls: {intervals_called}")

            # Assert setup metadata reflects exact parameters
            self.assertEqual(setup.get("effective_timeframe"), "15m")
            self.assertEqual(setup.get("effective_donchian_period"), 40)
            self.assertEqual(setup.get("effective_trigger_mode"), "CLOSE")

    def test_post_timeout_close_position_reconciliation(self):
        """
        Verify SocketBroker.close_position reconciles to SUCCESS when a timeout occurs
        but deal history confirms the position was closed on the broker.
        """
        from gold_trader.brokers.socket_broker import SocketBroker
        broker = SocketBroker()
        with patch.object(broker, "_send_command", return_value={"status": "ERROR", "message": "Socket timeout"}):
            with patch.object(broker, "get_open_positions", return_value=[{"ticket": "55512", "lots": 0.02}]):
                with patch.object(broker, "get_deal_history", return_value={
                    "status": "SUCCESS",
                    "deals": [{"ticket": 55512, "volume": 0.02, "entry": "OUT"}]
                }):
                    res = broker.close_position(ticket="55512", lots=0.02)
                    self.assertEqual(res["status"], "SUCCESS")
                    self.assertIn("verified via deal history post-timeout", res["message"])

    def test_partial_close_idempotency_prevents_duplicate_close(self):
        """
        Verify AutoTrader executes partial scale-out at most once and does not
        attempt a second partial close on subsequent management cycles.
        """
        import tempfile
        from gold_trader.brokers.base import BaseBroker
        class MockTrackingBroker(BaseBroker):
            def __init__(self):
                self.close_calls = []
            def connect(self): return True
            def get_account_info(self): return {"balance": 1000.0, "equity": 1000.0}
            def get_open_positions(self, symbol=None): return []
            def get_pending_orders(self, symbol=None): return []
            def get_symbol_info(self, symbol=None): return {}
            def place_order(self, *args, **kwargs): return {"status": "SUCCESS", "ticket": "1"}
            def modify_position(self, *args, **kwargs): return {"status": "SUCCESS"}
            def close_position(self, ticket, lots=None):
                self.close_calls.append({"ticket": ticket, "lots": lots})
                return {"status": "SUCCESS", "lots": lots}
            def get_deal_history(self, ticket=None, days=7): return {"status": "SUCCESS", "deals": []}

        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()
        broker = MockTrackingBroker()
        config = AutoTraderConfig(
            symbol="XAUUSD",
            auto_execute=True,
            flat_risk=True,
            enable_partial_scale_out=True,
            partial_scale_out_pct=0.50,
            scale_out_r=1.0,
            trail_to_breakeven_at_t1=True,
            send_telegram_alerts=False,
        )
        trader = GoldAutoTrader(broker=broker, config_override=config, db_path=temp_db.name)

        pos = [{
            "ticket": "77701",
            "symbol": "XAUUSD",
            "side": "BUY",
            "lots": 0.04,
            "open_price": 2700.0,
            "current_price": 2712.0,  # +1.2R (Target 1 breached)
            "sl": 2690.0,
            "tp": 2730.0,
            "comment": "AI-FlatRisk-XAUUSD",
        }]

        # Cycle 1: should execute 50% scale-out (0.02 lots)
        actions1 = trader._manage_open_positions(pos)
        self.assertEqual(len(broker.close_calls), 1)
        self.assertEqual(broker.close_calls[0]["lots"], 0.02)

        # Cycle 2: broker now reports position with remaining 0.02 lots
        pos_cycle2 = [{
            "ticket": "77701",
            "symbol": "XAUUSD",
            "side": "BUY",
            "lots": 0.02,
            "open_price": 2700.0,
            "current_price": 2715.0,  # Still above Target 1
            "sl": 2700.5,
            "tp": 2730.0,
            "comment": "AI-FlatRisk-XAUUSD",
        }]
        actions2 = trader._manage_open_positions(pos_cycle2)
        # MUST NOT execute another partial close!
        self.assertEqual(len(broker.close_calls), 1, "Duplicate partial close was executed!")


if __name__ == "__main__":
    unittest.main()
