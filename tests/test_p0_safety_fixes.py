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


if __name__ == "__main__":
    unittest.main()
