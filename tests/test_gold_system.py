"""
Unit tests for the Gold Trading System (Standard Library unittest).
"""
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from gold_trader.core.indicators import (
    calc_ema,
    calc_sma,
    calc_rsi,
    calc_atr,
    calc_supertrend,
    calc_donchian,
    calc_bollinger,
    calc_macd,
    calc_adx,
    extract_all_indicators,
)
from gold_trader.core.fibonacci import analyze_gold_fibonacci
from gold_trader.core.strategy_engine import (
    calculate_lot_size,
    generate_donchian_breakout_setup,
    generate_donchian_momentum_hybrid_setup,
)
from gold_trader.brokers.paper_broker import PaperBroker


class TestGoldTradingSystem(unittest.TestCase):

    def setUp(self):
        """Generate 60 mock candles for testing."""
        self.mock_candles = []
        base_price = 2000.0
        for i in range(60):
            o = base_price + i * 2.0
            h = o + 5.0
            l = o - 4.0
            c = o + 1.5
            self.mock_candles.append({
                "date": f"2026-01-01 {i:02d}:00",
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": 1000.0,
            })

    def test_indicators_calculation(self):
        closes = [c["close"] for c in self.mock_candles]
        highs = [c["high"] for c in self.mock_candles]
        lows = [c["low"] for c in self.mock_candles]

        ema = calc_ema(closes, 20)
        self.assertEqual(len(ema), len(closes))
        self.assertIsNotNone(ema[-1])
        self.assertGreater(ema[-1], 2000.0)

        rsi = calc_rsi(closes, 14)
        self.assertEqual(len(rsi), len(closes))
        self.assertIsNotNone(rsi[-1])
        self.assertTrue(0 <= rsi[-1] <= 100)

        atr = calc_atr(highs, lows, closes, 14)
        self.assertIsNotNone(atr[-1])
        self.assertGreater(atr[-1], 0)

        st = calc_supertrend(highs, lows, closes, 10, 3.0)
        self.assertIn(st["direction"][-1], (1, -1))

        dc = calc_donchian(highs, lows, 20)
        self.assertGreater(dc["upper"][-1], dc["lower"][-1])

        bb = calc_bollinger(closes, 20, 2.0)
        self.assertGreater(bb["upper"][-1], bb["middle"][-1])
        self.assertGreater(bb["middle"][-1], bb["lower"][-1])

        all_ind = extract_all_indicators(self.mock_candles)
        self.assertIn("price", all_ind)
        self.assertIn("ema20", all_ind)
        self.assertIn("supertrend_direction", all_ind)

    def test_fibonacci_golden_pocket(self):
        fib = analyze_gold_fibonacci(self.mock_candles, lookback=40)
        self.assertIn("swing_high", fib)
        self.assertIn("swing_low", fib)
        self.assertGreater(fib["swing_high"], fib["swing_low"])
        self.assertIn("golden_pocket_zone", fib)
        self.assertIn("0.618", fib["retracement_levels"])
        self.assertIn("0.786", fib["retracement_levels"])

    def test_lot_size_calculation(self):
        # $10,000 account, 1% risk ($100 risk), $20 stop distance on Gold (100 oz contract)
        # $100 / ($20 * 100) = 0.05 lots
        lots = calculate_lot_size(
            account_balance=10000.0,
            risk_pct=1.0,
            entry_price=2500.0,
            stop_loss_price=2480.0,
        )
        self.assertEqual(lots, 0.05)

    @patch("gold_trader.core.strategy_engine.fetch_gold_candles")
    def test_donchian_breakout_setup_generation(self, mock_fetch):
        mock_fetch.return_value = self.mock_candles
        setup = generate_donchian_breakout_setup(
            symbol="XAUUSD",
            account_balance=10000.0,
            risk_pct=1.0,
            timeframe="1h",
            donchian_period=20,
        )
        self.assertIn("action", setup)
        self.assertIn("entry_price", setup)
        self.assertIn("stop_loss", setup)
        self.assertIn("target_1", setup)
        self.assertIn("target_2", setup)
        self.assertIn("risk_reward_t1", setup)
        self.assertIn("quality_score", setup)
        self.assertIn("setup_name", setup)
        self.assertGreater(setup["target_1"], setup["stop_loss"])

    @patch("gold_trader.core.strategy_engine.fetch_gold_candles")
    def test_donchian_momentum_hybrid_setup_generation(self, mock_fetch):
        mock_fetch.return_value = self.mock_candles
        setup = generate_donchian_momentum_hybrid_setup(
            symbol="XAUUSD",
            account_balance=10000.0,
            risk_pct=1.0,
            timeframe="1h",
            donchian_period=15,
            rr_multiplier=1.8,
        )
        self.assertIn("action", setup)
        self.assertIn("entry_price", setup)
        self.assertIn("stop_loss", setup)
        self.assertIn("target_1", setup)
        self.assertIn("target_2", setup)
        self.assertIn("risk_reward_t1", setup)
        self.assertIn("quality_score", setup)
        self.assertIn("setup_name", setup)
        self.assertGreater(setup["target_1"], setup["stop_loss"])
        self.assertGreaterEqual(setup["risk_reward_t1"], 1.5)

    def test_paper_broker_workflow(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            broker = PaperBroker(db_path=db_path)
            acct = broker.get_account_info()
            self.assertEqual(acct["balance"], 10000.0)
            self.assertEqual(acct["equity"], 10000.0)

            # Place BUY order
            res = broker.place_order(symbol="XAUUSD", side="BUY", lots=0.1, sl=2450.0, tp=2550.0)
            self.assertEqual(res["status"], "SUCCESS")
            ticket = res["ticket"]
            self.assertIsNotNone(ticket)

            # Check Open Positions
            positions = broker.get_open_positions()
            self.assertEqual(len(positions), 1)
            self.assertEqual(positions[0]["ticket"], ticket)
            self.assertEqual(positions[0]["lots"], 0.1)

            # Modify Position
            mod_res = broker.modify_position(ticket, sl=2460.0, tp=2560.0)
            self.assertEqual(mod_res["status"], "SUCCESS")

            # Partial Close Position (close 0.04 of 0.1 lots)
            partial_res = broker.close_position(ticket, lots=0.04)
            self.assertEqual(partial_res["status"], "SUCCESS")
            self.assertEqual(partial_res["lots"], 0.04)
            pos_mid = broker.get_open_positions()
            self.assertEqual(len(pos_mid), 1)
            self.assertAlmostEqual(pos_mid[0]["lots"], 0.06)

            # Close Remaining Position
            close_res = broker.close_position(ticket)
            self.assertEqual(close_res["status"], "SUCCESS")
            self.assertAlmostEqual(close_res["lots"], 0.06)

            # Verify no open positions remain
            positions_after = broker.get_open_positions()
            self.assertEqual(len(positions_after), 0)

        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_trade_setup_and_backtest(self):
        from gold_trader.core.strategy_engine import generate_gold_trade_setup
        from gold_trader.core.backtester import backtest_gold_strategy, compare_gold_strategies
        from gold_trader.core.multi_agent import run_gold_multi_agent_debate

        # Test trade setup generation
        setup = generate_gold_trade_setup(symbol="GC", account_balance=10000.0, risk_pct=1.0)
        self.assertIn("action", setup)
        self.assertIn("entry_price", setup)
        self.assertIn("stop_loss", setup)
        self.assertIn("target_1", setup)
        self.assertIn("suggested_lot_size", setup)

        # Test backtest
        bt = backtest_gold_strategy(symbol="GC", strategy="supertrend", period="3mo", interval="1h")
        self.assertIn("metrics", bt)
        self.assertIn("win_rate_pct", bt["metrics"])

        # Test strategy comparison
        comp = compare_gold_strategies(symbol="GC", period="3mo", interval="1h")
        self.assertIn("leaderboard", comp)
        self.assertGreaterEqual(len(comp["leaderboard"]), 1)

        # Test multi-agent debate
        debate = run_gold_multi_agent_debate(symbol="GC", timeframe="1h")
        self.assertIn("consensus", debate)
        self.assertIn("decision", debate["consensus"])

        # Test EMA 5/20 Crossover Backtest (Long & Short)
        from gold_trader.core.backtester import backtest_ema_crossover_strategy
        ema_bt = backtest_ema_crossover_strategy(symbol="GC", fast_period=5, slow_period=20, period="1mo", interval="1h")
        self.assertIn("metrics", ema_bt)
        self.assertIn("long_trades", ema_bt["metrics"])
        self.assertIn("short_trades", ema_bt["metrics"])
        self.assertIn("total_points", ema_bt["metrics"])
        self.assertIn("mfe_distribution", ema_bt)

        # Test EMA with minimum target points
        ema_target_bt = backtest_ema_crossover_strategy(
            symbol="GC", fast_period=5, slow_period=20, min_target_points=5.0, stop_loss_points=5.0, period="1mo", interval="1h"
        )
        self.assertIn("metrics", ema_target_bt)
        self.assertGreater(ema_target_bt["metrics"]["total_trades"], 0)

    def test_demo_live_config_resolution(self):
        from gold_trader.config import Settings
        from gold_trader.brokers.xm_mt5_broker import XmMT5Broker
        from gold_trader.brokers.metaapi_broker import MetaApiBroker

        custom_settings = Settings(
            BROKER_MODE="paper",
            XM_ACCOUNT_TYPE="demo",
            XM_DEMO_LOGIN=111111,
            XM_DEMO_PASSWORD="demo_password",
            XM_DEMO_SERVER="XMGlobal-Demo 01",
            XM_LIVE_LOGIN=999999,
            XM_LIVE_PASSWORD="live_password",
            XM_LIVE_SERVER="XMGlobal-Real 50",
            METAAPI_DEMO_ACCOUNT_ID="meta_demo_123",
            METAAPI_LIVE_ACCOUNT_ID="meta_live_456",
        )

        # Test Demo Resolution
        demo_login, demo_pwd, demo_srv = custom_settings.get_xm_credentials("demo")
        self.assertEqual(demo_login, 111111)
        self.assertEqual(demo_pwd, "demo_password")
        self.assertEqual(demo_srv, "XMGlobal-Demo 01")
        self.assertEqual(custom_settings.get_metaapi_account_id("demo"), "meta_demo_123")

        # Test Live Resolution
        live_login, live_pwd, live_srv = custom_settings.get_xm_credentials("live")
        self.assertEqual(live_login, 999999)
        self.assertEqual(live_pwd, "live_password")
        self.assertEqual(live_srv, "XMGlobal-Real 50")
        self.assertEqual(custom_settings.get_metaapi_account_id("live"), "meta_live_456")

        # Test Broker Account Info offline fallback tagging
        broker_demo = XmMT5Broker(account_type="demo", login=111111, server="XMGlobal-Demo 01")
        info_demo = broker_demo.get_account_info()
        self.assertIn("DEMO", info_demo["broker"])

        broker_live = XmMT5Broker(account_type="live", login=999999, server="XMGlobal-Real 50")
        info_live = broker_live.get_account_info()
        self.assertIn("LIVE", info_live["broker"])

    def test_socket_broker_offline_and_mock_server(self):
        import json
        import socket
        import threading
        from gold_trader.brokers.socket_broker import SocketBroker

        # 1. Offline Test (non-existent port)
        broker_offline = SocketBroker(host="127.0.0.1", port=59999, timeout=0.1)
        acct_offline = broker_offline.get_account_info()
        self.assertIn("OFFLINE", acct_offline["broker"])
        self.assertEqual(acct_offline["balance"], 0.0)

        # 2. Mock MT5 Bridge TCP Server Test
        def mock_server():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", 58888))
                s.listen(1)
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(1024).decode()
                    req = json.loads(data.strip())
                    if req.get("command") == "ACCOUNT_INFO":
                        resp = {
                            "status": "SUCCESS",
                            "login": 168987477,
                            "server": "XMGlobal-MT5 2",
                            "balance": 10000.0,
                            "equity": 10000.0,
                            "margin": 0.0,
                            "free_margin": 10000.0,
                            "margin_level_pct": 0.0,
                            "leverage": 100,
                            "currency": "USD",
                            "trade_mode": "DEMO"
                        }
                        conn.sendall((json.dumps(resp) + "\n").encode())

        server_thread = threading.Thread(target=mock_server, daemon=True)
        server_thread.start()
        time.sleep(0.2)

        # Connect SocketBroker to mock server
        broker = SocketBroker(host="127.0.0.1", port=58888, timeout=2.0)
        acct = broker.get_account_info()
        self.assertEqual(acct["account_id"], "168987477")
        self.assertEqual(acct["balance"], 10000.0)
        self.assertIn("XMGlobal-MT5 2", acct["broker"])


if __name__ == "__main__":
    unittest.main()


