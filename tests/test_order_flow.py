"""
Unit Tests for ATAS-Grade Order Flow Analysis Engine (UAT).
"""
import os
import tempfile

from gold_trader.core.order_flow import (
    FootprintLevel,
    OrderFlowBar,
    OrderFlowDB,
    OrderFlowEngine,
    Trade,
)


def test_footprint_level_math():
    lvl = FootprintLevel(price=100.0, buy_volume=15.0, sell_volume=5.0, trade_count=4)
    assert lvl.total_volume == 20.0
    assert lvl.delta == 10.0


def test_order_flow_bar_aggregation():
    bar = OrderFlowBar(
        symbol="BTCUSDT",
        timeframe="15m",
        bar_start_ts=1000.0,
        open=100.0,
        high=100.0,
        low=100.0,
        close=100.0,
        tick_size=1.0,
    )

    # 3 aggressive buys at 101
    bar.add_trade(price=101.0, size=2.0, side="BUY")
    bar.add_trade(price=101.0, size=1.5, side="BUY")
    # 1 aggressive sell at 100
    bar.add_trade(price=100.0, size=1.0, side="SELL")

    assert bar.high == 101.0
    assert bar.low == 100.0
    assert bar.close == 100.0
    assert bar.total_volume == 4.5
    assert bar.buy_volume == 3.5
    assert bar.sell_volume == 1.0
    assert bar.delta == 2.5
    assert round(bar.delta_pct, 2) == round(2.5 / 4.5 * 100.0, 2)


def test_vpoc_and_value_area():
    bar = OrderFlowBar(
        symbol="BTCUSDT",
        timeframe="15m",
        bar_start_ts=1000.0,
        open=100.0,
        high=105.0,
        low=95.0,
        close=100.0,
        tick_size=1.0,
    )

    # Add heavy volume at 100 (VPOC)
    bar.add_trade(price=100.0, size=50.0, side="BUY")
    # Moderate volume at 101, 102
    bar.add_trade(price=101.0, size=15.0, side="BUY")
    bar.add_trade(price=102.0, size=10.0, side="BUY")
    # Moderate volume at 99, 98
    bar.add_trade(price=99.0, size=15.0, side="SELL")
    bar.add_trade(price=98.0, size=10.0, side="SELL")

    bar.finalize()

    assert bar.vpoc == 100.0
    assert bar.val <= 100.0
    assert bar.vah >= 100.0
    assert bar.value_area_volume >= 0.70 * bar.total_volume


def test_diagonal_stacked_imbalances():
    bar = OrderFlowBar(
        symbol="BTCUSDT",
        timeframe="15m",
        bar_start_ts=1000.0,
        open=100.0,
        high=105.0,
        low=95.0,
        close=104.0,
        tick_size=1.0,
    )

    # Set up 3 consecutive buy imbalances
    # At level 101: Buy volume 30 vs Sell volume at 100 of 5 -> ratio 6.0 >= 3.0
    # At level 102: Buy volume 40 vs Sell volume at 101 of 2 -> ratio 20.0 >= 3.0
    # At level 103: Buy volume 50 vs Sell volume at 102 of 3 -> ratio 16.6 >= 3.0
    bar.add_trade(price=100.0, size=5.0, side="SELL")
    bar.add_trade(price=101.0, size=30.0, side="BUY")
    bar.add_trade(price=101.0, size=2.0, side="SELL")
    bar.add_trade(price=102.0, size=40.0, side="BUY")
    bar.add_trade(price=102.0, size=3.0, side="SELL")
    bar.add_trade(price=103.0, size=50.0, side="BUY")

    bar.finalize(imbalance_ratio=3.0, min_imbalance_volume=1.0, stacked_min_levels=3)

    assert len(bar.stacked_buy_imbalances) >= 1
    zone = bar.stacked_buy_imbalances[0]
    assert zone.side == "BUY"
    assert zone.levels_count >= 3
    assert zone.start_price == 101.0
    assert zone.end_price == 103.0


def test_delta_divergence_and_absorption():
    bar1 = OrderFlowBar(
        symbol="BTCUSDT",
        timeframe="15m",
        bar_start_ts=1000.0,
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        tick_size=1.0,
    )
    bar1.add_trade(price=101.0, size=10.0, side="BUY")
    bar1.finalize()

    # Bar 2 makes higher high (105) but has heavy negative delta (aggressive selling absorbed)
    bar2 = OrderFlowBar(
        symbol="BTCUSDT",
        timeframe="15m",
        bar_start_ts=1900.0,
        open=101.0,
        high=105.0,
        low=100.0,
        close=102.0,
        tick_size=1.0,
    )
    bar2.add_trade(price=104.0, size=30.0, side="SELL")
    bar2.finalize(prior_bar=bar1)

    assert bar2.divergence == "BEARISH_DIVERGENCE"


def test_order_flow_db_isolated():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test_order_flow.db")
        db = OrderFlowDB(test_db)

        bar = OrderFlowBar(
            symbol="BTCUSDT",
            timeframe="15m",
            bar_start_ts=1000.0,
            open=50000.0,
            high=50100.0,
            low=49950.0,
            close=50080.0,
            tick_size=1.0,
        )
        bar.add_trade(price=50050.0, size=2.5, side="BUY")
        bar.finalize()
        bar.cvd = 2.5

        db.save_bar(bar)
        bars = db.get_recent_bars("BTCUSDT", "15m", 5)
        assert len(bars) == 1
        assert bars[0]["symbol"] == "BTCUSDT"
        assert bars[0]["close"] == 50050.0
        assert bars[0]["delta"] == 2.5
        assert bars[0]["cvd"] == 2.5

        # Record big trade
        trade = Trade(price=50000.0, size=1.5, side="BUY", timestamp=1000.0)
        db.record_big_trade(trade, "BTCUSDT")

        # Record signal
        db.record_signal("BTCUSDT", "STACKED_BUY_IMBALANCE", 50050.0, {"levels": 3}, 1000.0)


def test_order_flow_engine_trade_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test_engine.db")
        engine = OrderFlowEngine(
            symbol="BTCUSDT",
            timeframe_minutes=15,
            tick_size=1.0,
            imbalance_ratio=3.0,
            db_path=test_db,
        )

        # Feed trades across 2 bars
        t0 = (1700000000.0 // 900) * 900  # Exact 15m boundary
        engine.on_trade(price=50000.0, size=1.0, side="BUY", timestamp_s=t0 + 10)
        engine.on_trade(price=50010.0, size=0.5, side="SELL", timestamp_s=t0 + 50)

        metrics = engine.get_developing_metrics()
        assert metrics["symbol"] == "BTCUSDT"
        assert metrics["total_volume"] == 1.5
        assert metrics["delta"] == 0.5

        # Feed trade in next bar (t0 + 901)
        engine.on_trade(price=50020.0, size=2.0, side="BUY", timestamp_s=t0 + 901)

        # Bar 1 should now be completed and saved
        assert len(engine.completed_bars) == 1
        assert engine.completed_bars[0].delta == 0.5
        assert engine.running_cvd == 0.5
        assert engine.current_bar.bar_start_ts == t0 + 900
