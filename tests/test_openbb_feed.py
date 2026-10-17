"""
Unit tests for OpenBB Macroeconomic & News Catalyst Guard Integration.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import pytest

from gold_trader.core.openbb_feed import (
    check_high_impact_news_window,
    fetch_openbb_macro_data,
)
from gold_trader.core.auto_trader import AutoTraderConfig, GoldAutoTrader


def test_news_catalyst_guard_active_window():
    """Test that an event within buffer_minutes triggers PAUSE."""
    fixed_now = datetime(2026, 10, 19, 14, 0, 0, tzinfo=timezone.utc)
    mock_events = [
        {
            "source": "FED",
            "title": "FOMC Rate Decision",
            "time_utc": datetime(2026, 10, 19, 14, 5, 0, tzinfo=timezone.utc),  # 5 minutes away
        }
    ]

    with patch("gold_trader.core.openbb_feed._get_cached_economic_calendar", return_value=mock_events):
        result = check_high_impact_news_window(buffer_minutes=15, custom_now=fixed_now)
        assert result["is_in_news_window"] is True
        assert result["action"] == "PAUSE"
        assert "FOMC" in result["event_title"]
        assert abs(result["minutes_to_nearest_event"] - 5.0) < 0.1


def test_news_catalyst_guard_outside_window():
    """Test that an event 2 hours away permits trading."""
    fixed_now = datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc)
    mock_events = [
        {
            "source": "FED",
            "title": "FOMC Rate Decision",
            "time_utc": datetime(2026, 10, 19, 14, 0, 0, tzinfo=timezone.utc),  # 120 minutes away
        }
    ]

    with patch("gold_trader.core.openbb_feed._get_cached_economic_calendar", return_value=mock_events):
        result = check_high_impact_news_window(buffer_minutes=15, custom_now=fixed_now)
        assert result["is_in_news_window"] is False
        assert result["action"] == "TRADE_ALLOWED"
        assert abs(result["minutes_to_nearest_event"] - 120.0) < 0.1


def test_news_catalyst_guard_empty_calendar():
    """Test that empty calendar gracefully allows trading."""
    with patch("gold_trader.core.openbb_feed._get_cached_economic_calendar", return_value=[]):
        result = check_high_impact_news_window(buffer_minutes=15)
        assert result["is_in_news_window"] is False
        assert result["action"] == "TRADE_ALLOWED"


def test_fetch_openbb_macro_data_mocked():
    """Test parsing and composite scoring of macro data."""
    mock_obb = MagicMock()

    # Mock Yield Curve
    class YCItem:
        def __init__(self, mat, rate):
            self.maturity = mat
            self.rate = rate

    yc_mock = MagicMock()
    yc_mock.results = [
        YCItem("year_10", 0.0510),
        YCItem("year_2", 0.0470),
        YCItem("month_3", 0.0430),
    ]
    mock_obb.federal_reserve.yield_curve.return_value = yc_mock

    # Mock DVOL
    class DVOLItem:
        close = 38.5

    dvol_mock = MagicMock()
    dvol_mock.results = [DVOLItem()]
    mock_obb.deribit.volatility.index.return_value = dvol_mock

    # Mock COT
    class COTItem:
        non_commercial_positions_long_all = 200000
        non_commercial_positions_short_all = 40000

    cot_mock = MagicMock()
    cot_mock.results = [COTItem()]
    mock_obb.cftc.cot.return_value = cot_mock

    with patch("gold_trader.core.openbb_feed.get_obb_instance", return_value=mock_obb):
        with patch("gold_trader.core.openbb_feed._MACRO_CACHE", {"timestamp": 0.0, "data": {}}):
            data = fetch_openbb_macro_data()
            assert data["yield_10y"] == 5.10
            assert data["yield_2y"] == 4.70
            assert data["yield_spread_10y_2y"] == 0.40
            assert data["yield_curve_inverted"] is False
            assert data["btc_dvol"] == 38.5
            assert data["gold_cot_net_speculative"] == 160000
            assert data["macro_confluence_score"] >= 1


def test_auto_trader_blocks_entry_during_news_window():
    """Verify GoldAutoTrader evaluates rejection_reasons when News Catalyst Guard is triggered."""
    broker_mock = MagicMock()
    broker_mock.get_account_info.return_value = {"balance": 1000.0, "equity": 1000.0}
    broker_mock.get_open_positions.return_value = []
    broker_mock.get_pending_orders.return_value = []

    cfg = AutoTraderConfig(
        symbol="XAUUSD",
        mode="COMPOUNDING_TOUCH",
        enable_news_catalyst_guard=True,
        news_guard_buffer_minutes=15,
        session_filter=False,
    )

    trader = GoldAutoTrader(broker=broker_mock, config_override=cfg)

    # Mock setup returning BUY
    mock_setup = {
        "action": "BUY",
        "entry_price": 2700.0,
        "stop_loss": 2690.0,
        "take_profit": 2715.0,
        "lots": 0.01,
        "reward_risk_ratio": 1.5,
        "setup_name": "Test Setup",
        "quality_score": 95,
        "candle_time": "2026-10-19T14:00:00Z",
    }

    # Mock news guard returning True
    mock_news = {
        "is_in_news_window": True,
        "minutes_to_nearest_event": 3.0,
        "event_title": "US CPI Release",
        "event_time_utc": "2026-10-19 14:03 UTC",
        "action": "PAUSE",
    }

    with patch("gold_trader.core.auto_trader.generate_compounding_ladder_setup", return_value=mock_setup):
        with patch("gold_trader.core.openbb_feed.check_high_impact_news_window", return_value=mock_news):
            decision = trader.evaluate_and_trade()
            assert "News Catalyst Guard (OpenBB)" in decision.get("decision", "")
            # Ensure no order was sent to broker
            broker_mock.open_position.assert_not_called()
