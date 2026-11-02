import os
import sqlite3
import tempfile
import pytest
from datetime import datetime, timezone
from gold_trader.core.daily_journal import DailyJournalEngine, ASSET_REGISTRY


class TestDailyJournal:
    @pytest.fixture
    def setup_env(self, tmp_path):
        journals_dir = str(tmp_path / "journals")
        journal_db = str(tmp_path / "trading_journal.db")
        trade_db = str(tmp_path / "auto_trader.db")

        # Initialize mock trade executions table
        conn = sqlite3.connect(trade_db)
        conn.execute(
            """
            CREATE TABLE trade_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                ticket TEXT,
                symbol TEXT,
                action TEXT,
                lots REAL,
                entry_price REAL,
                stop_loss REAL,
                target_1 REAL,
                target_2 REAL,
                rr_ratio REAL,
                status TEXT,
                close_price REAL,
                pnl REAL,
                close_time TEXT,
                comment TEXT,
                tp1_taken INTEGER DEFAULT 0,
                peak_price REAL,
                partial_closed_lots REAL DEFAULT 0.0,
                partial_realized_usd REAL DEFAULT 0.0,
                initial_stop_loss REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE scan_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                price REAL,
                bias TEXT,
                action TEXT,
                details_json TEXT
            )
            """
        )
        conn.commit()
        conn.close()

        engine = DailyJournalEngine(journals_dir=journals_dir, journal_db_path=journal_db)
        return engine, trade_db, journals_dir, journal_db

    def test_pnl_not_double_counted_with_partial_realized(self, setup_env):
        """Verify that partial_realized_usd is NOT added to pnl (preventing double counting)."""
        engine, trade_db, journals_dir, _ = setup_env
        asset = {
            "symbol": "XAUUSD",
            "name": "Spot Gold",
            "icon": "🟡",
            "model": "Model 1",
            "contract_size": 100.0,
            "fixed_risk_usd": 75.44,
        }

        conn = sqlite3.connect(trade_db)
        # Position closed with total pnl = 73.65 and partial_realized_usd = 25.29
        conn.execute(
            """
            INSERT INTO trade_executions
            (timestamp, ticket, symbol, action, lots, entry_price, stop_loss, target_1, target_2, rr_ratio, status, close_price, pnl, close_time, comment, tp1_taken, peak_price, partial_closed_lots, partial_realized_usd, initial_stop_loss)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-10-29T10:00:00+00:00",
                "12345",
                "XAUUSD",
                "SELL",
                0.03,
                4020.00,
                3996.00, # trailed SL
                3996.00,
                3948.00,
                1.25,
                "CLOSED",
                3996.00,
                73.65, # Total net realized PnL in MT5
                "2026-10-30T05:30:00+00:00",
                "AI-FlatRisk-XAUUSD",
                1,
                3992.00,
                0.01,
                25.29, # Banked TP1 scale-out
                4045.00, # Initial planned SL
            ),
        )
        conn.commit()
        conn.close()

        # Fetch for 2026-10-30 (close date)
        trades_close_day = engine._fetch_trades_for_date(trade_db, asset, "2026-10-30")
        assert len(trades_close_day) == 1
        trade = trades_close_day[0]
        # Total realized MUST be 73.65, NOT 73.65 + 25.29 = 98.94
        assert trade["total_realized_usd"] == 73.65
        assert trade["planned_risk_usd"] == pytest.approx(75.0, abs=1.0) # (4045 - 4020) * 0.03 * 100 = 75.0
        assert trade["r_multiple"] == pytest.approx(0.98, abs=0.05)

        # Fetch for 2026-10-29 (open date) -> should be 0 because trade closed on 2026-10-30
        trades_open_day = engine._fetch_trades_for_date(trade_db, asset, "2026-10-29")
        assert len(trades_open_day) == 0

    def test_trailed_stop_planned_risk_fallback(self, setup_env):
        """Verify that when initial_stop_loss is not available and SL was trailed into profit, fixed_risk_usd is used."""
        engine, trade_db, _, _ = setup_env
        asset = {
            "symbol": "BTCUSD",
            "name": "Bitcoin",
            "icon": "🟠",
            "model": "Model 1",
            "contract_size": 1.0,
            "fixed_risk_usd": 75.44,
        }

        conn = sqlite3.connect(trade_db)
        # BUY order where SL was trailed past entry into profit: entry=80000, sl=80500
        conn.execute(
            """
            INSERT INTO trade_executions
            (timestamp, ticket, symbol, action, lots, entry_price, stop_loss, target_1, target_2, rr_ratio, status, close_price, pnl, close_time, comment, tp1_taken, peak_price, partial_closed_lots, partial_realized_usd, initial_stop_loss)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-10-30T10:00:00+00:00",
                "99999",
                "BTCUSD",
                "BUY",
                0.15,
                80000.0,
                80500.0, # in-profit trailed stop
                81000.0,
                82000.0,
                1.25,
                "CLOSED",
                80500.0,
                75.00,
                "2026-10-30T14:00:00+00:00",
                "AI-FlatRisk-BTCUSD",
                1,
                81200.0,
                0.07,
                42.00,
                None, # No initial_stop_loss recorded
            ),
        )
        conn.commit()
        conn.close()

        trades = engine._fetch_trades_for_date(trade_db, asset, "2026-10-30")
        assert len(trades) == 1
        t = trades[0]
        assert t["total_realized_usd"] == 75.00
        # Should fallback to fixed_risk_usd (75.44) rather than abs(80000 - 80500) * 0.15
        assert t["planned_risk_usd"] == 75.44
        assert t["r_multiple"] == pytest.approx(0.99, abs=0.02)
