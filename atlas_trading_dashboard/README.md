# Atlas Trade — Multi-Asset Dashboard Starter

A lightweight Streamlit UI starter for a personal trading system focused on:

- Gold: XAUUSD through XM / MetaTrader 5
- Bitcoin: BTC spot
- Ethereum: ETH spot
- Per-asset open-trade status and unrealized P&L
- Month-on-month realized P&L
- Overall cumulative realized P&L
- Trade history and basic system-health placeholders

## Important

This project ships in **demo mode** with fictional sample prices, positions, and P&L. It does not connect to MT5, an exchange, or place/cancel orders. Do not use sample values to make trading decisions.

The UI is deliberately a separate monitoring layer. Do not make trade execution depend on Streamlit or the browser.

## Run locally

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit, typically `http://localhost:8501`.

## Connect to your backend

The app checks the environment variable `TRADING_API_BASE_URL`. If set, it attempts:

`GET {TRADING_API_BASE_URL}/api/dashboard/overview`

If the endpoint fails, it falls back to demo data and displays a warning.

The endpoint should return JSON shaped like this:

```json
{
  "assets": [
    {
      "asset": "Gold",
      "symbol": "XAUUSD",
      "venue": "XM / MetaTrader 5",
      "status": "IN TRADE",
      "side": "BUY",
      "entry": 3940.0,
      "mark": 3954.2,
      "quantity": 0.03,
      "unrealized_pnl": 42.6,
      "realized_today": 86.2,
      "stop_loss": 3925.0,
      "take_profit": 3980.0,
      "last_update": "2026-10-23T12:00:00Z",
      "price_format": 2,
      "unit": "lots"
    }
  ],
  "monthly": [
    {"month": "2026-08-01", "pnl": 710.0},
    {"month": "2026-09-01", "pnl": 425.0},
    {"month": "2026-10-01", "pnl": 501.0}
  ],
  "overall": [
    {"date": "2026-10-01", "cumulative_pnl": 1636.0},
    {"date": "2026-10-15", "cumulative_pnl": 2137.0}
  ],
  "trades": [
    {
      "time": "2026-10-23 12:00:00",
      "asset": "XAUUSD",
      "side": "BUY",
      "qty": "0.03 lots",
      "status": "OPEN",
      "entry": 3940.0,
      "exit": null,
      "pnl": 42.6,
      "strategy": "Trend + Pullback"
    }
  ]
}
```

Set it before starting Streamlit:

```bash
# Linux/macOS
export TRADING_API_BASE_URL="http://127.0.0.1:8000"

# PowerShell
$env:TRADING_API_BASE_URL="http://127.0.0.1:8000"
streamlit run app.py
```

For a deployed environment, protect the API with authentication and TLS. Do not expose broker credentials to the dashboard/browser.

## Suggested backend data rules

1. Derive open positions from broker/exchange-confirmed positions, not only from model signals.
2. Calculate unrealized P&L from the correct contract/lot size, mark price, account currency and fees. XAUUSD CFD contract calculations are broker-specific.
3. Calculate realized P&L from closed fills. Include fees, commissions and funding where applicable.
4. Aggregate realized P&L by calendar month in the backend/database; distinguish the current month-to-date from completed months.
5. Keep a cumulative realized P&L series and show deposits/withdrawals separately from trading performance.
6. Expose feed timestamps and mark stale prices as stale.
7. Implement any emergency lock as an authenticated server-side execution control. The checkbox in this starter is only a visual placeholder.
8. Keep database queries bounded and use caching/pre-aggregated statistics so dashboard refreshes do not trigger full backtests or expensive model imports.

## Files

- `app.py` — complete Streamlit dashboard UI with demo data and an API adapter seam
- `requirements.txt` — runtime dependencies
