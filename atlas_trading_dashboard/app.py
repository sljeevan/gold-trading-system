import os
from datetime import datetime, timezone
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Aegis Alpha | Portfolio Overview",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Theme / styling ----
st.markdown("""
<style>
    .block-container {padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1500px;}
    [data-testid="stMetric"] {background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.16); padding: 14px 16px; border-radius: 14px;}
    .asset-card {border: 1px solid rgba(128,128,128,.22); border-radius: 15px; padding: 16px; min-height: 175px;}
    .muted {opacity: .68; font-size: .88rem;}
    .status-pill {display:inline-block; border-radius:999px; padding:3px 9px; font-size:.78rem; font-weight:650; background:rgba(22,163,74,.13); color:#16a34a;}
    .status-idle {background:rgba(128,128,128,.14); color:inherit;}
    .pnl-positive {color:#16a34a; font-weight:700;}
    .pnl-negative {color:#dc2626; font-weight:700;}
    div[data-testid="stTabs"] button {font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ---- Data provider seam ----
# Replace demo_data() with your repository's API/database adapter.
# The UI intentionally does not connect to a broker or place orders.
def demo_data():
    now = datetime.now(timezone.utc)
    assets = [
        {
            "asset": "Gold", "symbol": "XAUUSD", "venue": "XM / MetaTrader 5",
            "status": "IN TRADE", "side": "BUY", "entry": 3940.00, "mark": 3954.20,
            "quantity": 0.03, "unrealized_pnl": 42.60, "realized_today": 86.20,
            "stop_loss": 3925.00, "take_profit": 3980.00, "last_update": now,
            "price_format": 2, "unit": "lots",
        },
        {
            "asset": "Bitcoin", "symbol": "BTC/USD", "venue": "Spot exchange",
            "status": "IN TRADE", "side": "BUY", "entry": 108200.00, "mark": 108456.00,
            "quantity": 0.10, "unrealized_pnl": 25.60, "realized_today": 24.80,
            "stop_loss": None, "take_profit": None, "last_update": now,
            "price_format": 2, "unit": "BTC",
        },
        {
            "asset": "Ethereum", "symbol": "ETH/USD", "venue": "Spot exchange",
            "status": "NO TRADE", "side": None, "entry": None, "mark": 3850.00,
            "quantity": 0.0, "unrealized_pnl": 0.00, "realized_today": 13.80,
            "stop_loss": None, "take_profit": None, "last_update": now,
            "price_format": 2, "unit": "ETH",
        },
    ]

    monthly = pd.DataFrame({
        "month": pd.to_datetime([
            "2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01",
            "2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01",
            "2026-09-01", "2026-10-01"
        ]),
        "pnl": [185, 320, -95, 410, 280, 540, 390, 710, 425, 501],
    })
    overall = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=20, freq="15D"),
        "daily_pnl": [35, 60, -22, 75, 45, -35, 90, 30, 110, -42, 65, 85, -30, 120, 55, 70, -18, 95, 40, 61],
    })
    overall["cumulative_pnl"] = overall["daily_pnl"].cumsum()

    trades = pd.DataFrame([
        {"time": "2026-10-23 13:15:00", "asset": "XAUUSD", "side": "BUY", "qty": "0.03 lots", "status": "OPEN", "entry": 3940.00, "exit": None, "pnl": 42.60, "strategy": "Trend + Pullback"},
        {"time": "2026-10-23 12:40:00", "asset": "BTC/USD", "side": "BUY", "qty": "0.10 BTC", "status": "OPEN", "entry": 108200.00, "exit": None, "pnl": 25.60, "strategy": "Breakout Retest"},
        {"time": "2026-10-22 16:20:00", "asset": "ETH/USD", "side": "BUY", "qty": "0.50 ETH", "status": "CLOSED", "entry": 3812.00, "exit": 3775.20, "pnl": -18.40, "strategy": "Mean Reversion"},
        {"time": "2026-10-22 10:05:00", "asset": "XAUUSD", "side": "SELL", "qty": "0.02 lots", "status": "CLOSED", "entry": 3968.00, "exit": 3924.90, "pnl": 86.20, "strategy": "Trend + Pullback"},
        {"time": "2026-10-21 09:50:00", "asset": "BTC/USD", "side": "SELL", "qty": "0.05 BTC", "status": "CLOSED", "entry": 109500.00, "exit": 108900.00, "pnl": 30.00, "strategy": "Breakout Retest"},
    ])
    return {"assets": assets, "monthly": monthly, "overall": overall, "trades": trades}

def load_data():
    # Set TRADING_API_BASE_URL only after implementing an adapter for your actual API.
    # Expected endpoint contract is documented in README.md. Demo remains default.
    api_base = os.getenv("TRADING_API_BASE_URL", "").strip().rstrip("/")
    if api_base:
        try:
            import requests
            response = requests.get(f"{api_base}/api/dashboard/overview", timeout=2.5)
            response.raise_for_status()
            payload = response.json()
            # A real integration should normalize the payload into the same four keys.
            return normalize_api_payload(payload)
        except Exception as exc:
            st.sidebar.warning(f"API unavailable; showing demo data. ({type(exc).__name__})")
    return demo_data()

def normalize_api_payload(payload):
    """Minimal adapter. API payload must contain assets, monthly, overall, trades."""
    assets = payload.get("assets", [])
    monthly = pd.DataFrame(payload.get("monthly", []))
    overall = pd.DataFrame(payload.get("overall", []))
    trades = pd.DataFrame(payload.get("trades", []))
    if not monthly.empty and "month" in monthly:
        monthly["month"] = pd.to_datetime(monthly["month"])
    if not overall.empty and "date" in overall:
        overall["date"] = pd.to_datetime(overall["date"])
    account = payload.get("account") or {}
    return {"assets": assets, "monthly": monthly, "overall": overall, "trades": trades, "account": account, "is_live": True}

def money(value):
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"

def pnl_color(value):
    try:
        return "normal" if float(value) >= 0 else "inverse"
    except (TypeError, ValueError):
        return "normal"

data = load_data()
assets = data["assets"]
monthly = data["monthly"]
overall = data["overall"]
trades = data["trades"]
account = data.get("account") or {}
is_live = data.get("is_live", False)

# ---- Sidebar ----
with st.sidebar:
    st.title("🛡️ Aegis Alpha")
    st.caption("Quantitative Multi-Asset Trading Engine")
    st.divider()
    st.subheader("Display controls")
    mode = st.selectbox("Environment", ["Paper / Demo", "Live account view"], index=0)
    selected_asset = st.selectbox("Asset filter", ["All assets", "XAUUSD", "BTC/USD", "ETH/USD"])
    chart_range = st.selectbox("P&L range", ["Last 6 months", "Year to date", "All available"])
    show_closed = st.toggle("Include closed trades", value=True)
    st.divider()
    st.subheader("Safety status")
    lock_new_entries = st.toggle("Display: new-entry lock", value=True)
    st.caption("This toggle is visual only. Wire it to a secured backend control before use.")
    st.divider()
    if is_live:
        st.success("🟢 API Connected (Live Data)")
    else:
        st.caption("⚠️ This starter UI uses demo data unless an API adapter is configured. It cannot place trades.")

st.title("🛡️ Aegis Alpha | Portfolio Overview")
st.caption("Gold · Bitcoin · Ethereum  |  Autonomous Quantitative Execution · ML Meta-Labeling · Dynamic Risk Shield")

if is_live and account and account.get("balance") is not None:
    st.success(
        f"🟢 Live MT5 Connection Active — Account #{account.get('account_id', '168987477')} "
        f"({account.get('broker', 'XM MT5')}) | Leverage 1:{account.get('leverage', 1000)}"
    )
elif mode == "Live account view":
    st.warning("Live account view selected, but no live broker connection is implied. Verify the data source indicator before relying on any numbers.")
else:
    st.info("DEMO DATA — sample positions and P&L only. Replace the data provider with your actual trading-system API before using operationally.")

# ---- Account balance & equity metrics ----
if account and account.get("balance") is not None:
    ca1, ca2, ca3 = st.columns(3)
    ca1.metric("MT5 Account Balance", money(account.get("balance")))
    ca2.metric("MT5 Account Equity", money(account.get("equity")), delta=f"Margin: {money(account.get('margin'))}")
    ca3.metric("Free Margin", money(account.get("free_margin")), delta=f"Level: {account.get('margin_level_pct', 0)}%")

# ---- Top-level portfolio metrics ----
open_pnl = sum(float(a.get("unrealized_pnl") or 0) for a in assets)
today_realized = sum(float(a.get("realized_today") or 0) for a in assets)
active_positions = sum(1 for a in assets if a.get("status") == "IN TRADE")
total_realized = float(monthly["pnl"].sum()) if not monthly.empty and "pnl" in monthly else 0.0
gross_display = total_realized + open_pnl

c1, c2, c3, c4 = st.columns(4)
c1.metric("Realized P&L · all shown history", money(total_realized))
c2.metric("Open / unrealized P&L", money(open_pnl), help="Mark-to-market P&L of currently open positions; sample data in demo mode.")
c3.metric("Today's realized P&L", money(today_realized))
c4.metric("Assets in trade", f"{active_positions} / 3")

st.divider()
st.subheader("Asset status")
asset_cols = st.columns(3)
for col, a in zip(asset_cols, assets):
    with col:
        is_open = a.get("status") == "IN TRADE"
        status_label = "🟢 IN TRADE" if is_open else "⚪ NO OPEN TRADE"
        pnl = float(a.get("unrealized_pnl") or 0)
        st.markdown(f"### {a.get('asset', 'Asset')}")
        st.caption(f"{a.get('symbol', '')} · {a.get('venue', '')}")
        st.markdown(f"**{status_label}**")
        st.metric("Live / unrealized P&L", money(pnl), delta=f"{pnl:+,.2f} USD" if is_open else None)
        if is_open:
            side = a.get("side") or "—"
            st.write(f"**Position:** {side} · **Size:** {a.get('quantity', '—')} {a.get('unit', '')}")
            st.write(f"**Entry:** {money(a.get('entry'))}  ")
            st.write(f"**Mark:** {money(a.get('mark'))}")
            st.write(f"**Stop-loss:** {money(a.get('stop_loss'))}  ")
            st.write(f"**Take-profit:** {money(a.get('take_profit'))}")
        else:
            st.write("No open position reported.")
            st.write(f"**Latest mark:** {money(a.get('mark'))}")
        last = a.get("last_update")
        if last:
            st.caption(f"Last update: {last}")

st.divider()

# ---- Charts ----
left, right = st.columns([1, 1.25], gap="large")
with left:
    st.subheader("Month-on-month realized P&L")
    if not monthly.empty and {"month", "pnl"}.issubset(monthly.columns):
        plot_monthly = monthly.copy()
        if chart_range == "Last 6 months":
            plot_monthly = plot_monthly.tail(6)
        elif chart_range == "Year to date":
            plot_monthly = plot_monthly[plot_monthly["month"].dt.year == datetime.now().year]
        plot_monthly["month_label"] = plot_monthly["month"].dt.strftime("%b %Y")
        fig = px.bar(plot_monthly, x="month_label", y="pnl", color="pnl",
                     color_continuous_scale=["#dc6262", "#9ca3af", "#16a34a"],
                     labels={"month_label": "Month", "pnl": "Realized P&L (USD)", "color": "P&L"})
        fig.update_layout(showlegend=False, margin=dict(l=5, r=5, t=10, b=5), height=330)
        fig.add_hline(y=0, line_width=1, line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No monthly P&L records returned by the configured data source.")

with right:
    st.subheader("Overall cumulative P&L")
    if not overall.empty and {"date", "cumulative_pnl"}.issubset(overall.columns):
        plot_overall = overall.copy()
        fig = px.area(plot_overall, x="date", y="cumulative_pnl",
                      labels={"date": "Date", "cumulative_pnl": "Cumulative realized P&L (USD)"})
        fig.update_traces(line_color="#159b80", fillcolor="rgba(21,155,128,.15)")
        fig.update_layout(margin=dict(l=5, r=5, t=10, b=5), height=330)
        fig.add_hline(y=0, line_width=1, line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    elif not overall.empty and {"date", "daily_pnl"}.issubset(overall.columns):
        plot_overall = overall.copy()
        plot_overall["cumulative_pnl"] = plot_overall["daily_pnl"].cumsum()
        fig = px.area(plot_overall, x="date", y="cumulative_pnl")
        fig.update_layout(margin=dict(l=5, r=5, t=10, b=5), height=330)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No overall P&L history returned by the configured data source.")

st.divider()

# ---- Trade table ----
st.subheader("Trade Activity (Fills & Executions)")
if not trades.empty:
    view = trades.copy()
    if selected_asset != "All assets" and "asset" in view.columns:
        view = view[view["asset"].astype(str).str.upper() == selected_asset.upper()]
    if not show_closed and "status" in view.columns:
        view = view[view["status"].astype(str).str.upper() == "OPEN"]
    if "pnl" in view.columns:
        view["pnl"] = view["pnl"].apply(money)

    total_trades = len(view)
    if total_trades > 0:
        c_page, c_size = st.columns([3, 1])
        page_size = c_size.selectbox("Rows per page", [5, 10, 20, 50], index=1, key="trade_page_size")
        total_pages = max(1, (total_trades + page_size - 1) // page_size)
        current_page = c_page.number_input(f"Page (1 of {total_pages})", min_value=1, max_value=total_pages, value=1, step=1, key="trade_page_num")

        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, total_trades)
        paginated_view = view.iloc[start_idx:end_idx]

        st.caption(f"Showing trades {start_idx + 1}–{end_idx} of {total_trades}")
        st.dataframe(paginated_view, use_container_width=True, hide_index=True)
    else:
        st.info("No trades found for the selected filter.")
else:
    st.info("No trades found for the selected filter.")

# ---- Status / safe controls ----
st.divider()
st.subheader("System health checklist")
h1, h2, h3, h4 = st.columns(4)
h1.metric("Market feed", "Demo" if not os.getenv("TRADING_API_BASE_URL") else "API adapter")
h2.metric("MT5 / exchange", "Not verified")
h3.metric("Signal freshness", "Not verified")
h4.metric("New-entry lock", "ON" if lock_new_entries else "OFF")
st.caption(
    "The lock control is a UI placeholder and does not stop the execution engine. "
    "Implement a server-side, authenticated kill switch and reconcile broker positions before production use."
)
st.caption(f"Dashboard rendered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} local server time.")
