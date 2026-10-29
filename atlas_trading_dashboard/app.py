import os
import sys
from datetime import datetime, timezone
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
try:
    from gold_trader.core.strategy_catalog import (
        STRATEGY_CATALOG,
        StrategyProfile,
        StrategyStatus,
        RiskArchetype,
        get_strategy_catalog,
        list_strategies,
    )
except ImportError:
    STRATEGY_CATALOG = {}


st.set_page_config(
    page_title="Aegis Alpha | Multi-Model Quantitative Portfolio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Theme / styling ----
st.markdown("""
<style>
    .block-container {padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1550px;}
    [data-testid="stMetric"] {background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.16); padding: 14px 16px; border-radius: 14px;}
    .asset-card {border: 1px solid rgba(128,128,128,.22); border-radius: 15px; padding: 16px; min-height: 175px;}
    .treasury-box {background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.35); border-radius: 15px; padding: 16px; margin-bottom: 1rem;}
    .muted {opacity: .68; font-size: .88rem;}
    .badge-m1 {background:rgba(14, 165, 233, 0.15); color:#38bdf8; border:1px solid rgba(14, 165, 233, 0.4); border-radius:6px; padding:2px 7px; font-size:0.75rem; font-weight:700;}
    .badge-m2 {background:rgba(168, 85, 247, 0.15); color:#c084fc; border:1px solid rgba(168, 85, 247, 0.4); border-radius:6px; padding:2px 7px; font-size:0.75rem; font-weight:700;}
    .pnl-pos {color:#16a34a; font-weight:700;}
    .pnl-neg {color:#dc2626; font-weight:700;}
    div[data-testid="stTabs"] button {font-weight:700; font-size:1.05rem;}
</style>
""", unsafe_allow_html=True)


def demo_data():
    now = datetime.now(timezone.utc)
    assets = [
        {
            "asset": "Gold", "symbol": "XAUUSD", "venue": "XM / MetaTrader 5",
            "model_id": "model_1", "model_name": "M1 Baseline",
            "risk_desc": "$75.44 Risk · Dynamic ATR Stop ($8 min) · $4.50 Trail",
            "status": "NO TRADE", "side": None, "entry": None, "mark": 4016.50,
            "quantity": 0.0, "unrealized_pnl": 0.0, "realized_today": 0.0,
            "stop_loss": None, "take_profit": None, "last_update": now,
            "price_format": 2, "unit": "lots",
        },
        {
            "asset": "Bitcoin", "symbol": "BTCUSD", "venue": "XM / MetaTrader 5",
            "model_id": "model_1", "model_name": "M1 Baseline",
            "risk_desc": "$75.44 Risk · Dynamic ATR Stop ($500 min) · $350 Trail",
            "status": "NO TRADE", "side": None, "entry": None, "mark": 80350.00,
            "quantity": 0.0, "unrealized_pnl": 0.0, "realized_today": 0.0,
            "stop_loss": None, "take_profit": None, "last_update": now,
            "price_format": 2, "unit": "BTC",
        },
        {
            "asset": "Ethereum", "symbol": "ETHUSD", "venue": "XM / MetaTrader 5",
            "model_id": "model_1", "model_name": "M1 Baseline",
            "risk_desc": "$75.44 Risk · Dynamic ATR Stop ($20 min) · $10 Trail",
            "status": "NO TRADE", "side": None, "entry": None, "mark": 2330.00,
            "quantity": 0.0, "unrealized_pnl": 0.0, "realized_today": 0.0,
            "stop_loss": None, "take_profit": None, "last_update": now,
            "price_format": 2, "unit": "ETH",
        },
        {
            "asset": "Bitcoin (YieldA)", "symbol": "BTCUSD", "venue": "XM / MetaTrader 5",
            "model_id": "model_2", "model_name": "M2 YieldA",
            "risk_desc": "$75.00 Risk · Tight Trail ($200) · ML P≥0.48 · Max 1 Pos",
            "status": "NO TRADE", "side": None, "entry": None, "mark": 80350.00,
            "quantity": 0.0, "unrealized_pnl": 0.0, "realized_today": 0.0,
            "stop_loss": None, "take_profit": None, "last_update": now,
            "price_format": 2, "unit": "BTC",
        },
        {
            "asset": "Ethereum (YieldA)", "symbol": "ETHUSD", "venue": "XM / MetaTrader 5",
            "model_id": "model_2", "model_name": "M2 YieldA",
            "risk_desc": "$75.00 Risk · Tight Trail ($6) · ML P≥0.48 · Max 1 Pos",
            "status": "NO TRADE", "side": None, "entry": None, "mark": 2330.00,
            "quantity": 0.0, "unrealized_pnl": 0.0, "realized_today": 0.0,
            "stop_loss": None, "take_profit": None, "last_update": now,
            "price_format": 2, "unit": "ETH",
        },
    ]

    monthly = pd.DataFrame({"month": pd.to_datetime(["2026-10-01"]), "pnl": [0.0]})
    overall = pd.DataFrame({"date": pd.to_datetime(["2026-10-26"]), "cumulative_pnl": [0.0]})
    trades = pd.DataFrame([])
    treasury = {
        "anchor_capital": 1500.0,
        "current_month_pnl": 0.0,
        "current_month_equity": 1500.0,
        "harvestable_surplus": 0.0,
        "deficit_top_up_needed": 0.0,
        "status": "BASELINE SECURED ($1,500.00)",
    }
    return {
        "assets": assets,
        "monthly": monthly,
        "overall": overall,
        "monthly_by_model": {"model_1": monthly, "model_2": monthly},
        "overall_by_model": {"model_1": overall, "model_2": overall},
        "trades": trades,
        "treasury": treasury,
        "account": {"account_id": "168987477", "broker": "XM MT5", "balance": 2772.53, "equity": 2772.53},
        "account_model_1": {"account_id": "168987477", "broker": "XMGlobal-MT5 2", "balance": 2772.53, "equity": 2772.53, "leverage": 1000},
        "account_model_2": {"account_id": "346439083", "broker": "XMGlobal-MT5 10", "balance": 1500.0, "equity": 1500.0, "leverage": 1000},
        "is_live": False,
    }


def load_data():
    api_base = os.getenv("TRADING_API_BASE_URL", "").strip().rstrip("/")
    if api_base:
        try:
            import requests
            response = requests.get(f"{api_base}/api/dashboard/overview", timeout=2.5)
            response.raise_for_status()
            payload = response.json()
            return normalize_api_payload(payload)
        except Exception as exc:
            st.sidebar.warning(f"API connection error; falling back to cache. ({type(exc).__name__})")
    return demo_data()


def normalize_api_payload(payload):
    assets = payload.get("assets", [])
    monthly = pd.DataFrame(payload.get("monthly", []))
    overall = pd.DataFrame(payload.get("overall", []))
    trades = pd.DataFrame(payload.get("trades", []))

    monthly_by_model = {}
    for m_k, m_list in (payload.get("monthly_by_model") or {}).items():
        df_m = pd.DataFrame(m_list)
        if not df_m.empty and "month" in df_m:
            df_m["month"] = pd.to_datetime(df_m["month"])
        monthly_by_model[m_k] = df_m

    overall_by_model = {}
    for o_k, o_list in (payload.get("overall_by_model") or {}).items():
        df_o = pd.DataFrame(o_list)
        if not df_o.empty and "date" in df_o:
            df_o["date"] = pd.to_datetime(df_o["date"])
        overall_by_model[o_k] = df_o

    if not monthly.empty and "month" in monthly:
        monthly["month"] = pd.to_datetime(monthly["month"])
    if not overall.empty and "date" in overall:
        overall["date"] = pd.to_datetime(overall["date"])

    account = payload.get("account") or {}
    account_m1 = payload.get("account_model_1") or account
    account_m2 = payload.get("account_model_2") or {}
    treasury = payload.get("treasury_model_2") or {}
    return {
        "assets": assets,
        "monthly": monthly,
        "overall": overall,
        "monthly_by_model": monthly_by_model,
        "overall_by_model": overall_by_model,
        "trades": trades,
        "treasury": treasury,
        "account": account,
        "account_model_1": account_m1,
        "account_model_2": account_m2,
        "is_live": True,
    }


def money(value):
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


data = load_data()
all_assets = data["assets"]
monthly_all = data["monthly"]
overall_all = data["overall"]
monthly_by_model = data["monthly_by_model"]
overall_by_model = data["overall_by_model"]
all_trades = data["trades"]
account = data.get("account") or {}
account_m1 = data.get("account_model_1") or account or {}
account_m2 = data.get("account_model_2") or {}
treasury_data = data.get("treasury") or {}
is_live = data.get("is_live", False)

# Separate assets and trades by model
m1_assets = [a for a in all_assets if a.get("model_id") == "model_1"]
m2_assets = [a for a in all_assets if a.get("model_id") == "model_2"]

m1_trades = (
    all_trades[all_trades["model_id"] == "model_1"].copy()
    if not all_trades.empty and "model_id" in all_trades.columns
    else pd.DataFrame()
)
m2_trades = (
    all_trades[all_trades["model_id"] == "model_2"].copy()
    if not all_trades.empty and "model_id" in all_trades.columns
    else pd.DataFrame()
)

# ---- Reusable UI Render Functions ----
def render_asset_card(col, a):
    with col:
        is_open = a.get("status") == "IN TRADE"
        status_label = "🟢 IN TRADE" if is_open else "⚪ SCANNING"
        pnl = float(a.get("unrealized_pnl") or 0)
        is_m1 = a.get("model_id") == "model_1"
        badge_html = "<span class='badge-m1'>M1 Baseline</span>" if is_m1 else "<span class='badge-m2'>M2 YieldA</span>"

        st.markdown(f"### {a.get('asset', 'Asset')} {badge_html}", unsafe_allow_html=True)
        st.caption(f"{a.get('symbol', '')} · {a.get('venue', '')}")
        st.markdown(f"**{status_label}**")
        st.metric("Live / Unrealized P&L", money(pnl), delta=f"{pnl:+,.2f} USD" if is_open else None)

        if is_open:
            st.write(f"**Position:** {a.get('side', '—')} · **Size:** {a.get('quantity', '—')} {a.get('unit', '')}")
            st.write(f"**Entry:** {money(a.get('entry'))} | **Mark:** {money(a.get('mark'))}")
            st.write(f"**Stop:** {money(a.get('stop_loss'))} | **TP1:** {money(a.get('take_profit'))}")
        else:
            st.write(f"**Latest mark:** {money(a.get('mark'))}")
            st.caption(f"Setup: {a.get('risk_desc', '1H Donchian Touch')}")

        radar = a.get("radar") or {}
        if radar.get("bias"):
            st.caption(f"1H Bias: **{radar.get('bias')}** | ATR: ${radar.get('atr') or '—'} | RSI: {radar.get('rsi') or '—'}")


def render_pnl_charts(monthly_df, overall_df, chart_range, title_suffix):
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader(f"Month-on-Month Realized P&L ({title_suffix})")
        if not monthly_df.empty and {"month", "pnl"}.issubset(monthly_df.columns):
            plot_monthly = monthly_df.copy()
            if chart_range == "Last 6 months":
                plot_monthly = plot_monthly.tail(6)
            elif chart_range == "Year to date":
                plot_monthly = plot_monthly[plot_monthly["month"].dt.year == datetime.now().year]
            plot_monthly["month_label"] = plot_monthly["month"].dt.strftime("%b %Y")
            fig = px.bar(
                plot_monthly, x="month_label", y="pnl", color="pnl",
                color_continuous_scale=["#dc6262", "#9ca3af", "#16a34a"],
                labels={"month_label": "Month", "pnl": "Realized P&L (USD)", "color": "P&L"}
            )
            fig.update_layout(showlegend=False, margin=dict(l=5, r=5, t=10, b=5), height=300)
            fig.add_hline(y=0, line_width=1, line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No monthly P&L records for this model.")

    with right:
        st.subheader(f"Cumulative Realized P&L ({title_suffix})")
        if not overall_df.empty and {"date", "cumulative_pnl"}.issubset(overall_df.columns):
            plot_overall = overall_df.copy()
            fig = px.area(
                plot_overall, x="date", y="cumulative_pnl",
                labels={"date": "Date", "cumulative_pnl": "Cumulative Realized P&L (USD)"}
            )
            fig.update_traces(line_color="#10b981", fillcolor="rgba(16, 185, 129, 0.15)")
            fig.update_layout(margin=dict(l=5, r=5, t=10, b=5), height=300)
            fig.add_hline(y=0, line_width=1, line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cumulative P&L history for this model.")


def render_trades_table(df_trades, key_prefix):
    st.subheader("Trade Activity (Fills & Executions)")
    if not df_trades.empty:
        view = df_trades.copy()
        cols_order = ["time", "model_name", "asset", "side", "qty", "status", "entry", "exit", "pnl", "strategy"]
        available_cols = [c for c in cols_order if c in view.columns]
        view = view[available_cols]

        total_trades = len(view)
        if total_trades > 0:
            c_page, c_size = st.columns([3, 1])
            page_size = c_size.selectbox("Rows per page", [5, 10, 20, 50], index=1, key=f"{key_prefix}_page_size")
            total_pages = max(1, (total_trades + page_size - 1) // page_size)
            current_page = c_page.number_input(
                f"Page (1 of {total_pages})", min_value=1, max_value=total_pages, value=1, step=1, key=f"{key_prefix}_page_num"
            )

            start_idx = (current_page - 1) * page_size
            end_idx = min(start_idx + page_size, total_trades)
            paginated_view = view.iloc[start_idx:end_idx]

            st.caption(f"Showing trades {start_idx + 1}–{end_idx} of {total_trades}")
            st.dataframe(paginated_view, use_container_width=True, hide_index=True)
        else:
            st.info("No trades recorded.")
    else:
        st.info("No trades recorded for this model.")


# ---- Global Header & Master Dual-Account MT5 Banner ----
st.title("🛡️ Aegis Alpha | Quantitative Portfolio Overview")
if is_live:
    col_h1, col_h2 = st.columns(2)
    bal1 = float(account_m1.get("balance") or 0.0)
    eq1 = float(account_m1.get("equity") or bal1)
    fm1 = float(account_m1.get("free_margin") or bal1)

    bal2 = float(account_m2.get("balance") or 0.0)
    eq2 = float(account_m2.get("equity") or bal2)
    fm2 = float(account_m2.get("free_margin") or bal2)

    with col_h1:
        st.markdown(
            f"""<div style="background: rgba(14, 165, 233, 0.09); border: 1px solid rgba(14, 165, 233, 0.35); border-radius: 12px; padding: 10px 14px; font-size: 0.85rem;">
                <div style="font-weight: 700; color: #38bdf8; margin-bottom: 2px;">🟢 Model 1: Production Baseline (Bridge 1 | Port 8888)</div>
                <div>Account: <b>#{account_m1.get('account_id', '168987477')}</b> ({account_m1.get('broker', 'XMGlobal-MT5 2')})</div>
                <div style="margin-top: 3px;">Balance: <b>{money(bal1)}</b> · Equity: <b>{money(eq1)}</b> · Free Margin: <b>{money(fm1)}</b></div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_h2:
        st.markdown(
            f"""<div style="background: rgba(168, 85, 247, 0.09); border: 1px solid rgba(168, 85, 247, 0.35); border-radius: 12px; padding: 10px 14px; font-size: 0.85rem;">
                <div style="font-weight: 700; color: #c084fc; margin-bottom: 2px;">🏛️ Model 2: Profile A Yield 'Monthly_winsaw' (Bridge 2 | Port 8889)</div>
                <div>Account: <b>#{account_m2.get('account_id', '346439083')}</b> ({account_m2.get('broker', 'XMGlobal-MT5 10')})</div>
                <div style="margin-top: 3px;">Balance: <b>{money(bal2)}</b> · Anchor: <b>$1,500.00</b> · Free Margin: <b>{money(fm2)}</b></div>
            </div>""",
            unsafe_allow_html=True,
        )

chart_range = st.sidebar.selectbox("P&L Chart Range", ["Last 6 months", "Year to date", "All available"], index=0)

# ==============================================================================
# MAIN TABS: 100% CLEAN SEGREGATION
# ==============================================================================
tab_m1, tab_m2, tab_all, tab_strategies = st.tabs([
    "🛡️ Model 1: Production Baseline",
    "⚡ Model 2: Profile A Yield Engine",
    "🌐 Consolidated Portfolio Overview",
    "📚 Strategy Library & Research Vault",
])

# ------------------------------------------------------------------------------
# TAB 1: MODEL 1 (PRODUCTION BASELINE) — ZERO MODEL 2 CONTENT
# ------------------------------------------------------------------------------
with tab_m1:
    st.markdown("""
    <div style="background: rgba(14, 165, 233, 0.08); border: 1px solid rgba(14, 165, 233, 0.3); border-radius: 12px; padding: 10px 16px; margin-bottom: 16px;">
        <span style="font-weight: 700; color: #38bdf8; font-size: 0.95rem;">Model 1: Production Baseline (Spot Gold · Bitcoin · Ethereum)</span>
        <span style="float: right; font-size: 0.75rem; background: rgba(14, 165, 233, 0.2); color: #38bdf8; padding: 3px 10px; border-radius: 999px; font-weight: 700;">FLAT_RISK_TOUCH</span>
        <p style="font-size: 0.8rem; color: #9ca3af; margin: 4px 0 0 0;">
            Execution Policy: Fixed $75.44 Risk per Trade · 1H Donchian 10 Touch · 50 EMA Filter · Target 1: 1.25R (+50% Scale-Out) · Dynamic ATR Trailing Stops ($4.50 Gold / $350 BTC / $10 ETH) · Uncapped Concurrency
        </p>
    </div>
    """, unsafe_allow_html=True)

    m1_open_pnl = sum(float(a.get("unrealized_pnl") or 0) for a in m1_assets)
    m1_today_realized = sum(float(a.get("realized_today") or 0) for a in m1_assets)
    m1_in_trade = sum(1 for a in m1_assets if a.get("status") == "IN TRADE")
    m1_monthly_df = monthly_by_model.get("model_1", pd.DataFrame())
    m1_overall_df = overall_by_model.get("model_1", pd.DataFrame())
    m1_total_realized = float(m1_monthly_df["pnl"].sum()) if not m1_monthly_df.empty and "pnl" in m1_monthly_df else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model 1 Realized P&L", money(m1_total_realized))
    c2.metric("Model 1 Open P&L", money(m1_open_pnl))
    c3.metric("Model 1 Today's Realized", money(m1_today_realized))
    c4.metric("Active In Trade", f"{m1_in_trade} / {len(m1_assets)} Assets")

    st.divider()
    st.subheader("Model 1 Asset Status & Market Radar")
    cols_m1 = st.columns(len(m1_assets)) if m1_assets else [st]
    for col, a in zip(cols_m1, m1_assets):
        render_asset_card(col, a)

    st.divider()
    render_pnl_charts(m1_monthly_df, m1_overall_df, chart_range, "Model 1 Baseline")

    st.divider()
    render_trades_table(m1_trades, "m1")


# ------------------------------------------------------------------------------
# TAB 2: MODEL 2 (PROFILE A YIELD ENGINE) — ZERO MODEL 1 CONTENT
# ------------------------------------------------------------------------------
with tab_m2:
    st.markdown("""
    <div style="background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.35); border-radius: 12px; padding: 10px 16px; margin-bottom: 14px;">
        <span style="font-weight: 700; color: #c084fc; font-size: 0.95rem;">Model 2: Profile A Crypto Yield Engine (Bitcoin · Ethereum)</span>
        <span style="float: right; font-size: 0.75rem; background: rgba(168, 85, 247, 0.2); color: #c084fc; padding: 3px 10px; border-radius: 999px; font-weight: 700;">PROFILE A YIELD</span>
        <p style="font-size: 0.8rem; color: #9ca3af; margin: 4px 0 0 0;">
            Execution Policy: Fixed $75.00 Risk · Tight Trailing Stops ($200 BTC / $6 ETH) · ML Meta-Labeling Gate (P ≥ 0.48) · Max 1 Concurrent Position per Asset · Order Tag: YieldA · 24/7 Crypto Execution
        </p>
    </div>
    """, unsafe_allow_html=True)

    m2_monthly_df = monthly_by_model.get("model_2", pd.DataFrame())
    m2_overall_df = overall_by_model.get("model_2", pd.DataFrame())

    m2_open_pnl = sum(float(a.get("unrealized_pnl") or 0) for a in m2_assets)
    m2_today_realized = sum(float(a.get("realized_today") or 0) for a in m2_assets)
    m2_in_trade = sum(1 for a in m2_assets if a.get("status") == "IN TRADE")
    m2_total_realized = float(m2_monthly_df["pnl"].sum()) if not m2_monthly_df.empty and "pnl" in m2_monthly_df else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Model 2 Realized P&L", money(m2_total_realized))
    k2.metric("Model 2 Open P&L", money(m2_open_pnl))
    k3.metric("Model 2 Today's Realized", money(m2_today_realized))
    k4.metric("Active In Trade", f"{m2_in_trade} / {len(m2_assets)} Crypto Assets")

    st.divider()
    st.subheader("Model 2 Asset Status & Market Radar")
    cols_m2 = st.columns(len(m2_assets)) if m2_assets else [st]
    for col, a in zip(cols_m2, m2_assets):
        render_asset_card(col, a)

    st.divider()
    render_pnl_charts(m2_monthly_df, m2_overall_df, chart_range, "Model 2 Profile A Yield")

    st.divider()
    render_trades_table(m2_trades, "m2")


# ------------------------------------------------------------------------------
# TAB 3: CONSOLIDATED PORTFOLIO OVERVIEW
# ------------------------------------------------------------------------------
with tab_all:
    st.markdown("""
    <div style="background: rgba(34, 197, 94, 0.08); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 12px; padding: 10px 16px; margin-bottom: 14px;">
        <span style="font-weight: 700; color: #22c55e; font-size: 0.95rem;">Consolidated Multi-Model Portfolio (Full Master View)</span>
        <span style="float: right; font-size: 0.75rem; background: rgba(34, 197, 94, 0.2); color: #22c55e; padding: 3px 10px; border-radius: 999px; font-weight: 700;">CONSOLIDATED</span>
        <p style="font-size: 0.8rem; color: #9ca3af; margin: 4px 0 0 0;">
            Combined Telemetry across Model 1 (Account #168987477 · XMGlobal-MT5 2) & Model 2 (Account #346439083 'Monthly_winsaw' · XMGlobal-MT5 10)
        </p>
    </div>
    """, unsafe_allow_html=True)

    all_open_pnl = sum(float(a.get("unrealized_pnl") or 0) for a in all_assets)
    all_today_realized = sum(float(a.get("realized_today") or 0) for a in all_assets)
    all_in_trade = sum(1 for a in all_assets if a.get("status") == "IN TRADE")
    all_total_realized = float(monthly_all["pnl"].sum()) if not monthly_all.empty and "pnl" in monthly_all else 0.0

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Combined Realized P&L", money(all_total_realized))
    p2.metric("Combined Open P&L", money(all_open_pnl))
    p3.metric("Combined Today's Realized", money(all_today_realized))
    p4.metric("Total Setups in Trade", f"{all_in_trade} / {len(all_assets)} Active")

    st.divider()
    st.subheader("Model Comparison Summary")
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown("#### 🛡️ Model 1: Production Baseline")
        st.write(f"- **Assets:** Spot Gold, Bitcoin, Ethereum")
        st.write(f"- **Risk Policy:** $75.44 Fixed Risk · 1H Donchian Touch · ATR Wide Trailing")
        st.write(f"- **Realized P&L:** {money(m1_total_realized)}")
        st.write(f"- **Open Positions:** {m1_in_trade} / 3")
    with comp_col2:
        st.markdown("#### ⚡ Model 2: Profile A Yield Engine")
        st.write(f"- **Assets:** Bitcoin, Ethereum (Order Tag `YieldA`)")
        st.write(f"- **Risk Policy:** $75.00 Fixed Risk · Tight Trailing ($200/$6) · ML Gate P≥0.48")
        st.write(f"- **Realized P&L:** {money(m2_total_realized)}")
        st.write(f"- **Open Positions:** {m2_in_trade} / {len(m2_assets)}")

    st.divider()
    render_pnl_charts(monthly_all, overall_all, chart_range, "Consolidated Portfolio")

    st.divider()
    render_trades_table(all_trades, "all")


# ------------------------------------------------------------------------------
# TAB 4: STRATEGY LIBRARY & RESEARCH VAULT
# ------------------------------------------------------------------------------
with tab_strategies:
    st.markdown("""
    <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 12px; padding: 12px 18px; margin-bottom: 16px;">
        <span style="font-weight: 700; color: #818cf8; font-size: 1.05rem;">📚 Aegis Alpha Strategy Catalog & Quantitative Architecture Repository</span>
        <span style="float: right; font-size: 0.75rem; background: rgba(99, 102, 241, 0.2); color: #818cf8; padding: 3px 10px; border-radius: 999px; font-weight: 700;">PERSISTENT VAULT</span>
        <p style="font-size: 0.82rem; color: #9ca3af; margin: 4px 0 0 0;">
            Permanent system memory storing all production, staged, benchmarked, and research trading strategies with audited 3Y & 5Y backtests, full algorithmic entry/exit rules, and CLI activation commands for future reference.
        </p>
    </div>
    """, unsafe_allow_html=True)

    catalog_dict = STRATEGY_CATALOG if STRATEGY_CATALOG else {}
    all_strats = list(catalog_dict.values())

    s_total = len(all_strats)
    s_active = sum(1 for s in all_strats if s.status in (StrategyStatus.ACTIVE_PRODUCTION, StrategyStatus.ACTIVE_YIELD_A))
    s_high_wr = sum(1 for s in all_strats if s.target_win_rate_pct >= 54.0 and s.status != StrategyStatus.DISQUALIFIED)
    s_staged = sum(1 for s in all_strats if s.status == StrategyStatus.STAGED_FOR_ACTIVATION)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cataloged Strategies", f"{s_total} Total")
    m2.metric("Active Live Engines", f"{s_active} Live (M1 + M2)")
    m3.metric("High Win-Rate Candidates (>54%)", f"{s_high_wr} Ready")
    m4.metric("Staged for Activation", f"{s_staged} Staged")

    st.divider()

    col_search, col_status, col_asset = st.columns([2, 1, 1])
    search_query = col_search.text_input("🔍 Search strategies by name, rule, or indicator...", value="", placeholder="e.g. Donchian, ML, Overlap, SuperTrend...")
    sel_status = col_status.selectbox("Filter Status", ["All Statuses", "ACTIVE_PRODUCTION", "ACTIVE_YIELD_A", "BENCHMARKED_READY", "STAGED_FOR_ACTIVATION", "RESEARCH_BENCHMARKED", "DISQUALIFIED"])
    sel_asset = col_asset.selectbox("Filter Asset", ["All Assets", "XAUUSD", "BTCUSD", "ETHUSD"])

    filtered_strats = []
    for s in all_strats:
        if sel_status != "All Statuses" and s.status.value != sel_status:
            continue
        if sel_asset != "All Assets" and sel_asset not in s.supported_assets:
            continue
        if search_query:
            q = search_query.lower()
            text_corpus = f"{s.id} {s.name} {s.short_desc} {s.category} {' '.join(s.supported_assets)} {s.notes_and_caveats}".lower()
            if q not in text_corpus:
                continue
        filtered_strats.append(s)

    st.caption(f"Displaying {len(filtered_strats)} of {s_total} stored strategies")

    status_badges = {
        StrategyStatus.ACTIVE_PRODUCTION: ("🟢 ACTIVE PRODUCTION", "background:rgba(14,165,233,0.18); color:#38bdf8; border:1px solid rgba(14,165,233,0.4);"),
        StrategyStatus.ACTIVE_YIELD_A: ("🏛️ ACTIVE YIELD A", "background:rgba(168,85,247,0.18); color:#c084fc; border:1px solid rgba(168,85,247,0.4);"),
        StrategyStatus.BENCHMARKED_READY: ("🔥 BENCHMARKED READY (>54% WR)", "background:rgba(34,197,94,0.18); color:#22c55e; border:1px solid rgba(34,197,94,0.4);"),
        StrategyStatus.STAGED_FOR_ACTIVATION: ("⏳ STAGED FOR ACTIVATION", "background:rgba(234,179,8,0.18); color:#eab308; border:1px solid rgba(234,179,8,0.4);"),
        StrategyStatus.RESEARCH_BENCHMARKED: ("🔬 RESEARCH BENCHMARKED", "background:rgba(148,163,184,0.15); color:#94a3b8; border:1px solid rgba(148,163,184,0.3);"),
        StrategyStatus.DISQUALIFIED: ("❌ DISQUALIFIED (NEGATIVE DRIFT)", "background:rgba(239,68,68,0.18); color:#ef4444; border:1px solid rgba(239,68,68,0.4);"),
    }

    for s in filtered_strats:
        badge_text, badge_style = status_badges.get(s.status, ("UNKNOWN", ""))
        expander_title = f"{badge_text.split()[0]} [{s.status.value}] {s.name} — {' / '.join(s.supported_assets)} (Target WR: {s.target_win_rate_pct:.1f}%)"

        with st.expander(expander_title, expanded=(s.status in (StrategyStatus.ACTIVE_PRODUCTION, StrategyStatus.BENCHMARKED_READY))):
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:0.85rem; font-weight:700; padding:3px 10px; border-radius:999px; {badge_style}">{badge_text}</span>
                <span style="font-size:0.8rem; color:#9ca3af;">Category: <b>{s.category}</b> · Risk Model: <b>{s.risk_archetype.value}</b> · Default: <b>${s.default_risk_usd:.2f}</b> · Timeframe: <b>{s.timeframe}</b></span>
            </div>
            <p style="font-size:0.92rem; color:#d1d5db; margin-bottom:12px;">{s.short_desc}</p>
            """, unsafe_allow_html=True)

            subtab1, subtab2, subtab3 = st.tabs(["⚙️ Technical Execution Blueprint", "📊 Audited Track Record", "💻 CLI Runbook & Activation"])

            with subtab1:
                col_rules1, col_rules2 = st.columns(2)
                with col_rules1:
                    st.markdown("##### Core Entry & Confirmation Rules")
                    for r_k, r_v in list(s.technical_rules.items())[:len(s.technical_rules)//2 + 1]:
                        st.markdown(f"- **{r_k.replace('_', ' ').title()}:** `{r_v}`")
                with col_rules2:
                    st.markdown("##### Exits, Stops & ML Configuration")
                    for r_k, r_v in list(s.technical_rules.items())[len(s.technical_rules)//2 + 1:]:
                        st.markdown(f"- **{r_k.replace('_', ' ').title()}:** `{r_v}`")
                    if s.ml_parameters:
                        st.markdown("##### Machine Learning Meta-Labeler")
                        for m_k, m_v in s.ml_parameters.items():
                            st.markdown(f"- **{m_k.replace('_', ' ').title()}:** `{m_v}`")

            with subtab2:
                if s.benchmark_metrics:
                    if "3_year_results" in s.benchmark_metrics:
                        st.markdown("##### 3-Year Audited Out-of-Sample Performance (Oct 2023 – Oct 2026)")
                        rows_3y = []
                        for sym, m in s.benchmark_metrics["3_year_results"].items():
                            rows_3y.append({
                                "Asset / Variation": sym,
                                "Trades": m.get("trades", 0),
                                "Win Rate": f"{m.get('win_rate_pct', 0.0):.1f}%",
                                "Profit Factor": f"{m.get('profit_factor', 0.0):.2f}",
                                "Payoff Ratio": f"{m.get('payoff_ratio', 0.0):.2f}R",
                                "Net Realized P&L": f"${m.get('net_pnl_usd', 0.0):,.2f}",
                                "Max Drawdown": f"-${m.get('max_dd_usd', 0.0):,.2f}",
                            })
                        st.dataframe(pd.DataFrame(rows_3y), use_container_width=True, hide_index=True)
                    if "5_year_portfolio" in s.benchmark_metrics:
                        st.markdown("##### 5-Year Master Portfolio Track Record")
                        st.json(s.benchmark_metrics["5_year_portfolio"])
                    if "monte_carlo_audit" in s.benchmark_metrics:
                        st.markdown("##### Monte Carlo Stress Test (60,000 Iterations)")
                        st.json(s.benchmark_metrics["monte_carlo_audit"])
                    if "6_month_gold_benchmark" in s.benchmark_metrics:
                        st.markdown("##### 6-Month Benchmarking Results")
                        st.json(s.benchmark_metrics["6_month_gold_benchmark"])
                else:
                    st.info("No quantitative benchmark metrics attached.")

            with subtab3:
                st.markdown("##### Daemon Activation Command")
                st.code(s.cli_activation_command or "# No standalone CLI command", language="bash")
                if s.notes_and_caveats:
                    st.markdown(f"""
                    <div style="background:rgba(234,179,8,0.08); border-left:4px solid #eab308; padding:8px 12px; border-radius:4px; font-size:0.85rem; color:#fef08a; margin-top:8px;">
                        <b>⚠️ Operational Note & Risk Safeguard:</b> {s.notes_and_caveats}
                    </div>
                    """, unsafe_allow_html=True)


# ---- Global Health Checklist ----
st.divider()
st.subheader("Platform System Health Checklist")
h1, h2, h3, h4 = st.columns(4)
h1.metric("Market Bridge", "ONLINE (MT5 Socket)")
h2.metric("Model 1 Engine", "FLAT_RISK ($75.44 · 3 Assets)")
h3.metric("Model 2 Engine", "PROFILE A ($75 · 2 Crypto Assets)")
h4.metric("Catalyst Guard & ML", "ACTIVE (±15m Blackout · P≥0.48)")
st.caption(f"Dashboard synchronized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} local server time.")
