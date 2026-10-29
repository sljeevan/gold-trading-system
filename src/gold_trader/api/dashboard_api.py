"""
FastAPI Backend & Standalone UI Server for Atlas Multi-Asset Trading Dashboard.
Supports Dual Production Models:
- Model 1: Production Baseline (Gold $75.44, BTC $75.44, ETH $75.44 | Wide Trailing | Uncapped Concurrency)
- Model 2: Profile A Crypto Yield Engine (BTC $75.00, ETH $75.00 | Tight Trailing | Max 1 Pos | $1,500 Treasury Harvest)
Provides:
- Web UI at "/" matching exact Atlas Trade design (Tailwind + Plotly.js + Multi-Model switcher)
- REST API at "/api/dashboard/overview"
- REST API at "/api/dashboard/treasury"
- Health at "/api/health"
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from gold_trader.brokers.socket_broker import SocketBroker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DashboardAPI")

app = FastAPI(title="Aegis Alpha Quantitative Trading System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.expanduser("~/.gold_trading_data")

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "model_1": {
        "id": "model_1",
        "name": "Model 1: Production Baseline",
        "short_name": "M1 Baseline",
        "badge_color": "sky",
        "bridge_port": 8888,
        "account_id": "168987477",
        "server": "XMGlobal-MT5 2",
        "account_label": "Model 1 Baseline (XMGlobal-MT5 2)",
        "order_tag": "Standard",
        "risk_per_trade": "$75.44 Fixed",
        "tp_policy": "1.25R (+50% Cash) · 3.0R TP2 Runner",
        "trailing_buffer": "$4.50 Gold / $350 BTC / $10 ETH",
        "concurrency": "Uncapped (1 Trade/Bar)",
        "dbs": {
            "XAUUSD": os.path.join(DATA_DIR, "auto_trader.db"),
            "BTCUSD": os.path.join(DATA_DIR, "btc_auto_trader.db"),
            "ETHUSD": os.path.join(DATA_DIR, "eth_auto_trader.db"),
        },
        "assets": [
            {
                "asset": "Gold",
                "symbol": "XAUUSD",
                "venue": "XM / MetaTrader 5",
                "unit": "lots",
                "price_format": 2,
                "default_mark": 4031.50,
                "model_id": "model_1",
                "model_name": "M1 Baseline",
                "risk_desc": "$75.44 Risk · Dynamic ATR Stop ($8 min) · $4.50 Trail",
            },
            {
                "asset": "Bitcoin",
                "symbol": "BTCUSD",
                "venue": "XM / MetaTrader 5",
                "unit": "BTC",
                "price_format": 2,
                "default_mark": 81850.0,
                "model_id": "model_1",
                "model_name": "M1 Baseline",
                "risk_desc": "$75.44 Risk · Dynamic ATR Stop ($500 min) · $350 Trail",
            },
            {
                "asset": "Ethereum",
                "symbol": "ETHUSD",
                "venue": "XM / MetaTrader 5",
                "unit": "ETH",
                "price_format": 2,
                "default_mark": 2404.0,
                "model_id": "model_1",
                "model_name": "M1 Baseline",
                "risk_desc": "$75.44 Risk · Dynamic ATR Stop ($20 min) · $10 Trail",
            },
        ],
    },
    "model_2": {
        "id": "model_2",
        "name": "Model 2: Profile A Crypto Yield Engine",
        "short_name": "M2 YieldA",
        "badge_color": "purple",
        "bridge_port": 8889,
        "account_id": "346439083",
        "server": "XMGlobal-MT5 10",
        "account_label": "Monthly_winsaw (XMGlobal-MT5 10)",
        "order_tag": "YieldA",
        "risk_per_trade": "$75.00 Fixed",
        "tp_policy": "1.25R (+50% Cash) · 3.0R TP2 Runner",
        "trailing_buffer": "$200 BTC / $6 ETH (Tight)",
        "concurrency": "Max 1 Position / Asset",
        "treasury_anchor": "$1,500.00 Monthly Watermark Harvest",
        "dbs": {
            "BTCUSD": os.path.join(DATA_DIR, "yield_a_btc_auto_trader.db"),
            "ETHUSD": os.path.join(DATA_DIR, "yield_a_eth_auto_trader.db"),
        },
        "assets": [
            {
                "asset": "Bitcoin (YieldA)",
                "symbol": "BTCUSD",
                "venue": "XM / MetaTrader 5",
                "unit": "BTC",
                "price_format": 2,
                "default_mark": 81850.0,
                "model_id": "model_2",
                "model_name": "M2 YieldA",
                "risk_desc": "$75.00 Risk · Tight Trail ($200) · ML P≥0.48 · Max 1 Pos",
            },
            {
                "asset": "Ethereum (YieldA)",
                "symbol": "ETHUSD",
                "venue": "XM / MetaTrader 5",
                "unit": "ETH",
                "price_format": 2,
                "default_mark": 2404.0,
                "model_id": "model_2",
                "model_name": "M2 YieldA",
                "risk_desc": "$75.00 Risk · Tight Trail ($6) · ML P≥0.48 · Max 1 Pos",
            },
        ],
    },
}

_CACHE: Dict[str, Any] = {
    "timestamp": 0.0,
    "payload": None,
    "ttl_seconds": 2.5,
    "ticks": {
        "XAUUSD": 4031.50,
        "BTCUSD": 81850.0,
        "ETHUSD": 2404.0,
    },
}

_broker_instances: Dict[int, Optional[SocketBroker]] = {}


def get_broker(port: int = 8888) -> Optional[SocketBroker]:
    global _broker_instances
    if port not in _broker_instances or _broker_instances[port] is None:
        try:
            b = SocketBroker(port=port)
            if b.connect():
                _broker_instances[port] = b
            else:
                _broker_instances[port] = None
        except Exception as e:
            logger.warning(f"MT5 broker bridge connect failed on port {port}: {e}")
            _broker_instances[port] = None
    return _broker_instances.get(port)


def ensure_sqlite_wal(db_path: str):
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path, timeout=1.0)
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            conn.commit()
            conn.close()
        except Exception:
            pass


def fetch_latest_scan_telemetry() -> Dict[str, Dict[str, Any]]:
    telemetry: Dict[str, Dict[str, Any]] = {"model_1": {}, "model_2": {}}
    for m_id, mcfg in MODEL_CONFIGS.items():
        for sym, path in mcfg["dbs"].items():
            if not os.path.exists(path):
                continue
            try:
                conn = sqlite3.connect(path, timeout=1.0)
                cur = conn.cursor()
                row = cur.execute(
                    "SELECT price, bias, decision, details_json, timestamp FROM scan_logs ORDER BY id DESC LIMIT 1"
                ).fetchone()
                conn.close()
                if row:
                    p, bias, decision, details_raw, ts = row
                    det = {}
                    if details_raw:
                        try:
                            det = json.loads(details_raw).get("setup_details", {})
                        except Exception:
                            pass
                    telemetry[m_id][sym] = {
                        "price": round(float(p), 2) if p else None,
                        "bias": bias or "NEUTRAL",
                        "decision": decision or "",
                        "atr": round(float(det.get("atr")), 2) if det.get("atr") else None,
                        "rsi": round(float(det.get("rsi")), 1) if det.get("rsi") else None,
                        "supertrend": det.get("supertrend_direction", 0),
                        "donchian_upper": round(float(det.get("donchian_upper")), 2) if det.get("donchian_upper") else None,
                        "donchian_lower": round(float(det.get("donchian_lower")), 2) if det.get("donchian_lower") else None,
                        "donchian_middle": round(float(det.get("donchian_middle")), 2) if det.get("donchian_middle") else None,
                        "reason": (det.get("reasons") or ["Scanning for 1H Touch"])[0],
                        "timestamp": ts[:19].replace("T", " ") if ts else "",
                    }
            except Exception as e:
                logger.warning(f"Error reading scan_logs for {m_id} {sym}: {e}")
    return telemetry


def fetch_historic_and_pnl_data(m2_account: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    monthly_pnl_all = defaultdict(float)
    monthly_pnl_by_model = {"model_1": defaultdict(float), "model_2": defaultdict(float)}

    daily_pnl_all = defaultdict(float)
    daily_pnl_by_model = {"model_1": defaultdict(float), "model_2": defaultdict(float)}

    today_realized_by_sym = defaultdict(float)
    today_realized_by_model_sym = defaultdict(float)
    all_trades: List[Dict[str, Any]] = []

    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for m_id, mcfg in MODEL_CONFIGS.items():
        for sym, path in mcfg["dbs"].items():
            if not os.path.exists(path):
                continue
            try:
                ensure_sqlite_wal(path)
                conn = sqlite3.connect(path, timeout=2.0)
                cur = conn.cursor()
                rows = cur.execute(
                    """
                    SELECT timestamp, ticket, symbol, action, lots, status, entry_price, close_price, pnl, comment, close_time, stop_loss, target_1, target_2, rr_ratio
                    FROM trade_executions
                    ORDER BY id ASC
                    """
                ).fetchall()
                conn.close()

                for r in rows:
                    ts, ticket, symbol, action, lots, status, entry, exit_p, pnl_val, comment, close_t, sl_val, tp1_val, tp2_val, rr_val = r
                    pnl = float(pnl_val or 0.0)
                    t_str = close_t or ts

                    if t_str and status == "CLOSED":
                        dt_str = t_str[:10]
                        m_str = t_str[:7] + "-01"
                        monthly_pnl_all[m_str] += pnl
                        monthly_pnl_by_model[m_id][m_str] += pnl

                        daily_pnl_all[dt_str] += pnl
                        daily_pnl_by_model[m_id][dt_str] += pnl

                        if dt_str == today_utc:
                            today_realized_by_sym[symbol] += pnl
                            today_realized_by_model_sym[(m_id, symbol)] += pnl

                    strat_desc = comment or "1H Donchian 10 Touch"
                    if m_id == "model_2":
                        strat_desc = f"Profile A Tight TSL ({comment or 'YieldA'})"
                    elif "FlatRisk" in (comment or ""):
                        strat_desc = "1H Donchian 10 Touch ($75.44)"

                    all_trades.append(
                        {
                            "time": ts[:19].replace("T", " ") if ts else "",
                            "ticket": str(ticket or ""),
                            "asset": symbol,
                            "side": action,
                            "qty": (
                                f"{lots:.2f} lots"
                                if "XAU" in symbol
                                else (f"{lots:.2f} BTC" if "BTC" in symbol else f"{lots:.2f} ETH")
                            ),
                            "status": status,
                            "entry": round(float(entry), 2) if entry else None,
                            "exit": round(float(exit_p), 2) if exit_p else None,
                            "sl": round(float(sl_val), 2) if sl_val else None,
                            "tp": round(float(tp1_val), 2) if tp1_val else None,
                            "tp2": round(float(tp2_val), 2) if tp2_val else None,
                            "rr": round(float(rr_val), 2) if rr_val else 1.25,
                            "pnl": round(pnl, 2),
                            "comment": comment or "",
                            "model_id": m_id,
                            "model_name": mcfg["short_name"],
                            "model_full": mcfg["name"],
                            "strategy": strat_desc,
                        }
                    )
            except Exception as exc:
                logger.warning(f"Error querying {m_id} {path}: {exc}")

    all_trades.sort(key=lambda x: x["time"], reverse=True)

    # Format monthly lists
    monthly_all_list = [{"month": m, "pnl": round(val, 2)} for m, val in sorted(monthly_pnl_all.items())]
    monthly_m1_list = [{"month": m, "pnl": round(val, 2)} for m, val in sorted(monthly_pnl_by_model["model_1"].items())]
    monthly_m2_list = [{"month": m, "pnl": round(val, 2)} for m, val in sorted(monthly_pnl_by_model["model_2"].items())]

    # Format cumulative lists
    def build_cum_list(d_dict: Dict[str, float]) -> List[Dict[str, Any]]:
        cum = 0.0
        out = []
        for d, val in sorted(d_dict.items()):
            cum += val
            out.append({"date": d, "cumulative_pnl": round(cum, 2)})
        return out

    overall_all_list = build_cum_list(daily_pnl_all)
    overall_m1_list = build_cum_list(daily_pnl_by_model["model_1"])
    overall_m2_list = build_cum_list(daily_pnl_by_model["model_2"])

    # Treasury Model 2 Watermark calculations
    current_month_prefix = datetime.now(timezone.utc).strftime("%Y-%m")
    m2_closed_trades = [t for t in all_trades if t["model_id"] == "model_2" and t["status"] == "CLOSED"]
    m2_month_closed = [t for t in m2_closed_trades if t["time"].startswith(current_month_prefix)]
    m2_cur_month_pnl = sum(t["pnl"] for t in m2_month_closed)
    m2_total_realized = sum(t["pnl"] for t in m2_closed_trades)

    anchor = 1500.0
    live_m2_equity = 0.0
    if m2_account and isinstance(m2_account, dict):
        live_m2_equity = float(m2_account.get("equity") or m2_account.get("balance") or 0.0)
    else:
        # Fallback to query port 8889 broker if available
        b2 = get_broker(8889)
        if b2:
            try:
                acc_info = b2.get_account_info()
                if acc_info and isinstance(acc_info, dict):
                    live_m2_equity = float(acc_info.get("equity") or acc_info.get("balance") or 0.0)
            except Exception:
                pass

    cur_equity = round(live_m2_equity if live_m2_equity > 0 else (anchor + m2_cur_month_pnl), 2)
    surplus = round(max(0.0, cur_equity - anchor), 2)
    deficit = round(max(0.0, anchor - cur_equity), 2)

    status_text = (
        f"READY TO HARVEST (+${surplus:,.2f})"
        if surplus > 0
        else (f"DEFICIT (Top-up Required: ${deficit:,.2f})" if deficit > 0 else f"BASELINE SECURED (${anchor:,.2f})")
    )

    m2_wins = [t for t in m2_closed_trades if t["pnl"] > 0]
    win_rate_m2 = round(len(m2_wins) / len(m2_closed_trades) * 100, 1) if m2_closed_trades else 0.0

    treasury_m2 = {
        "anchor_capital": anchor,
        "current_month": current_month_prefix,
        "current_month_pnl": round(m2_cur_month_pnl, 2),
        "current_month_equity": cur_equity,
        "harvestable_surplus": surplus,
        "deficit_top_up_needed": deficit,
        "status": status_text,
        "total_trades": len(m2_closed_trades),
        "win_rate": win_rate_m2,
        "total_realized": round(m2_total_realized, 2),
    }

    return {
        "monthly_all": monthly_all_list,
        "monthly_by_model": {
            "model_1": monthly_m1_list,
            "model_2": monthly_m2_list,
        },
        "overall_all": overall_all_list,
        "overall_by_model": {
            "model_1": overall_m1_list,
            "model_2": overall_m2_list,
        },
        "trades": all_trades,
        "today_realized_by_sym": today_realized_by_sym,
        "today_realized_by_model_sym": today_realized_by_model_sym,
        "treasury_model_2": treasury_m2,
    }


def fetch_live_asset_cards(
    today_realized_by_model_sym: Dict[Any, float],
    today_realized_by_sym: Dict[str, float],
) -> Dict[str, Any]:
    b1 = get_broker(8888)
    b2 = get_broker(8889)
    now_iso = datetime.now(timezone.utc).isoformat()
    telemetry = fetch_latest_scan_telemetry()

    open_positions_m1: List[Dict[str, Any]] = []
    if b1:
        try:
            open_positions_m1 = b1.get_open_positions()
        except Exception as e:
            logger.warning(f"Failed to fetch Bridge 1 live open positions: {e}")

    open_positions_m2: List[Dict[str, Any]] = []
    if b2:
        try:
            open_positions_m2 = b2.get_open_positions()
        except Exception as e:
            logger.warning(f"Failed to fetch Bridge 2 live open positions: {e}")

    open_by_model_sym: Dict[tuple, Dict[str, Any]] = {}
    tagged_positions: List[Dict[str, Any]] = []

    for p in open_positions_m1:
        sym = p.get("symbol", "").upper()
        if "GOLD" in sym or "XAU" in sym:
            sym = "XAUUSD"
        elif "BTC" in sym:
            sym = "BTCUSD"
        elif "ETH" in sym:
            sym = "ETHUSD"

        p_tagged = dict(p)
        p_tagged["model_id"] = "model_1"
        p_tagged["model_name"] = MODEL_CONFIGS["model_1"]["short_name"]
        p_tagged["symbol_norm"] = sym
        tagged_positions.append(p_tagged)
        open_by_model_sym[("model_1", sym)] = p_tagged

    for p in open_positions_m2:
        sym = p.get("symbol", "").upper()
        if "BTC" in sym:
            sym = "BTCUSD"
        elif "ETH" in sym:
            sym = "ETHUSD"

        p_tagged = dict(p)
        p_tagged["model_id"] = "model_2"
        p_tagged["model_name"] = MODEL_CONFIGS["model_2"]["short_name"]
        p_tagged["symbol_norm"] = sym
        tagged_positions.append(p_tagged)
        open_by_model_sym[("model_2", sym)] = p_tagged

    all_cards: List[Dict[str, Any]] = []
    cards_by_model: Dict[str, List[Dict[str, Any]]] = {"model_1": [], "model_2": []}

    for m_id, mcfg in MODEL_CONFIGS.items():
        broker_for_model = b1 if m_id == "model_1" else b2
        for item in mcfg["assets"]:
            sym = item["symbol"]
            mark_price = item["default_mark"]

            if broker_for_model:
                try:
                    info = broker_for_model.get_symbol_info(sym)
                    if info.get("status") == "SUCCESS":
                        bid = float(info.get("bid", 0.0))
                        if bid > 0:
                            mark_price = bid
                            _CACHE["ticks"][sym] = bid
                except Exception:
                    mark_price = _CACHE["ticks"].get(sym, mark_price)
            else:
                mark_price = _CACHE["ticks"].get(sym, mark_price)

            radar_info = telemetry.get(m_id, {}).get(
                sym,
                {
                    "price": round(mark_price, 2),
                    "bias": "NEUTRAL",
                    "decision": "",
                    "atr": None,
                    "rsi": None,
                    "supertrend": 0,
                    "donchian_upper": None,
                    "donchian_lower": None,
                    "donchian_middle": None,
                    "reason": f"Scanning for 1H Touch ({mcfg['short_name']})",
                    "timestamp": "",
                },
            )

            pos = open_by_model_sym.get((m_id, sym))
            today_realized = float(
                today_realized_by_model_sym.get((m_id, sym), 0.0)
                or (today_realized_by_sym.get(sym, 0.0) if m_id == "model_1" else 0.0)
            )

            if pos:
                card_data = {
                    "asset": item["asset"],
                    "symbol": sym,
                    "venue": item["venue"],
                    "model_id": m_id,
                    "model_name": mcfg["short_name"],
                    "model_full": mcfg["name"],
                    "badge_color": mcfg["badge_color"],
                    "risk_desc": item["risk_desc"],
                    "status": "IN TRADE",
                    "side": pos.get("side", "BUY"),
                    "entry": float(pos.get("open_price", mark_price)),
                    "mark": float(pos.get("current_price", mark_price)),
                    "quantity": float(pos.get("lots", 0.01)),
                    "unrealized_pnl": round(float(pos.get("pnl", 0.0)), 2),
                    "realized_today": round(today_realized, 2),
                    "stop_loss": float(pos.get("sl")) if pos.get("sl") else None,
                    "take_profit": float(pos.get("tp")) if pos.get("tp") else None,
                    "ticket": str(pos.get("ticket", "")),
                    "comment": pos.get("comment", ""),
                    "last_update": now_iso,
                    "price_format": item["price_format"],
                    "unit": item["unit"],
                    "radar": radar_info,
                }
            else:
                card_data = {
                    "asset": item["asset"],
                    "symbol": sym,
                    "venue": item["venue"],
                    "model_id": m_id,
                    "model_name": mcfg["short_name"],
                    "model_full": mcfg["name"],
                    "badge_color": mcfg["badge_color"],
                    "risk_desc": item["risk_desc"],
                    "status": "NO TRADE",
                    "side": None,
                    "entry": None,
                    "mark": round(mark_price, 2),
                    "quantity": 0.0,
                    "unrealized_pnl": 0.0,
                    "realized_today": round(today_realized, 2),
                    "stop_loss": None,
                    "take_profit": None,
                    "ticket": None,
                    "comment": None,
                    "last_update": now_iso,
                    "price_format": item["price_format"],
                    "unit": item["unit"],
                    "radar": radar_info,
                }

            all_cards.append(card_data)
            cards_by_model[m_id].append(card_data)

    return {
        "all_cards": all_cards,
        "cards_by_model": cards_by_model,
        "open_positions": tagged_positions,
    }


@app.get("/api/health")
def health():
    broker = get_broker()
    account_info = None
    if broker:
        try:
            account_info = broker.get_account_info()
        except Exception:
            pass
    return {
        "status": "ONLINE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "broker_connected": broker is not None,
        "account_info": account_info,
        "models_active": ["model_1", "model_2"],
    }


@app.get("/api/dashboard/treasury")
def get_treasury_status():
    history_data = fetch_historic_and_pnl_data()
    return history_data["treasury_model_2"]


@app.get("/api/dashboard/overview")
def get_dashboard_overview():
    now_ts = time.time()
    if _CACHE["payload"] and (now_ts - _CACHE["timestamp"] < _CACHE["ttl_seconds"]):
        return _CACHE["payload"]

    b1 = get_broker(8888)
    b2 = get_broker(8889)
    acc1 = None
    if b1:
        try:
            acc1 = b1.get_account_info()
            if acc1 and "broker" in acc1:
                acc1["account_name"] = "Model 1 Baseline"
                acc1["server_name"] = "XMGlobal-MT5 2"
                acc1["bridge_port"] = 8888
        except Exception as exc:
            logger.warning(f"Error fetching account 1 info: {exc}")

    acc2 = None
    if b2:
        try:
            acc2 = b2.get_account_info()
            if acc2 and "broker" in acc2:
                acc2["account_name"] = "Model 2 Monthly_winsaw"
                acc2["server_name"] = "XMGlobal-MT5 10"
                acc2["bridge_port"] = 8889
        except Exception as exc:
            logger.warning(f"Error fetching account 2 info: {exc}")

    history_data = fetch_historic_and_pnl_data(m2_account=acc2)
    asset_data = fetch_live_asset_cards(
        history_data["today_realized_by_model_sym"],
        history_data["today_realized_by_sym"],
    )

    payload = {
        "assets": asset_data["all_cards"],
        "assets_model_1": asset_data["cards_by_model"]["model_1"],
        "assets_model_2": asset_data["cards_by_model"]["model_2"],
        "open_positions": asset_data["open_positions"],
        "monthly": history_data["monthly_all"],
        "overall": history_data["overall_all"],
        "monthly_by_model": history_data["monthly_by_model"],
        "overall_by_model": history_data["overall_by_model"],
        "trades": history_data["trades"],
        "treasury_model_2": history_data["treasury_model_2"],
        "account": acc1,
        "account_model_1": acc1,
        "account_model_2": acc2,
        "accounts": {
            "model_1": acc1,
            "model_2": acc2,
        },
        "models": [
            {
                "id": "model_1",
                "name": "Model 1: Production Baseline",
                "short_name": "M1 Baseline",
                "account_id": "168987477",
                "account_name": "Production Baseline",
                "server": "XMGlobal-MT5 2",
                "bridge_port": 8888,
                "assets": ["XAUUSD", "BTCUSD", "ETHUSD"],
                "risk": "$75.44 Fixed",
                "trailing": "Baseline ($4.50 Gold / $350 BTC / $10 ETH)",
                "concurrency": "Uncapped",
                "order_tag": "Standard",
            },
            {
                "id": "model_2",
                "name": "Model 2: Profile A Yield Engine",
                "short_name": "M2 YieldA",
                "account_id": "346439083",
                "account_name": "Monthly_winsaw",
                "server": "XMGlobal-MT5 10",
                "bridge_port": 8889,
                "assets": ["BTCUSD", "ETHUSD"],
                "risk": "$75.00 Fixed",
                "trailing": "Tight ($200 BTC / $6 ETH)",
                "concurrency": "Max 1 Pos / Asset",
                "order_tag": "YieldA",
                "treasury_anchor": "$1,500.00 Watermark Harvest",
            },
        ],
    }

    _CACHE["payload"] = payload
    _CACHE["timestamp"] = now_ts
    return payload


# ---- Standalone UI Handler with Multi-Model Support & Treasury Widget ----
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Aegis Alpha | Multi-Model Quantitative Portfolio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { background-color: #0e1117; color: #e6e6e6; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .metric-card { background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.09); border-radius: 12px; padding: 16px; }
    .asset-card { background: rgba(255, 255, 255, 0.025); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 14px; padding: 18px; transition: border-color 0.2s ease; }
    .asset-card:hover { border-color: rgba(255, 255, 255, 0.18); }
    .status-pill { display: inline-block; border-radius: 999px; padding: 3px 10px; font-size: 0.75rem; font-weight: 700; background: rgba(22, 163, 74, 0.15); color: #22c55e; }
    .status-idle { background: rgba(148, 163, 184, 0.12); color: #94a3b8; }
    .pnl-pos { color: #22c55e; font-weight: 700; }
    .pnl-neg { color: #ef4444; font-weight: 700; }
    .donchian-track { position: relative; height: 6px; background: #1f2937; border-radius: 3px; }
    .donchian-pin { position: absolute; top: -4px; width: 14px; height: 14px; border-radius: 50%; transform: translateX(-50%); }
    .runner-flight-track { position: relative; height: 8px; background: #1f2937; border-radius: 4px; overflow: hidden; }
    .runner-flight-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
    .badge-m1 { background: rgba(14, 165, 233, 0.15); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.4); }
    .badge-m2 { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
  </style>
</head>
<body class="p-4 md:p-8 max-w-7xl mx-auto">

  <!-- Header -->
  <div class="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-gray-800 gap-4">
    <div>
      <h1 class="text-2xl md:text-3xl font-bold tracking-tight text-white flex items-center gap-2">
        <span>🛡️</span> Aegis Alpha | Quantitative Trading System
      </h1>
      <p class="text-sm text-gray-400 mt-1">Dual Production Engines · Spot Gold · Bitcoin · Ethereum | Autonomous Execution & ML Gate</p>
    </div>
    <div class="flex flex-wrap items-center gap-3">
      <button id="btn-audio-toggle" onclick="toggleAudioAlerts()" class="text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 px-3 py-1.5 rounded-full text-gray-300 flex items-center gap-1.5 transition-colors cursor-pointer">
        <span id="audio-icon">🔔</span> Alerts: <span id="audio-status" class="font-bold text-gray-400">OFF</span>
      </button>
      <span id="broker-status-badge" class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-emerald-950 text-emerald-400 border border-emerald-800">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> MT5 Demo #168987477 Connected
      </span>
      <span id="last-refresh" class="text-xs text-gray-500 font-mono">Live</span>
    </div>
  </div>

  <!-- Multi-Model Switcher Tabs -->
  <div class="flex flex-wrap items-center justify-between gap-3 my-4 p-2 bg-gray-900/80 border border-gray-800 rounded-xl">
    <div class="flex flex-wrap items-center gap-2">
      <span class="text-xs font-bold text-gray-400 uppercase tracking-wider px-2">Active Model:</span>
      <button onclick="setActiveModelView('model_1')" id="tab-m1" class="px-4 py-2 rounded-lg text-xs font-bold transition-all bg-sky-600 text-white shadow-sm border border-sky-500">
        🛡️ Model 1: Production Baseline (Gold · BTC · ETH)
      </button>
      <button onclick="setActiveModelView('model_2')" id="tab-m2" class="px-4 py-2 rounded-lg text-xs font-bold transition-all bg-gray-800 text-gray-300 hover:text-white border border-gray-700">
        ⚡ Model 2: Profile A Yield (BTC · ETH · Harvest)
      </button>
      <button onclick="setActiveModelView('ALL')" id="tab-all" class="px-4 py-2 rounded-lg text-xs font-bold transition-all bg-gray-800 text-gray-300 hover:text-white border border-gray-700">
        🌐 Consolidated Portfolio Overview
      </button>
    </div>
    <div class="text-xs text-gray-400 flex items-center gap-2 px-2">
      <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
      <span id="model-status-indicator">Both Engines Active on MT5</span>
    </div>
  </div>

  <!-- Model 2 Treasury Harvest Widget (Hidden when Model 1 is active) -->
  <div id="treasury-card" class="bg-gradient-to-r from-purple-950/40 via-gray-900/90 to-purple-950/40 border border-purple-500/30 rounded-xl p-4 my-4 shadow-lg" style="display: none;">
    <div class="flex flex-col md:flex-row md:items-center justify-between pb-3 border-b border-purple-500/20 gap-3">
      <div class="flex items-center gap-2.5">
        <span class="px-2.5 py-1 rounded-full text-xs font-bold bg-purple-900/60 text-purple-300 border border-purple-600/50 flex items-center gap-1.5">
          <span>🏛️</span> MODEL 2 TREASURY HARVEST
        </span>
        <span class="text-xs text-gray-300">Monthly Watermark Rule: Harvest 100% of profits above Anchor Seed; replenish if deficit.</span>
      </div>
      <div id="treasury-status-badge" class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-purple-950 text-purple-300 border border-purple-800">
        BASELINE SECURED ($1,500.00)
      </div>
    </div>
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-3 text-center">
      <div class="bg-gray-900/60 p-2.5 rounded-lg border border-gray-800">
        <div class="text-[11px] text-gray-400 uppercase tracking-wider">Anchor Capital Seed</div>
        <div class="flex items-center justify-center gap-1 mt-1">
          <span class="text-gray-400 text-sm">$</span>
          <input type="number" id="input-anchor-seed" value="1500" step="50" min="100" 
                 class="bg-gray-800 text-white font-bold text-lg font-mono rounded px-2 py-0.5 w-28 text-center border border-gray-700 focus:border-purple-500 focus:outline-none"
                 oninput="onAnchorSeedChange(this.value)" />
        </div>
        <div class="text-[10px] text-purple-400 mt-0.5">Editable Seed Input</div>
      </div>
      <div class="bg-gray-900/60 p-2.5 rounded-lg border border-gray-800">
        <div class="text-[11px] text-gray-400 uppercase tracking-wider">Current Month P&L</div>
        <div id="treasury-month-pnl" class="text-xl font-bold text-emerald-400 font-mono mt-0.5">+$0.00</div>
        <div class="text-[10px] text-gray-500">October 2026 Engine P&L</div>
      </div>
      <div class="bg-purple-950/40 p-2.5 rounded-lg border border-purple-500/30">
        <div class="text-[11px] text-purple-300 uppercase tracking-wider font-semibold">Ready to Harvest Cash</div>
        <div id="treasury-surplus" class="text-xl font-bold text-emerald-400 font-mono mt-0.5">$0.00</div>
        <div class="text-[10px] text-purple-400/80">Surplus above Anchor Seed</div>
      </div>
      <div class="bg-gray-900/60 p-2.5 rounded-lg border border-gray-800">
        <div class="text-[11px] text-gray-400 uppercase tracking-wider">Deficit Top-Up Needed</div>
        <div id="treasury-deficit" class="text-xl font-bold text-gray-300 font-mono mt-0.5">$0.00</div>
        <div class="text-[10px] text-gray-500">Replenishment required</div>
      </div>
    </div>
  </div>

  <!-- Master KPI Metrics -->
  <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 my-4">
    <div class="metric-card border border-emerald-500/30">
      <div id="label-account-balance" class="text-xs font-medium text-emerald-400 uppercase tracking-wider">Account Balance</div>
      <div id="metric-account-balance" class="text-2xl font-bold text-white mt-1">$0.00</div>
    </div>
    <div class="metric-card border border-emerald-500/30">
      <div id="label-account-equity" class="text-xs font-medium text-emerald-400 uppercase tracking-wider">Account Equity</div>
      <div id="metric-account-equity" class="text-2xl font-bold text-white mt-1">$0.00</div>
    </div>
    <div class="metric-card">
      <div class="text-xs font-medium text-gray-400 uppercase tracking-wider">Today's Realized P&L</div>
      <div id="metric-today-pnl" class="text-2xl font-bold text-white mt-1">$0.00</div>
    </div>
    <div class="metric-card">
      <div class="text-xs font-medium text-gray-400 uppercase tracking-wider">Open / Unrealized P&L</div>
      <div id="metric-open-pnl" class="text-2xl font-bold text-white mt-1">$0.00</div>
    </div>
    <div class="metric-card">
      <div class="text-xs font-medium text-gray-400 uppercase tracking-wider">Realized P&L · All History</div>
      <div id="metric-total-pnl" class="text-2xl font-bold text-white mt-1">$0.00</div>
    </div>
    <div class="metric-card">
      <div class="text-xs font-medium text-gray-400 uppercase tracking-wider">Active In Trade</div>
      <div id="metric-active-count" class="text-2xl font-bold text-white mt-1">0 Positions</div>
    </div>
  </div>

  <!-- Quantitative Performance Ribbon -->
  <div class="metric-card my-4 py-3 bg-gradient-to-r from-gray-900/80 via-gray-900/40 to-gray-900/80">
    <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 text-center divide-y sm:divide-y-0 sm:divide-x divide-gray-800/80">
      <div class="pt-2 sm:pt-0">
        <div class="text-[11px] font-medium text-gray-400 uppercase tracking-wider">Win Rate</div>
        <div id="stat-win-rate" class="text-xl font-bold text-emerald-400 mt-0.5">0.0%</div>
        <div id="stat-win-loss-count" class="text-[10px] text-gray-500 font-mono">0W / 0L (0 Trades)</div>
      </div>
      <div class="pt-2 sm:pt-0">
        <div class="text-[11px] font-medium text-gray-400 uppercase tracking-wider">Profit Factor</div>
        <div id="stat-profit-factor" class="text-xl font-bold text-emerald-400 mt-0.5">0.00</div>
        <div class="text-[10px] text-gray-500">Gross Win / Gross Loss</div>
      </div>
      <div class="pt-2 sm:pt-0">
        <div class="text-[11px] font-medium text-gray-400 uppercase tracking-wider">Avg Win</div>
        <div id="stat-avg-win" class="text-xl font-bold text-emerald-400 mt-0.5">+$0.00</div>
        <div class="text-[10px] text-gray-500">Per winning execution</div>
      </div>
      <div class="pt-2 sm:pt-0">
        <div class="text-[11px] font-medium text-gray-400 uppercase tracking-wider">Avg Loss</div>
        <div id="stat-avg-loss" class="text-xl font-bold text-rose-400 mt-0.5">-$0.00</div>
        <div class="text-[10px] text-gray-500">Per losing execution</div>
      </div>
      <div class="pt-2 sm:pt-0">
        <div class="text-[11px] font-medium text-gray-400 uppercase tracking-wider">Expectancy</div>
        <div id="stat-expectancy" class="text-xl font-bold text-emerald-400 mt-0.5">+$0.00</div>
        <div class="text-[10px] text-gray-500">Average P&L per trade</div>
      </div>
      <div class="pt-2 sm:pt-0">
        <div class="text-[11px] font-medium text-gray-400 uppercase tracking-wider">Max Drawdown</div>
        <div id="stat-max-dd" class="text-xl font-bold text-rose-400 mt-0.5">-$0.00</div>
        <div class="text-[10px] text-gray-500">Peak-to-trough in cycle</div>
      </div>
    </div>
  </div>

  <!-- Asset Cards Section -->
  <div class="flex items-center justify-between mb-3 mt-8">
    <h2 class="text-lg font-semibold text-gray-200 flex items-center gap-2">
      <span>💎</span> Active Asset Status & Bot Market Radars
    </h2>
    <span id="asset-card-count" class="text-xs text-gray-400 font-mono">Showing 5 Setups across 2 Models</span>
  </div>
  <div id="asset-cards-grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 mb-8">
    <!-- Populated dynamically via JS -->
  </div>

  <!-- Charts Section -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 my-8">
    <div class="metric-card">
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-base font-semibold text-gray-200">Month-on-Month Realized P&L</h3>
        <span id="chart-monthly-tag" class="text-xs text-gray-400 font-mono">All Models</span>
      </div>
      <div id="chart-monthly" style="height: 320px;"></div>
    </div>
    <div class="metric-card">
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-base font-semibold text-gray-200">Overall Cumulative Realized P&L</h3>
        <span id="chart-overall-tag" class="text-xs text-gray-400 font-mono">All Models</span>
      </div>
      <div id="chart-overall" style="height: 320px;"></div>
    </div>
  </div>

  <!-- Trade Activity Table -->
  <div class="metric-card my-8">
    <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-gray-800 gap-3">
      <div>
        <h3 class="text-base font-semibold text-gray-200">Trade Activity (Fills & Executions)</h3>
        <p class="text-[11px] text-gray-500 mt-0.5">Click column headers to sort · Click any row to expand ticket, model & SL/TP details</p>
      </div>
      <div class="flex flex-wrap items-center gap-2.5">
        <select id="model-filter" onchange="onFilterChange()" class="bg-gray-800 border border-gray-700 text-xs rounded-lg px-2.5 py-1.5 text-gray-300 focus:outline-none focus:border-emerald-500 cursor-pointer">
          <option value="ALL">All Models</option>
          <option value="model_1">Model 1 (Baseline)</option>
          <option value="model_2">Model 2 (Profile A Yield)</option>
        </select>
        <select id="asset-filter" onchange="onFilterChange()" class="bg-gray-800 border border-gray-700 text-xs rounded-lg px-2.5 py-1.5 text-gray-300 focus:outline-none focus:border-emerald-500 cursor-pointer">
          <option value="ALL">All Assets</option>
          <option value="XAUUSD">Gold (XAUUSD)</option>
          <option value="BTCUSD">Bitcoin (BTCUSD)</option>
          <option value="ETHUSD">Ethereum (ETHUSD)</option>
        </select>
        <select id="side-filter" onchange="onFilterChange()" class="bg-gray-800 border border-gray-700 text-xs rounded-lg px-2.5 py-1.5 text-gray-300 focus:outline-none focus:border-emerald-500 cursor-pointer">
          <option value="ALL">All Sides</option>
          <option value="BUY">BUY Only</option>
          <option value="SELL">SELL Only</option>
        </select>
        <button onclick="toggleClosedTrades()" id="btn-toggle-closed" class="text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 px-3 py-1.5 rounded-lg text-gray-300 transition-colors">
          Hide Closed
        </button>
        <button onclick="exportTradesCSV()" class="text-xs bg-emerald-950 hover:bg-emerald-900 border border-emerald-800 text-emerald-400 px-3 py-1.5 rounded-lg flex items-center gap-1 font-semibold transition-colors cursor-pointer">
          <span>📥</span> Export CSV
        </button>
      </div>
    </div>
    <div class="overflow-x-auto mt-4">
      <table class="w-full text-left text-xs">
        <thead class="text-gray-400 uppercase bg-gray-900/50 select-none">
          <tr>
            <th class="py-2.5 px-3 cursor-pointer hover:text-white transition-colors" onclick="sortTable('time')">Time (UTC) <span id="sort-time" class="text-gray-500">↕</span></th>
            <th class="py-2.5 px-3">Model</th>
            <th class="py-2.5 px-3 cursor-pointer hover:text-white transition-colors" onclick="sortTable('asset')">Asset <span id="sort-asset" class="text-gray-500">↕</span></th>
            <th class="py-2.5 px-3 cursor-pointer hover:text-white transition-colors" onclick="sortTable('side')">Side <span id="sort-side" class="text-gray-500">↕</span></th>
            <th class="py-2.5 px-3 cursor-pointer hover:text-white transition-colors" onclick="sortTable('qty')">Quantity <span id="sort-qty" class="text-gray-500">↕</span></th>
            <th class="py-2.5 px-3 cursor-pointer hover:text-white transition-colors" onclick="sortTable('status')">Status <span id="sort-status" class="text-gray-500">↕</span></th>
            <th class="py-2.5 px-3">Entry</th>
            <th class="py-2.5 px-3">Exit</th>
            <th class="py-2.5 px-3 cursor-pointer hover:text-white transition-colors" onclick="sortTable('pnl')">P&L ($) <span id="sort-pnl" class="text-gray-500">↕</span></th>
            <th class="py-2.5 px-3">Strategy / Tag</th>
          </tr>
        </thead>
        <tbody id="trades-tbody" class="divide-y divide-gray-800/60 font-mono">
          <!-- Populated dynamically via JS -->
        </tbody>
      </table>
    </div>
    <!-- Pagination Footer -->
    <div id="trades-pagination-bar" class="flex flex-col sm:flex-row items-center justify-between pt-4 mt-3 border-t border-gray-800/80 gap-3 text-xs text-gray-400">
      <div class="flex items-center gap-3">
        <span id="trades-range-info" class="font-medium text-gray-300">Showing 0 trades</span>
        <div class="flex items-center gap-1.5 ml-2">
          <span class="text-gray-500">Per page:</span>
          <select id="trades-page-size" onchange="onPageSizeChange(this.value)" class="bg-gray-800 border border-gray-700 text-xs rounded px-2 py-1 text-gray-300 focus:outline-none focus:border-emerald-500 cursor-pointer">
            <option value="5">5</option>
            <option value="10" selected>10</option>
            <option value="20">20</option>
            <option value="50">50</option>
          </select>
        </div>
      </div>
      <div id="trades-pagination-nav" class="flex items-center gap-1">
        <!-- Buttons injected dynamically -->
      </div>
    </div>
  </div>

  <!-- System Health Checklist -->
  <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-4 border-t border-gray-800 text-xs">
    <div class="metric-card text-center">
      <div class="text-gray-400">Market Bridge</div>
      <div class="font-bold text-emerald-400 mt-1">ONLINE (MT5 Socket)</div>
    </div>
    <div class="metric-card text-center">
      <div class="text-gray-400">Model 1 Engine</div>
      <div class="font-bold text-sky-400 mt-1">FLAT_RISK ($75.44)</div>
    </div>
    <div class="metric-card text-center">
      <div class="text-gray-400">Model 2 Engine</div>
      <div class="font-bold text-purple-400 mt-1">PROFILE A ($75 · $1.5K)</div>
    </div>
    <div class="metric-card text-center">
      <div class="text-gray-400">Catalyst Guard & ML</div>
      <div class="font-bold text-emerald-400 mt-1">ACTIVE (±15m · P≥0.48)</div>
    </div>
  </div>

  <script>
    let globalData = null;
    let activeModelView = "model_1"; // Default to Model 1 tab
    let customAnchorSeed = parseFloat(localStorage.getItem("aegis_anchor_seed") || "1500");
    let showClosed = true;
    let tradesCurrentPage = 1;
    let tradesPageSize = 10;
    let sortColumn = "time";
    let sortAsc = false;
    let expandedTradeIndex = null;
    let audioAlertsEnabled = localStorage.getItem("aegis_audio_alerts") === "true";
    let previousTradeCount = null;

    function initAudioUI() {
      const statusEl = document.getElementById("audio-status");
      const iconEl = document.getElementById("audio-icon");
      if (statusEl) {
        statusEl.innerText = audioAlertsEnabled ? "ON" : "OFF";
        statusEl.className = audioAlertsEnabled ? "font-bold text-emerald-400" : "font-bold text-gray-400";
      }
      if (iconEl) {
        iconEl.innerText = audioAlertsEnabled ? "🔔" : "🔕";
      }
    }

    function toggleAudioAlerts() {
      audioAlertsEnabled = !audioAlertsEnabled;
      localStorage.setItem("aegis_audio_alerts", audioAlertsEnabled);
      initAudioUI();
      if (audioAlertsEnabled) {
        playChime();
      }
    }

    function onAnchorSeedChange(val) {
      const num = parseFloat(val);
      if (!isNaN(num) && num > 0) {
        customAnchorSeed = num;
        localStorage.setItem("aegis_anchor_seed", num.toString());
        if (globalData && globalData.treasury_model_2) {
          renderTreasuryWidget(globalData.treasury_model_2);
        }
      }
    }

    function playChime() {
      if (!audioAlertsEnabled) return;
      try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) return;
        const ctx = new AudioContext();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        const now = ctx.currentTime;
        osc.type = "sine";
        osc.frequency.setValueAtTime(587.33, now); // D5
        osc.frequency.setValueAtTime(880.00, now + 0.08); // A5
        gain.gain.setValueAtTime(0.12, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
        osc.start(now);
        osc.stop(now + 0.35);
      } catch (e) {
        console.warn("Audio chime error:", e);
      }
    }

    function formatMoney(num) {
      if (num === null || num === undefined) return "—";
      const val = parseFloat(num);
      return (val < 0 ? "-" : "") + "$" + Math.abs(val).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    function setActiveModelView(m) {
      activeModelView = m;
      const isM1 = m === "model_1";
      const isM2 = m === "model_2";
      const isAll = m === "ALL";

      const btnM1 = document.getElementById("tab-m1");
      const btnM2 = document.getElementById("tab-m2");
      const btnAll = document.getElementById("tab-all");

      if (btnM1) btnM1.className = isM1 ? "px-4 py-2 rounded-lg text-xs font-bold transition-all bg-sky-600 text-white shadow-sm border border-sky-500" : "px-4 py-2 rounded-lg text-xs font-bold transition-all bg-gray-800 text-gray-300 hover:text-white border border-gray-700";
      if (btnM2) btnM2.className = isM2 ? "px-4 py-2 rounded-lg text-xs font-bold transition-all bg-purple-600 text-white shadow-sm border border-purple-500" : "px-4 py-2 rounded-lg text-xs font-bold transition-all bg-gray-800 text-gray-300 hover:text-white border border-gray-700";
      if (btnAll) btnAll.className = isAll ? "px-4 py-2 rounded-lg text-xs font-bold transition-all bg-emerald-600 text-white shadow-sm border border-emerald-500" : "px-4 py-2 rounded-lg text-xs font-bold transition-all bg-gray-800 text-gray-300 hover:text-white border border-gray-700";

      // Synchronize trade filter dropdown
      const modelDropdown = document.getElementById("model-filter");
      if (modelDropdown) {
        modelDropdown.value = m;
      }

      tradesCurrentPage = 1;
      expandedTradeIndex = null;
      if (globalData) {
        updateView();
      }
    }

    function onFilterChange() {
      const modelDropdown = document.getElementById("model-filter");
      if (modelDropdown && modelDropdown.value !== activeModelView) {
        activeModelView = modelDropdown.value;
        setActiveModelView(activeModelView);
        return;
      }
      tradesCurrentPage = 1;
      expandedTradeIndex = null;
      renderTrades();
    }

    function toggleClosedTrades() {
      showClosed = !showClosed;
      document.getElementById('btn-toggle-closed').innerText = showClosed ? "Hide Closed" : "Show All";
      tradesCurrentPage = 1;
      expandedTradeIndex = null;
      renderTrades();
    }

    function onPageSizeChange(val) {
      tradesPageSize = parseInt(val, 10) || 10;
      tradesCurrentPage = 1;
      renderTrades();
    }

    function goToTradesPage(page) {
      tradesCurrentPage = page;
      renderTrades();
    }

    function sortTable(col) {
      if (sortColumn === col) {
        sortAsc = !sortAsc;
      } else {
        sortColumn = col;
        sortAsc = col === "asset" || col === "side" || col === "status" || col === "model_id";
      }
      tradesCurrentPage = 1;
      renderTrades();
    }

    function toggleTradeRow(idx) {
      expandedTradeIndex = (expandedTradeIndex === idx) ? null : idx;
      renderTrades();
    }

    function exportTradesCSV() {
      if (!globalData || !globalData.trades || globalData.trades.length === 0) {
        alert("No trades available to export.");
        return;
      }
      const modelFilter = document.getElementById("model-filter").value;
      const assetFilter = document.getElementById("asset-filter").value;
      const sideFilter = document.getElementById("side-filter").value;
      let list = globalData.trades;
      if (modelFilter !== "ALL") list = list.filter(t => t.model_id === modelFilter);
      if (assetFilter !== "ALL") list = list.filter(t => t.asset && t.asset.toUpperCase() === assetFilter.toUpperCase());
      if (sideFilter !== "ALL") list = list.filter(t => t.side && t.side.toUpperCase() === sideFilter.toUpperCase());
      if (!showClosed) list = list.filter(t => t.status === "OPEN" || t.status === "ACTIVE");

      const headers = ["Time (UTC)", "Model", "Ticket", "Asset", "Side", "Quantity", "Status", "Entry Price", "Exit Price", "Stop Loss", "Target 1", "Realized P&L ($)", "Strategy"];
      const rows = list.map(t => [
        `"${t.time}"`,
        `"${t.model_name}"`,
        `"${t.ticket || ''}"`,
        `"${t.asset}"`,
        `"${t.side}"`,
        `"${t.qty}"`,
        `"${t.status}"`,
        t.entry ?? "",
        t.exit ?? "",
        t.sl ?? "",
        t.tp ?? "",
        t.pnl ?? 0,
        `"${t.strategy}"`
      ]);

      const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\\n");
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      const dateStr = new Date().toISOString().slice(0, 10);
      link.setAttribute("href", url);
      link.setAttribute("download", `aegis_alpha_trades_${dateStr}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }

    function computePerformanceKPIs(trades) {
      let filtered = trades || [];
      if (activeModelView !== "ALL") {
        filtered = filtered.filter(t => t.model_id === activeModelView);
      }
      const closed = filtered.filter(t => t.status === "CLOSED");
      if (closed.length === 0) {
        document.getElementById("stat-win-rate").innerText = "0.0%";
        document.getElementById("stat-win-loss-count").innerText = "0W / 0L (0 Trades)";
        document.getElementById("stat-profit-factor").innerText = "0.00";
        document.getElementById("stat-avg-win").innerText = "+$0.00";
        document.getElementById("stat-avg-loss").innerText = "-$0.00";
        document.getElementById("stat-expectancy").innerText = "+$0.00";
        document.getElementById("stat-max-dd").innerText = "-$0.00";
        return;
      }

      const pnls = closed.map(t => parseFloat(t.pnl || 0));
      const wins = pnls.filter(p => p > 0);
      const losses = pnls.filter(p => p < 0);
      const grossWin = wins.reduce((s, w) => s + w, 0);
      const grossLoss = Math.abs(losses.reduce((s, l) => s + l, 0));
      const winRate = ((wins.length / closed.length) * 100).toFixed(1);
      const pf = grossLoss > 0 ? (grossWin / grossLoss).toFixed(2) : (grossWin > 0 ? "∞" : "0.00");
      const avgWin = wins.length > 0 ? (grossWin / wins.length).toFixed(2) : "0.00";
      const avgLoss = losses.length > 0 ? (grossLoss / losses.length).toFixed(2) : "0.00";
      const expectancy = ((grossWin - grossLoss) / closed.length).toFixed(2);

      let peak = 0;
      let cum = 0;
      let maxDD = 0;
      const chron = [...closed].sort((a, b) => (a.time || '').localeCompare(b.time || ''));
      for (const t of chron) {
        cum += parseFloat(t.pnl || 0);
        if (cum > peak) peak = cum;
        const dd = peak - cum;
        if (dd > maxDD) maxDD = dd;
      }

      document.getElementById("stat-win-rate").innerText = `${winRate}%`;
      document.getElementById("stat-win-loss-count").innerText = `${wins.length}W / ${losses.length}L (${closed.length} Trades)`;
      document.getElementById("stat-profit-factor").innerText = pf;
      document.getElementById("stat-avg-win").innerText = `+$${avgWin}`;
      document.getElementById("stat-avg-loss").innerText = `-$${avgLoss}`;
      document.getElementById("stat-expectancy").innerText = `${parseFloat(expectancy) >= 0 ? '+' : ''}$${expectancy}`;
      document.getElementById("stat-expectancy").className = `text-xl font-bold mt-0.5 ${parseFloat(expectancy) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`;
      document.getElementById("stat-max-dd").innerText = `-$${maxDD.toFixed(2)}`;
    }

    function renderTreasuryWidget(treasury) {
      if (!treasury) return;
      const anchorInput = document.getElementById("input-anchor-seed");
      if (anchorInput && document.activeElement !== anchorInput) {
        anchorInput.value = customAnchorSeed;
      }
      
      const anchor = customAnchorSeed || parseFloat(treasury.anchor_capital || 1500.0);
      const equity = parseFloat(treasury.current_month_equity || 0);
      const monthPnl = parseFloat(treasury.current_month_pnl || 0);

      // Determine effective equity vs anchor seed
      const effectiveEquity = equity > 0 ? equity : (anchor + monthPnl);
      const surplus = Math.max(0, Math.round((effectiveEquity - anchor) * 100) / 100);
      const deficit = Math.max(0, Math.round((anchor - effectiveEquity) * 100) / 100);

      document.getElementById("treasury-month-pnl").innerText = `${monthPnl >= 0 ? '+' : ''}${formatMoney(monthPnl)}`;
      document.getElementById("treasury-month-pnl").className = `text-xl font-bold font-mono mt-0.5 ${monthPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`;
      document.getElementById("treasury-surplus").innerText = formatMoney(surplus);
      document.getElementById("treasury-deficit").innerText = formatMoney(deficit);

      const badge = document.getElementById("treasury-status-badge");
      if (badge) {
        if (surplus > 0) {
          badge.className = "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-950 text-emerald-400 border border-emerald-800 animate-pulse";
          badge.innerText = `READY TO HARVEST (+${formatMoney(surplus)})`;
        } else if (deficit > 0) {
          badge.className = "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-rose-950 text-rose-400 border border-rose-800";
          badge.innerText = `DEFICIT (TOP-UP ${formatMoney(deficit)})`;
        } else {
          badge.className = "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-purple-950 text-purple-300 border border-purple-800";
          badge.innerText = `BASELINE SECURED (${formatMoney(anchor)})`;
        }
      }
    }

    function renderAssetCards(assets) {
      const container = document.getElementById("asset-cards-grid");
      container.innerHTML = "";

      let filtered = assets || [];
      if (activeModelView !== "ALL") {
        filtered = filtered.filter(a => a.model_id === activeModelView);
      }

      document.getElementById("asset-card-count").innerText = `Showing ${filtered.length} Setups (${activeModelView === 'ALL' ? 'Both Models' : (activeModelView === 'model_1' ? 'Model 1 Baseline' : 'Model 2 Profile A')})`;

      filtered.forEach(a => {
        const inTrade = a.status === "IN TRADE";
        const pillClass = inTrade ? "status-pill" : "status-pill status-idle";
        const pillText = inTrade ? "🟢 IN TRADE" : "⚪ SCANNING";
        const pnl = parseFloat(a.unrealized_pnl || 0);
        const pnlClass = pnl >= 0 ? "pnl-pos" : "pnl-neg";
        const radar = a.radar || {};
        const isM1 = a.model_id === "model_1";
        const modelBadgeClass = isM1 ? "badge-m1" : "badge-m2";

        let detailsHtml = "";
        if (inTrade) {
          const entry = parseFloat(a.entry || a.mark);
          const mark = parseFloat(a.mark || entry);
          const sl = parseFloat(a.stop_loss || (entry - 10));
          const tp1 = parseFloat(a.take_profit || (entry + 12.5));
          const isBuy = (a.side || "BUY") === "BUY";
          
          let flightPct = 0;
          if (isBuy) {
            flightPct = tp1 > entry ? ((mark - entry) / (tp1 - entry)) * 100 : 50;
          } else {
            flightPct = tp1 < entry ? ((entry - mark) / (entry - tp1)) * 100 : 50;
          }
          flightPct = Math.max(0, Math.min(100, Math.round(flightPct)));
          const barColor = pnl >= 0 ? "bg-emerald-500" : "bg-rose-500";

          detailsHtml = `
            <div class="mt-3 text-xs space-y-2 text-gray-300">
              <div class="flex justify-between"><span class="text-gray-400">Position:</span> <span class="font-bold text-white">${a.side || '—'} · ${a.quantity} ${a.unit}</span></div>
              <div class="flex justify-between"><span class="text-gray-400">Entry Price:</span> <span>${formatMoney(a.entry)}</span></div>
              <div class="flex justify-between"><span class="text-gray-400">Live Mark:</span> <span class="font-bold text-white">${formatMoney(a.mark)}</span></div>
              <div class="flex justify-between"><span class="text-gray-400">Stop-Loss (Risk):</span> <span class="text-rose-400">${formatMoney(a.stop_loss)}</span></div>
              <div class="flex justify-between"><span class="text-gray-400">Target 1 (+1.25R):</span> <span class="text-emerald-400">${formatMoney(a.take_profit)}</span></div>

              <!-- Runner Flight Path -->
              <div class="pt-2 border-t border-gray-800/80">
                <div class="flex justify-between text-[11px] mb-1">
                  <span class="text-gray-400">TP1 Progress:</span>
                  <span class="font-mono font-bold ${pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}">${flightPct}%</span>
                </div>
                <div class="runner-flight-track">
                  <div class="runner-flight-fill ${barColor}" style="width: ${flightPct}%;"></div>
                </div>
                <div class="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                  <span>Entry: ${formatMoney(a.entry)}</span>
                  <span>Target: ${formatMoney(a.take_profit)}</span>
                </div>
              </div>
            </div>
          `;
        } else {
          const bias = (radar.bias || "NEUTRAL").toUpperCase();
          let biasBadge = `<span class="px-2 py-0.5 rounded text-[10px] bg-gray-800 text-gray-400 border border-gray-700 font-bold">NEUTRAL ↔</span>`;
          if (bias === "BEARISH") {
            biasBadge = `<span class="px-2 py-0.5 rounded text-[10px] bg-rose-950 text-rose-400 border border-rose-800 font-bold">BEARISH ↓</span>`;
          } else if (bias === "BULLISH") {
            biasBadge = `<span class="px-2 py-0.5 rounded text-[10px] bg-emerald-950 text-emerald-400 border border-emerald-800 font-bold">BULLISH ↑</span>`;
          }

          const markVal = parseFloat(a.mark || 0);
          const dLower = parseFloat(radar.donchian_lower || (markVal * 0.98));
          const dUpper = parseFloat(radar.donchian_upper || (markVal * 1.02));
          let pinPct = 50;
          if (dUpper > dLower) {
            pinPct = Math.max(5, Math.min(95, Math.round(((markVal - dLower) / (dUpper - dLower)) * 100)));
          }

          const atrText = radar.atr !== null && radar.atr !== undefined ? `$${radar.atr}` : "—";
          const rsiVal = parseFloat(radar.rsi || 50);
          let rsiColor = "text-gray-300";
          if (rsiVal >= 70) rsiColor = "text-rose-400 font-bold";
          else if (rsiVal <= 30) rsiColor = "text-emerald-400 font-bold";

          detailsHtml = `
            <div class="mt-3 text-xs space-y-2.5 text-gray-400">
              <div class="flex justify-between items-center">
                <span>1H Trend Bias:</span>
                <div>${biasBadge}</div>
              </div>

              <!-- Donchian 10 Channel Proximity -->
              <div>
                <div class="flex justify-between text-[10px] text-gray-500 mb-1">
                  <span>Donchian Low: ${formatMoney(radar.donchian_lower)}</span>
                  <span>High: ${formatMoney(radar.donchian_upper)}</span>
                </div>
                <div class="donchian-track">
                  <div class="donchian-pin bg-emerald-400 shadow-md shadow-emerald-500/50" style="left: ${pinPct}%;"></div>
                </div>
              </div>

              <!-- Micro Indicators Grid -->
              <div class="grid grid-cols-2 gap-2 pt-1 font-mono text-[11px]">
                <div class="bg-gray-900/60 p-1.5 rounded border border-gray-800/80 text-center">
                  <div class="text-[10px] text-gray-500">1H ATR Stop</div>
                  <div class="text-gray-200 font-bold">${atrText}</div>
                </div>
                <div class="bg-gray-900/60 p-1.5 rounded border border-gray-800/80 text-center">
                  <div class="text-[10px] text-gray-500">1H RSI (14)</div>
                  <div class="${rsiColor}">${radar.rsi !== null && radar.rsi !== undefined ? radar.rsi : '—'}</div>
                </div>
              </div>

              <!-- Setup / Policy Description -->
              <div class="bg-gray-900/50 p-2 rounded border border-gray-800/60 text-[11px] text-gray-400 line-clamp-2" title="${a.risk_desc}">
                <span class="text-gray-500">Setup:</span> ${a.risk_desc}
              </div>

              <div class="flex justify-between pt-1 border-t border-gray-800/60 text-[11px]">
                <span>Today's Realized:</span>
                <span class="${a.realized_today >= 0 ? 'text-emerald-400' : 'text-rose-400'} font-bold">${formatMoney(a.realized_today)}</span>
              </div>
            </div>
          `;
        }

        const card = document.createElement("div");
        card.className = "asset-card";
        card.innerHTML = `
          <div class="flex items-center justify-between">
            <div>
              <div class="flex items-center gap-2">
                <h3 class="text-lg font-bold text-white">${a.asset}</h3>
                <span class="px-2 py-0.5 rounded text-[10px] font-bold ${modelBadgeClass}">${a.model_name}</span>
              </div>
              <p class="text-xs text-gray-400">${a.symbol} · ${a.venue}</p>
            </div>
            <span class="${pillClass}">${pillText}</span>
          </div>
          <div class="mt-3 pt-3 border-t border-gray-800">
            <div class="text-xs text-gray-400">Live Mark / Unrealized P&L</div>
            <div class="flex items-baseline justify-between mt-0.5">
              <span class="text-lg font-bold text-white font-mono">${formatMoney(a.mark)}</span>
              <span class="text-lg ${pnlClass} font-mono">${formatMoney(pnl)}</span>
            </div>
          </div>
          ${detailsHtml}
        `;
        container.appendChild(card);
      });
    }

    function renderCharts(monthly, overall, monthlyByModel, overallByModel) {
      let mList = monthly || [];
      let oList = overall || [];
      let label = "All Models";

      if (activeModelView === "model_1") {
        mList = (monthlyByModel && monthlyByModel.model_1) || [];
        oList = (overallByModel && overallByModel.model_1) || [];
        label = "Model 1 (Baseline)";
      } else if (activeModelView === "model_2") {
        mList = (monthlyByModel && monthlyByModel.model_2) || [];
        oList = (overallByModel && overallByModel.model_2) || [];
        label = "Model 2 (Profile A Yield)";
      }

      document.getElementById("chart-monthly-tag").innerText = label;
      document.getElementById("chart-overall-tag").innerText = label;

      if (mList && mList.length > 0) {
        const xMonths = mList.map(m => m.month.slice(0, 7));
        const yPnl = mList.map(m => m.pnl);
        const colors = yPnl.map(p => p >= 0 ? '#16a34a' : '#dc2626');

        Plotly.react('chart-monthly', [{
          x: xMonths,
          y: yPnl,
          type: 'bar',
          marker: { color: colors }
        }], {
          plot_bgcolor: 'rgba(0,0,0,0)',
          paper_bgcolor: 'rgba(0,0,0,0)',
          font: { color: '#9ca3af', family: 'sans-serif' },
          margin: { l: 40, r: 20, t: 20, b: 40 },
          yaxis: { gridcolor: '#1f2937', zerolinecolor: '#4b5563' },
          xaxis: { gridcolor: '#1f2937' }
        }, { responsive: true, displayModeBar: false });
      }

      if (oList && oList.length > 0) {
        const xDates = oList.map(o => o.date);
        const yCum = oList.map(o => o.cumulative_pnl);

        Plotly.react('chart-overall', [{
          x: xDates,
          y: yCum,
          type: 'scatter',
          mode: 'lines',
          fill: 'tozeroy',
          line: { color: '#10b981', width: 2 },
          fillcolor: 'rgba(16, 185, 129, 0.15)'
        }], {
          plot_bgcolor: 'rgba(0,0,0,0)',
          paper_bgcolor: 'rgba(0,0,0,0)',
          font: { color: '#9ca3af', family: 'sans-serif' },
          margin: { l: 40, r: 20, t: 20, b: 40 },
          yaxis: { gridcolor: '#1f2937', zerolinecolor: '#4b5563' },
          xaxis: { gridcolor: '#1f2937' }
        }, { responsive: true, displayModeBar: false });
      }
    }

    function renderTrades() {
      if (!globalData || !globalData.trades) return;
      const modelFilter = document.getElementById("model-filter").value;
      const assetFilter = document.getElementById("asset-filter").value;
      const sideFilter = document.getElementById("side-filter").value;
      const tbody = document.getElementById("trades-tbody");
      tbody.innerHTML = "";

      let filtered = [...globalData.trades];
      if (modelFilter !== "ALL") {
        filtered = filtered.filter(t => t.model_id === modelFilter);
      }
      if (assetFilter !== "ALL") {
        filtered = filtered.filter(t => t.asset && t.asset.toUpperCase() === assetFilter.toUpperCase());
      }
      if (sideFilter !== "ALL") {
        filtered = filtered.filter(t => t.side && t.side.toUpperCase() === sideFilter.toUpperCase());
      }
      if (!showClosed) {
        filtered = filtered.filter(t => t.status === "OPEN" || t.status === "ACTIVE");
      }

      // Sort
      filtered.sort((a, b) => {
        let valA = a[sortColumn];
        let valB = b[sortColumn];
        if (sortColumn === "pnl" || sortColumn === "entry" || sortColumn === "exit") {
          valA = parseFloat(valA || 0);
          valB = parseFloat(valB || 0);
          return sortAsc ? valA - valB : valB - valA;
        }
        valA = (valA || "").toString().toLowerCase();
        valB = (valB || "").toString().toLowerCase();
        return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
      });

      // Update sort indicators
      ["time", "asset", "side", "qty", "status", "pnl"].forEach(col => {
        const el = document.getElementById(`sort-${col}`);
        if (el) {
          if (sortColumn === col) {
            el.innerText = sortAsc ? "▲" : "▼";
            el.className = "text-emerald-400 font-bold";
          } else {
            el.innerText = "↕";
            el.className = "text-gray-500";
          }
        }
      });

      const totalItems = filtered.length;
      const totalPages = Math.max(1, Math.ceil(totalItems / tradesPageSize));
      if (tradesCurrentPage > totalPages) tradesCurrentPage = totalPages;
      if (tradesCurrentPage < 1) tradesCurrentPage = 1;

      const startIdx = (tradesCurrentPage - 1) * tradesPageSize;
      const endIdx = Math.min(startIdx + tradesPageSize, totalItems);
      const pageItems = filtered.slice(startIdx, endIdx);

      if (pageItems.length === 0) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="10" class="py-8 text-center text-gray-500 italic font-sans">No trade activity recorded for this filter</td>`;
        tbody.appendChild(tr);
      } else {
        pageItems.forEach((t, i) => {
          const globalIdx = startIdx + i;
          const isClosed = t.status === "CLOSED";
          const pnl = parseFloat(t.pnl || 0);
          const pnlClass = pnl >= 0 ? "text-emerald-400 font-bold" : "text-rose-400 font-bold";
          const sideClass = t.side === "BUY" ? "text-emerald-400" : "text-rose-400";
          const statusBadge = isClosed 
            ? `<span class="px-2 py-0.5 rounded text-[10px] bg-gray-800 text-gray-400">CLOSED</span>`
            : `<span class="px-2 py-0.5 rounded text-[10px] bg-emerald-950 text-emerald-400 font-bold border border-emerald-800">OPEN</span>`;

          const isM1 = t.model_id === "model_1";
          const modelBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-bold ${isM1 ? 'badge-m1' : 'badge-m2'}">${t.model_name}</span>`;

          const isExpanded = expandedTradeIndex === globalIdx;

          const tr = document.createElement("tr");
          tr.className = "hover:bg-gray-800/40 transition-colors cursor-pointer";
          tr.onclick = () => toggleTradeRow(globalIdx);
          tr.innerHTML = `
            <td class="py-2.5 px-3 text-gray-400 whitespace-nowrap">${t.time}</td>
            <td class="py-2.5 px-3 whitespace-nowrap">${modelBadge}</td>
            <td class="py-2.5 px-3 font-semibold text-white">${t.asset}</td>
            <td class="py-2.5 px-3 ${sideClass} font-bold">${t.side}</td>
            <td class="py-2.5 px-3 text-gray-300">${t.qty}</td>
            <td class="py-2.5 px-3">${statusBadge}</td>
            <td class="py-2.5 px-3 text-gray-300">${formatMoney(t.entry)}</td>
            <td class="py-2.5 px-3 text-gray-300">${formatMoney(t.exit)}</td>
            <td class="py-2.5 px-3 ${pnlClass}">${formatMoney(pnl)}</td>
            <td class="py-2.5 px-3 text-gray-400">${t.strategy} ${isExpanded ? '▲' : '▼'}</td>
          `;
          tbody.appendChild(tr);

          if (isExpanded) {
            const detailTr = document.createElement("tr");
            detailTr.className = "bg-gray-900/70 border-b border-gray-800";
            detailTr.innerHTML = `
              <td colspan="10" class="p-3">
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-sans">
                  <div class="bg-gray-800/50 p-2 rounded border border-gray-700/60">
                    <span class="text-gray-400 block text-[10px]">MT5 Deal Ticket</span>
                    <span class="font-mono text-white font-bold">${t.ticket || '—'}</span>
                  </div>
                  <div class="bg-gray-800/50 p-2 rounded border border-gray-700/60">
                    <span class="text-gray-400 block text-[10px]">Dynamic Stop-Loss</span>
                    <span class="font-mono text-rose-400 font-bold">${formatMoney(t.sl)}</span>
                  </div>
                  <div class="bg-gray-800/50 p-2 rounded border border-gray-700/60">
                    <span class="text-gray-400 block text-[10px]">Target 1 (${t.rr || 1.25}R)</span>
                    <span class="font-mono text-emerald-400 font-bold">${formatMoney(t.tp)}</span>
                  </div>
                  <div class="bg-gray-800/50 p-2 rounded border border-gray-700/60">
                    <span class="text-gray-400 block text-[10px]">Execution Model & Comment</span>
                    <span class="font-mono text-gray-300">${t.model_full} · ${t.comment || 'None'}</span>
                  </div>
                </div>
              </td>
            `;
            tbody.appendChild(detailTr);
          }
        });
      }

      // Update Range / Count Info
      const rangeEl = document.getElementById("trades-range-info");
      if (rangeEl) {
        if (totalItems === 0) {
          rangeEl.innerText = "Showing 0 trades";
        } else {
          rangeEl.innerText = `Showing ${startIdx + 1}–${endIdx} of ${totalItems} trades (Page ${tradesCurrentPage} of ${totalPages})`;
        }
      }

      // Render Navigation Buttons
      const navEl = document.getElementById("trades-pagination-nav");
      if (navEl) {
        navEl.innerHTML = "";
        if (totalPages > 1) {
          const prevBtn = document.createElement("button");
          prevBtn.innerHTML = "&larr; Prev";
          prevBtn.className = "px-2.5 py-1 text-xs rounded-lg border border-gray-700 bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors disabled:opacity-30 disabled:cursor-not-allowed";
          prevBtn.disabled = tradesCurrentPage === 1;
          prevBtn.onclick = () => goToTradesPage(tradesCurrentPage - 1);
          navEl.appendChild(prevBtn);

          let pages = [];
          if (totalPages <= 7) {
            for (let i = 1; i <= totalPages; i++) pages.push(i);
          } else {
            pages.push(1);
            if (tradesCurrentPage > 3) pages.push("...");
            const start = Math.max(2, tradesCurrentPage - 1);
            const end = Math.min(totalPages - 1, tradesCurrentPage + 1);
            for (let i = start; i <= end; i++) pages.push(i);
            if (tradesCurrentPage < totalPages - 2) pages.push("...");
            pages.push(totalPages);
          }

          pages.forEach(p => {
            if (p === "...") {
              const span = document.createElement("span");
              span.className = "px-1 text-gray-500 select-none text-xs";
              span.innerText = "...";
              navEl.appendChild(span);
            } else {
              const btn = document.createElement("button");
              btn.innerText = p;
              if (p === tradesCurrentPage) {
                btn.className = "px-2.5 py-1 text-xs rounded-lg font-bold bg-emerald-600 text-white border border-emerald-500 shadow-sm";
              } else {
                btn.className = "px-2.5 py-1 text-xs rounded-lg border border-gray-700 bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors cursor-pointer";
              }
              btn.onclick = () => goToTradesPage(p);
              navEl.appendChild(btn);
            }
          });

          const nextBtn = document.createElement("button");
          nextBtn.innerHTML = "Next &rarr;";
          nextBtn.className = "px-2.5 py-1 text-xs rounded-lg border border-gray-700 bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors disabled:opacity-30 disabled:cursor-not-allowed";
          nextBtn.disabled = tradesCurrentPage === totalPages;
          nextBtn.onclick = () => goToTradesPage(tradesCurrentPage + 1);
          navEl.appendChild(nextBtn);
        }
      }
    }

    function updateView() {
      if (!globalData) return;
      const data = globalData;

      // 100% Segregation: Hide Treasury Card in Model 1; Show in Model 2 and Consolidated
      const treasuryCard = document.getElementById("treasury-card");
      if (treasuryCard) {
        if (activeModelView === "model_1") {
          treasuryCard.style.display = "none";
        } else {
          treasuryCard.style.display = "block";
        }
      }

      // Update KPIs based on activeModelView
      let activeCards = data.assets || [];
      let activeTrades = data.trades || [];
      if (activeModelView !== "ALL") {
        activeCards = activeCards.filter(a => a.model_id === activeModelView);
        activeTrades = activeTrades.filter(t => t.model_id === activeModelView);
      }

      const openPnl = activeCards.reduce((sum, a) => sum + (parseFloat(a.unrealized_pnl) || 0), 0);
      const todayRealized = activeCards.reduce((sum, a) => sum + (parseFloat(a.realized_today) || 0), 0);
      const activeCount = activeCards.filter(a => a.status === "IN TRADE").length;

      let totalRealized = 0;
      if (activeModelView === "ALL") {
        totalRealized = data.monthly.reduce((sum, m) => sum + (parseFloat(m.pnl) || 0), 0);
      } else if (activeModelView === "model_1") {
        totalRealized = (data.monthly_by_model.model_1 || []).reduce((sum, m) => sum + (parseFloat(m.pnl) || 0), 0);
      } else if (activeModelView === "model_2") {
        totalRealized = (data.monthly_by_model.model_2 || []).reduce((sum, m) => sum + (parseFloat(m.pnl) || 0), 0);
      }

      document.getElementById("metric-total-pnl").innerText = formatMoney(totalRealized);
      document.getElementById("metric-open-pnl").innerText = formatMoney(openPnl);
      document.getElementById("metric-today-pnl").innerText = formatMoney(todayRealized);
      document.getElementById("metric-active-count").innerText = `${activeCount} / ${activeCards.length}`;

      document.getElementById("metric-total-pnl").className = "text-2xl font-bold mt-1 " + (totalRealized >= 0 ? "text-emerald-400" : "text-rose-400");
      document.getElementById("metric-open-pnl").className = "text-2xl font-bold mt-1 " + (openPnl >= 0 ? "text-emerald-400" : "text-rose-400");
      document.getElementById("metric-today-pnl").className = "text-2xl font-bold mt-1 " + (todayRealized >= 0 ? "text-emerald-400" : "text-rose-400");

      // Dynamic Dual-Account Display based on activeModelView
      const badge = document.getElementById("broker-status-badge");
      const balEl = document.getElementById("metric-account-balance");
      const eqEl = document.getElementById("metric-account-equity");
      const accLabelEl = document.getElementById("label-account-balance");
      const accEqLabelEl = document.getElementById("label-account-equity");

      const acc1 = data.account_model_1 || data.account || {};
      const acc2 = data.account_model_2 || {};
      const bal1 = parseFloat(acc1.balance) || 0;
      const bal2 = parseFloat(acc2.balance) || 0;
      const eq1 = parseFloat(acc1.equity) || 0;
      const eq2 = parseFloat(acc2.equity) || 0;

      if (activeModelView === "model_1") {
        if (balEl) balEl.innerText = formatMoney(bal1);
        if (eqEl) eqEl.innerText = formatMoney(eq1);
        if (accLabelEl) accLabelEl.innerText = "Model 1 Balance (#" + (acc1.account_id || "168987477") + ")";
        if (accEqLabelEl) accEqLabelEl.innerText = "Model 1 Equity";
        if (badge) badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-sky-400 animate-pulse"></span> MT5 Demo #${acc1.account_id || '168987477'} (${formatMoney(bal1)}) Connected`;
      } else if (activeModelView === "model_2") {
        if (balEl) balEl.innerText = formatMoney(bal2);
        if (eqEl) eqEl.innerText = formatMoney(eq2);
        if (accLabelEl) accLabelEl.innerText = "Monthly_winsaw Balance (#" + (acc2.account_id || "346439083") + ")";
        if (accEqLabelEl) accEqLabelEl.innerText = "Monthly_winsaw Equity";
        if (badge) badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-purple-400 animate-pulse"></span> MT5 'Monthly_winsaw' #${acc2.account_id || '346439083'} (${formatMoney(bal2)}) Connected`;
      } else {
        const totalBal = bal1 + bal2;
        const totalEq = eq1 + eq2;
        if (balEl) balEl.innerText = formatMoney(totalBal);
        if (eqEl) eqEl.innerText = formatMoney(totalEq);
        if (accLabelEl) accLabelEl.innerText = "Combined Dual Account Balance";
        if (accEqLabelEl) accEqLabelEl.innerText = "Combined Total Equity";
        if (badge) badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Dual MT5 Accounts Connected (${formatMoney(totalBal)})`;
      }

      computePerformanceKPIs(data.trades);
      renderTreasuryWidget(data.treasury_model_2);
      renderAssetCards(data.assets);
      renderCharts(data.monthly, data.overall, data.monthly_by_model, data.overall_by_model);
      renderTrades();
    }

    async function refreshDashboard() {
      try {
        const res = await fetch("/api/dashboard/overview");
        if (!res.ok) throw new Error("API status " + res.status);
        const data = await res.json();
        globalData = data;

        if (data.trades) {
          const currentCount = data.trades.length;
          if (previousTradeCount !== null && currentCount > previousTradeCount) {
            playChime();
          }
          previousTradeCount = currentCount;
        }

        updateView();
        document.getElementById("last-refresh").innerText = "Updated " + new Date().toLocaleTimeString();
      } catch (err) {
        console.error("Dashboard refresh error:", err);
      }
    }

    let refreshTimer = null;
    function scheduleEcoRefresh() {
      if (refreshTimer) clearTimeout(refreshTimer);
      const delay = document.hidden ? 10000 : 3000;
      refreshTimer = setTimeout(async () => {
        await refreshDashboard();
        scheduleEcoRefresh();
      }, delay);
    }

    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) {
        refreshDashboard();
      }
      scheduleEcoRefresh();
    });

    initAudioUI();
    refreshDashboard().then(scheduleEcoRefresh);
  </script>
</body>
</html>
"""


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def index_page():
    return HTML_CONTENT


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("gold_trader.api.dashboard_api:app", host="127.0.0.1", port=8787, reload=False)
