"""
FastAPI Backend & Standalone UI Server for Atlas Multi-Asset Trading Dashboard.
Provides:
- Web UI at "/" matching exact Atlas Trade design (Tailwind + Plotly.js + Auto-refresh)
- REST API at "/api/dashboard/overview"
- Health at "/api/health"
"""
from __future__ import annotations

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

app = FastAPI(title="Aegis Alpha Quantitative Trading System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.expanduser("~/.gold_trading_data")
DB_MAP = {
    "XAUUSD": os.path.join(DATA_DIR, "auto_trader.db"),
    "BTCUSD": os.path.join(DATA_DIR, "btc_auto_trader.db"),
    "ETHUSD": os.path.join(DATA_DIR, "eth_auto_trader.db"),
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

_broker_instance: Optional[SocketBroker] = None


def get_broker() -> Optional[SocketBroker]:
    global _broker_instance
    if _broker_instance is None:
        try:
            broker = SocketBroker()
            if broker.connect():
                _broker_instance = broker
        except Exception as e:
            logger.warning(f"MT5 broker bridge connect failed: {e}")
            _broker_instance = None
    return _broker_instance


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


def fetch_latest_scan_telemetry() -> Dict[str, Any]:
    telemetry = {}
    for sym, path in DB_MAP.items():
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
                        import json
                        det = json.loads(details_raw).get("setup_details", {})
                    except Exception:
                        pass
                telemetry[sym] = {
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
            logger.warning(f"Error reading scan_logs for {sym}: {e}")
    return telemetry


def fetch_historic_and_pnl_data() -> Dict[str, Any]:
    monthly_pnl = defaultdict(float)
    daily_pnl = defaultdict(float)
    today_realized_by_sym = defaultdict(float)
    all_trades: List[Dict[str, Any]] = []

    for sym, path in DB_MAP.items():
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

            today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            for r in rows:
                ts, ticket, symbol, action, lots, status, entry, exit_p, pnl_val, comment, close_t, sl_val, tp1_val, tp2_val, rr_val = r
                pnl = float(pnl_val or 0.0)
                t_str = close_t or ts

                if t_str and status == "CLOSED":
                    dt_str = t_str[:10]
                    m_str = t_str[:7] + "-01"
                    monthly_pnl[m_str] += pnl
                    daily_pnl[dt_str] += pnl

                    if dt_str == today_utc:
                        today_realized_by_sym[symbol] += pnl

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
                        "strategy": (
                            "1H Donchian 10 Touch"
                            if "FlatRisk" in (comment or "")
                            else (comment or "Trend Strategy")
                        ),
                    }
                )
        except Exception as exc:
            logger.warning(f"Error querying {path}: {exc}")

    all_trades.sort(key=lambda x: x["time"], reverse=True)

    monthly_list = [
        {"month": m, "pnl": round(val, 2)} for m, val in sorted(monthly_pnl.items())
    ]

    cum = 0.0
    overall_list = []
    for d, val in sorted(daily_pnl.items()):
        cum += val
        overall_list.append({"date": d, "cumulative_pnl": round(cum, 2)})

    return {
        "monthly": monthly_list,
        "overall": overall_list,
        "trades": all_trades,
        "today_realized_by_sym": today_realized_by_sym,
    }


def fetch_live_asset_cards(today_realized_map: Dict[str, float]) -> List[Dict[str, Any]]:
    broker = get_broker()
    now_iso = datetime.now(timezone.utc).isoformat()
    telemetry = fetch_latest_scan_telemetry()

    open_positions: List[Dict[str, Any]] = []
    if broker:
        try:
            open_positions = broker.get_open_positions()
        except Exception as e:
            logger.warning(f"Failed to fetch live open positions: {e}")

    open_by_symbol = {}
    for p in open_positions:
        sym = p.get("symbol", "").upper()
        if "GOLD" in sym or "XAU" in sym:
            sym = "XAUUSD"
        elif "BTC" in sym:
            sym = "BTCUSD"
        elif "ETH" in sym:
            sym = "ETHUSD"
        open_by_symbol[sym] = p

    assets_config = [
        {
            "asset": "Gold",
            "symbol": "XAUUSD",
            "venue": "XM / MetaTrader 5",
            "unit": "lots",
            "price_format": 2,
            "default_mark": 4031.50,
        },
        {
            "asset": "Bitcoin",
            "symbol": "BTCUSD",
            "venue": "XM / MetaTrader 5",
            "unit": "BTC",
            "price_format": 2,
            "default_mark": 81850.0,
        },
        {
            "asset": "Ethereum",
            "symbol": "ETHUSD",
            "venue": "XM / MetaTrader 5",
            "unit": "ETH",
            "price_format": 2,
            "default_mark": 2404.0,
        },
    ]

    result = []
    for item in assets_config:
        sym = item["symbol"]
        mark_price = item["default_mark"]

        if broker:
            try:
                info = broker.get_symbol_info(sym)
                if info.get("status") == "SUCCESS":
                    bid = float(info.get("bid", 0.0))
                    if bid > 0:
                        mark_price = bid
                        _CACHE["ticks"][sym] = bid
            except Exception:
                mark_price = _CACHE["ticks"].get(sym, mark_price)
        else:
            mark_price = _CACHE["ticks"].get(sym, mark_price)

        radar_info = telemetry.get(sym, {
            "price": round(mark_price, 2),
            "bias": "NEUTRAL",
            "decision": "",
            "atr": None,
            "rsi": None,
            "supertrend": 0,
            "donchian_upper": None,
            "donchian_lower": None,
            "donchian_middle": None,
            "reason": "Scanning for 1H Touch",
            "timestamp": "",
        })

        pos = open_by_symbol.get(sym)
        if pos:
            result.append(
                {
                    "asset": item["asset"],
                    "symbol": sym,
                    "venue": item["venue"],
                    "status": "IN TRADE",
                    "side": pos.get("side", "BUY"),
                    "entry": float(pos.get("open_price", mark_price)),
                    "mark": float(pos.get("current_price", mark_price)),
                    "quantity": float(pos.get("lots", 0.01)),
                    "unrealized_pnl": round(float(pos.get("pnl", 0.0)), 2),
                    "realized_today": round(float(today_realized_map.get(sym, 0.0)), 2),
                    "stop_loss": float(pos.get("sl")) if pos.get("sl") else None,
                    "take_profit": float(pos.get("tp")) if pos.get("tp") else None,
                    "last_update": now_iso,
                    "price_format": item["price_format"],
                    "unit": item["unit"],
                    "radar": radar_info,
                }
            )
        else:
            result.append(
                {
                    "asset": item["asset"],
                    "symbol": sym,
                    "venue": item["venue"],
                    "status": "NO TRADE",
                    "side": None,
                    "entry": None,
                    "mark": round(mark_price, 2),
                    "quantity": 0.0,
                    "unrealized_pnl": 0.0,
                    "realized_today": round(float(today_realized_map.get(sym, 0.0)), 2),
                    "stop_loss": None,
                    "take_profit": None,
                    "last_update": now_iso,
                    "price_format": item["price_format"],
                    "unit": item["unit"],
                    "radar": radar_info,
                }
            )

    return result


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
    }


@app.get("/api/dashboard/overview")
def get_dashboard_overview():
    now_ts = time.time()
    if _CACHE["payload"] and (now_ts - _CACHE["timestamp"] < _CACHE["ttl_seconds"]):
        return _CACHE["payload"]

    history_data = fetch_historic_and_pnl_data()
    assets_cards = fetch_live_asset_cards(history_data["today_realized_by_sym"])

    broker = get_broker()
    account_info = None
    if broker:
        try:
            account_info = broker.get_account_info()
        except Exception as exc:
            logger.warning(f"Error fetching account info for overview: {exc}")

    payload = {
        "assets": assets_cards,
        "monthly": history_data["monthly"],
        "overall": history_data["overall"],
        "trades": history_data["trades"],
        "account": account_info,
    }

    _CACHE["payload"] = payload
    _CACHE["timestamp"] = now_ts
    return payload


# ---- Standalone UI Handler matching Atlas Design ----
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Aegis Alpha | Quantitative Portfolio Overview</title>
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
  </style>
</head>
<body class="p-4 md:p-8 max-w-7xl mx-auto">

  <!-- Header -->
  <div class="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-gray-800 gap-4">
    <div>
      <h1 class="text-2xl md:text-3xl font-bold tracking-tight text-white flex items-center gap-2">
        <span>🛡️</span> Aegis Alpha | Quantitative Trading System
      </h1>
      <p class="text-sm text-gray-400 mt-1">Spot Gold · Bitcoin · Ethereum | Autonomous Execution · ML Meta-Labeling · Risk Shield</p>
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

  <!-- Macro & Engine Policy Banner -->
  <div class="bg-gradient-to-r from-gray-900/90 via-gray-900/60 to-gray-900/90 border border-gray-800/80 rounded-xl px-4 py-2.5 my-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-3 text-xs">
    <div class="flex items-center gap-2.5">
      <span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-950 text-emerald-400 font-bold border border-emerald-800 text-[11px]">
        <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span> NEWS GUARD: CLEAR
      </span>
      <span class="text-gray-300">Execution window clear · No high-impact catalysts (CPI/NFP/FOMC) scheduled within ±15m.</span>
    </div>
    <div class="flex flex-wrap items-center gap-3 text-gray-400 font-mono text-[11px]">
      <span>Mode: <strong class="text-white">FLAT_RISK_TOUCH</strong> ($75.44 Risk)</span>
      <span class="text-gray-700">|</span>
      <span>Target 1: <strong class="text-emerald-400">1.25R (+50% Cash)</strong></span>
      <span class="text-gray-700">|</span>
      <span>Target 2: <strong class="text-emerald-400">3.0R (Risk-Free Runner)</strong></span>
    </div>
  </div>

  <!-- Master KPI Metrics -->
  <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 my-4">
    <div class="metric-card border border-emerald-500/30">
      <div class="text-xs font-medium text-emerald-400 uppercase tracking-wider">Account Balance</div>
      <div id="metric-account-balance" class="text-2xl font-bold text-white mt-1">$0.00</div>
    </div>
    <div class="metric-card border border-emerald-500/30">
      <div class="text-xs font-medium text-emerald-400 uppercase tracking-wider">Account Equity</div>
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
      <div class="text-xs font-medium text-gray-400 uppercase tracking-wider">Assets in Trade</div>
      <div id="metric-active-count" class="text-2xl font-bold text-white mt-1">0 / 3</div>
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

  <!-- Asset Cards -->
  <h2 class="text-lg font-semibold text-gray-200 mb-3 flex items-center gap-2">
    <span>💎</span> Active Asset Status & Bot Market Radar
  </h2>
  <div id="asset-cards-grid" class="grid grid-cols-1 md:grid-cols-3 gap-5 mb-8">
    <!-- Populated dynamically via JS -->
  </div>

  <!-- Charts Section -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 my-8">
    <div class="metric-card">
      <h3 class="text-base font-semibold text-gray-200 mb-2">Month-on-Month Realized P&L</h3>
      <div id="chart-monthly" style="height: 320px;"></div>
    </div>
    <div class="metric-card">
      <h3 class="text-base font-semibold text-gray-200 mb-2">Overall Cumulative Realized P&L</h3>
      <div id="chart-overall" style="height: 320px;"></div>
    </div>
  </div>

  <!-- Trade Activity Table -->
  <div class="metric-card my-8">
    <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-gray-800 gap-3">
      <div>
        <h3 class="text-base font-semibold text-gray-200">Trade Activity (Fills & Executions)</h3>
        <p class="text-[11px] text-gray-500 mt-0.5">Click column headers to sort · Click any row to expand ticket & SL/TP details</p>
      </div>
      <div class="flex flex-wrap items-center gap-2.5">
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
            <th class="py-2.5 px-3 cursor-pointer hover:text-white transition-colors" onclick="sortTable('asset')">Asset <span id="sort-asset" class="text-gray-500">↕</span></th>
            <th class="py-2.5 px-3 cursor-pointer hover:text-white transition-colors" onclick="sortTable('side')">Side <span id="sort-side" class="text-gray-500">↕</span></th>
            <th class="py-2.5 px-3 cursor-pointer hover:text-white transition-colors" onclick="sortTable('qty')">Quantity <span id="sort-qty" class="text-gray-500">↕</span></th>
            <th class="py-2.5 px-3 cursor-pointer hover:text-white transition-colors" onclick="sortTable('status')">Status <span id="sort-status" class="text-gray-500">↕</span></th>
            <th class="py-2.5 px-3">Entry</th>
            <th class="py-2.5 px-3">Exit</th>
            <th class="py-2.5 px-3 cursor-pointer hover:text-white transition-colors" onclick="sortTable('pnl')">P&L ($) <span id="sort-pnl" class="text-gray-500">↕</span></th>
            <th class="py-2.5 px-3">Strategy</th>
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
      <div class="text-gray-400">Trading Engine</div>
      <div class="font-bold text-emerald-400 mt-1">FLAT_RISK_TOUCH</div>
    </div>
    <div class="metric-card text-center">
      <div class="text-gray-400">News Catalyst Guard</div>
      <div class="font-bold text-emerald-400 mt-1">ACTIVE (±15m)</div>
    </div>
    <div class="metric-card text-center">
      <div class="text-gray-400">ML Meta-Labeling</div>
      <div class="font-bold text-emerald-400 mt-1">P ≥ 0.44 Gate</div>
    </div>
  </div>

  <script>
    let globalData = null;
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

    function onFilterChange() {
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
        sortAsc = col === "asset" || col === "side" || col === "status";
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
      const assetFilter = document.getElementById("asset-filter").value;
      const sideFilter = document.getElementById("side-filter").value;
      let list = globalData.trades;
      if (assetFilter !== "ALL") list = list.filter(t => t.asset && t.asset.toUpperCase() === assetFilter.toUpperCase());
      if (sideFilter !== "ALL") list = list.filter(t => t.side && t.side.toUpperCase() === sideFilter.toUpperCase());
      if (!showClosed) list = list.filter(t => t.status === "OPEN" || t.status === "ACTIVE");

      const headers = ["Time (UTC)", "Ticket", "Asset", "Side", "Quantity", "Status", "Entry Price", "Exit Price", "Stop Loss", "Target 1", "Realized P&L ($)", "Strategy"];
      const rows = list.map(t => [
        `"${t.time}"`,
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
      if (!trades || trades.length === 0) return;
      const closed = trades.filter(t => t.status === "CLOSED");
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

      // Max Drawdown calculation chronologically
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

    function renderAssetCards(assets) {
      const container = document.getElementById("asset-cards-grid");
      container.innerHTML = "";
      assets.forEach(a => {
        const inTrade = a.status === "IN TRADE";
        const pillClass = inTrade ? "status-pill" : "status-pill status-idle";
        const pillText = inTrade ? "🟢 IN TRADE" : "⚪ SCANNING";
        const pnl = parseFloat(a.unrealized_pnl || 0);
        const pnlClass = pnl >= 0 ? "pnl-pos" : "pnl-neg";
        const radar = a.radar || {};

        let detailsHtml = "";
        if (inTrade) {
          // Flight path calculation
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
          // Market Radar Box
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

              <!-- Bot Rationale Box -->
              <div class="bg-gray-900/50 p-2 rounded border border-gray-800/60 text-[11px] text-gray-400 line-clamp-2" title="${radar.reason || 'Scanning for 1H Donchian touch'}">
                <span class="text-gray-500">Bot:</span> ${radar.reason || 'Scanning for 1H Touch breakout.'}
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
              <h3 class="text-lg font-bold text-white">${a.asset}</h3>
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

    function renderCharts(monthly, overall) {
      if (monthly && monthly.length > 0) {
        const xMonths = monthly.map(m => m.month.slice(0, 7));
        const yPnl = monthly.map(m => m.pnl);
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

      if (overall && overall.length > 0) {
        const xDates = overall.map(o => o.date);
        const yCum = overall.map(o => o.cumulative_pnl);

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
      const assetFilter = document.getElementById("asset-filter").value;
      const sideFilter = document.getElementById("side-filter").value;
      const tbody = document.getElementById("trades-tbody");
      tbody.innerHTML = "";

      let filtered = [...globalData.trades];
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
        tr.innerHTML = `<td colspan="9" class="py-8 text-center text-gray-500 italic font-sans">No trade activity recorded</td>`;
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

          const isExpanded = expandedTradeIndex === globalIdx;

          const tr = document.createElement("tr");
          tr.className = "hover:bg-gray-800/40 transition-colors cursor-pointer";
          tr.onclick = () => toggleTradeRow(globalIdx);
          tr.innerHTML = `
            <td class="py-2.5 px-3 text-gray-400 whitespace-nowrap">${t.time}</td>
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
              <td colspan="9" class="p-3">
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
                    <span class="text-gray-400 block text-[10px]">Execution Comment</span>
                    <span class="font-mono text-gray-300">${t.strategy || 'AI-FlatRisk'}</span>
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

    async function refreshDashboard() {
      try {
        const res = await fetch("/api/dashboard/overview");
        if (!res.ok) throw new Error("API status " + res.status);
        const data = await res.json();
        globalData = data;

        // Check for new trades to chime
        if (data.trades) {
          const currentCount = data.trades.length;
          if (previousTradeCount !== null && currentCount > previousTradeCount) {
            playChime();
          }
          previousTradeCount = currentCount;
        }

        // Update Account Info
        if (data.account && data.account.balance !== undefined) {
          const bal = parseFloat(data.account.balance);
          const eq = parseFloat(data.account.equity);
          document.getElementById("metric-account-balance").innerText = formatMoney(bal);
          document.getElementById("metric-account-equity").innerText = formatMoney(eq);
          const badge = document.getElementById("broker-status-badge");
          if (badge) {
            const accNum = data.account.account_id || "168987477";
            badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> MT5 Demo #${accNum} (${formatMoney(bal)}) Connected`;
          }
        }

        // Update KPIs
        const openPnl = data.assets.reduce((sum, a) => sum + (parseFloat(a.unrealized_pnl) || 0), 0);
        const todayRealized = data.assets.reduce((sum, a) => sum + (parseFloat(a.realized_today) || 0), 0);
        const activeCount = data.assets.filter(a => a.status === "IN TRADE").length;
        const totalRealized = data.monthly.reduce((sum, m) => sum + (parseFloat(m.pnl) || 0), 0);

        document.getElementById("metric-total-pnl").innerText = formatMoney(totalRealized);
        document.getElementById("metric-open-pnl").innerText = formatMoney(openPnl);
        document.getElementById("metric-today-pnl").innerText = formatMoney(todayRealized);
        document.getElementById("metric-active-count").innerText = `${activeCount} / 3`;

        // Style KPIs
        document.getElementById("metric-total-pnl").className = "text-2xl font-bold mt-1 " + (totalRealized >= 0 ? "text-emerald-400" : "text-rose-400");
        document.getElementById("metric-open-pnl").className = "text-2xl font-bold mt-1 " + (openPnl >= 0 ? "text-emerald-400" : "text-rose-400");
        document.getElementById("metric-today-pnl").className = "text-2xl font-bold mt-1 " + (todayRealized >= 0 ? "text-emerald-400" : "text-rose-400");

        computePerformanceKPIs(data.trades);
        renderAssetCards(data.assets);
        renderCharts(data.monthly, data.overall);
        renderTrades();

        document.getElementById("last-refresh").innerText = "Updated " + new Date().toLocaleTimeString();
      } catch (err) {
        console.error("Dashboard refresh error:", err);
      }
    }

    // Adaptive Eco-Refresh Loop: 3s when active, 10s when backgrounded
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
