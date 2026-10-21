"""
Standalone MT5 Socket Bridge Server running inside Wine Python.
Listens on TCP Port 8888 and executes trades natively via MetaTrader 5 SDK.
Compatible with SocketBroker protocol.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict

import MetaTrader5 as mt5

HOST = "127.0.0.1"
PORT = 8888
PATH = r"C:\Program Files\XM Global MT5\terminal64.exe"

def get_env_credentials():
    login = 168987477
    server = "XMGlobal-MT5 2"
    password = ""
    env_path = "/home/mcpuser/MCP/gold-trading-system/.env"
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("XM_DEMO_PASSWORD="):
                    password = line.strip().split("=", 1)[1].strip().strip("\"'")
                elif line.startswith("XM_DEMO_LOGIN="):
                    try:
                        login = int(line.strip().split("=", 1)[1].strip().strip("\"'"))
                    except:
                        pass
                elif line.startswith("XM_DEMO_SERVER="):
                    server = line.strip().split("=", 1)[1].strip().strip("\"'")
    return login, password, server

def ensure_mt5():
    login, password, server = get_env_credentials()
    if not mt5.initialize(path=PATH, login=login, password=password, server=server, timeout=60000):
        print(f"[{datetime.now()}] mt5.initialize failed: {mt5.last_error()}", flush=True)
        return False
    return True

def map_symbol(requested: str) -> str:
    """Find the exact gold symbol available on the active broker account."""
    requested_upper = (requested or "XAUUSD").upper()
    candidates = [requested_upper]
    if "GOLD" in requested_upper or "XAU" in requested_upper:
        candidates = ["GOLD", "GOLDmicro", "XAUUSD", "GOLD#", "GOLDm", "XAUUSDm", "XAUUSD#"]
    elif "BTC" in requested_upper:
        candidates = ["BTCUSD", "BTCUSD#", "BTCUSDm", "BTC"]
    elif "ETH" in requested_upper:
        candidates = ["ETHUSD", "ETHUSD#", "ETHUSDm", "ETH"]
    
    # Priority 1: Candidate with full trading permissions and live ticks
    for c in candidates:
        if mt5.symbol_select(c, True):
            info = mt5.symbol_info(c)
            tick = mt5.symbol_info_tick(c)
            if info and tick and info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                return c

    # Priority 2: Any candidate with live ticks
    for c in candidates:
        if mt5.symbol_select(c, True):
            tick = mt5.symbol_info_tick(c)
            if tick is not None:
                return c
    return requested_upper

def handle_client(conn: socket.socket, addr: Any):
    try:
        data = conn.recv(8192)
        if not data:
            return
        
        raw_text = data.decode("utf-8", errors="replace").strip()
        req = json.loads(raw_text)
        
        # Support both 'command' and 'action'
        cmd = req.get("command", req.get("action", "")).upper()
        
        ensure_mt5()
        resp: Dict[str, Any] = {"status": "ERROR"}

        if cmd in ("PING", "STATUS", "HEALTH"):
            resp = {"status": "SUCCESS", "message": "MT5 Bridge Online"}

        elif cmd in ("SYMBOL_INFO", "GET_TICK", "TICK"):
            raw_sym = req.get("symbol", "XAUUSD")
            symbol = map_symbol(raw_sym)
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
            tick = mt5.symbol_info_tick(symbol)
            if not info or not tick:
                resp = {"status": "ERROR", "message": f"Symbol {symbol} not found or no tick"}
            else:
                resp = {
                    "status": "SUCCESS",
                    "symbol": symbol,
                    "bid": float(tick.bid),
                    "ask": float(tick.ask),
                    "last": float(tick.last) if tick.last else float(tick.bid),
                    "spread": int(info.spread),
                    "digits": int(info.digits),
                    "point": float(info.point),
                    "stops_level": int(info.trade_stops_level),
                    "trade_execution": int(getattr(info, "trade_exemode", getattr(info, "execution_mode", 0))),
                    "trade_mode": int(info.trade_mode),
                    "volume_min": float(info.volume_min),
                    "volume_max": float(info.volume_max),
                    "volume_step": float(info.volume_step),
                }

        elif cmd in ("ACCOUNT_INFO", "GET_ACCOUNT", "ACCOUNT"):
            info = mt5.account_info()
            if info:
                resp = {
                    "status": "SUCCESS",
                    "login": str(info.login),
                    "balance": float(info.balance),
                    "equity": float(info.equity),
                    "margin": float(info.margin),
                    "free_margin": float(info.margin_free),
                    "margin_level_pct": float(info.margin_level) if info.margin_level else 0.0,
                    "leverage": int(info.leverage),
                    "currency": info.currency,
                    "server": info.server,
                    "company": info.company,
                    "trade_mode": "DEMO" if info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else "LIVE",
                    "trade_allowed": bool(info.trade_allowed),
                }
            else:
                resp = {"status": "ERROR", "message": f"Failed to get account info: {mt5.last_error()}"}

        elif cmd in ("GET_POSITIONS", "POSITIONS", "LIST_POSITIONS"):
            filter_sym = req.get("symbol", "")
            if filter_sym:
                filter_sym = map_symbol(filter_sym)
                positions = mt5.positions_get(symbol=filter_sym)
            else:
                positions = mt5.positions_get()
                
            pos_list = []
            if positions:
                for p in positions:
                    pos_list.append({
                        "ticket": str(p.ticket),
                        "symbol": p.symbol,
                        "side": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                        "lots": float(p.volume),
                        "open_price": float(p.price_open),
                        "current_price": float(p.price_current),
                        "sl": float(p.sl) if p.sl else None,
                        "tp": float(p.tp) if p.tp else None,
                        "pnl": float(p.profit),
                        "open_time": datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
                        "comment": p.comment,
                    })
            resp = {"status": "SUCCESS", "positions": pos_list}

        elif cmd in ("GET_RATES", "RATES", "GET_CANDLES", "CANDLES"):
            raw_sym = req.get("symbol", "XAUUSD")
            symbol = map_symbol(raw_sym)
            timeframe_str = str(req.get("timeframe", "15m")).lower()
            count = int(req.get("count", 100))

            tf_map = {
                "1m": mt5.TIMEFRAME_M1,
                "5m": mt5.TIMEFRAME_M5,
                "15m": mt5.TIMEFRAME_M15,
                "30m": mt5.TIMEFRAME_M30,
                "1h": mt5.TIMEFRAME_H1,
                "4h": mt5.TIMEFRAME_H4,
                "1d": mt5.TIMEFRAME_D1,
            }
            tf = tf_map.get(timeframe_str, mt5.TIMEFRAME_M15)

            mt5.symbol_select(symbol, True)
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            if rates is not None and len(rates) > 0:
                candles_out = []
                for r in rates:
                    dt_str = datetime.fromtimestamp(int(r["time"]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                    candles_out.append({
                        "date": dt_str,
                        "open": round(float(r["open"]), 2),
                        "high": round(float(r["high"]), 2),
                        "low": round(float(r["low"]), 2),
                        "close": round(float(r["close"]), 2),
                        "volume": float(r["tick_volume"]),
                    })
                resp = {"status": "SUCCESS", "symbol": symbol, "timeframe": timeframe_str, "candles": candles_out}
            else:
                resp = {"status": "ERROR", "message": f"Failed to get rates for {symbol}: {mt5.last_error()}"}

        elif cmd in ("GET_ORDERS", "PENDING_ORDERS", "GET_PENDING_ORDERS"):
            filter_sym = req.get("symbol", "")
            if filter_sym:
                filter_sym = map_symbol(filter_sym)
                orders = mt5.orders_get(symbol=filter_sym)
            else:
                orders = mt5.orders_get()

            ord_list = []
            if orders:
                for o in orders:
                    ord_list.append({
                        "ticket": str(o.ticket),
                        "symbol": o.symbol,
                        "side": "BUY" if o.type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_BUY_LIMIT) else "SELL",
                        "order_type": "LIMIT" if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT) else "OTHER",
                        "lots": float(o.volume_initial),
                        "price": float(o.price_open),
                        "sl": float(o.sl) if o.sl else None,
                        "tp": float(o.tp) if o.tp else None,
                        "time_setup": datetime.fromtimestamp(o.time_setup, tz=timezone.utc).isoformat(),
                        "comment": o.comment,
                    })
            resp = {"status": "SUCCESS", "orders": ord_list}

        elif cmd in ("CANCEL_ORDER", "DELETE_ORDER", "REMOVE_ORDER"):
            ticket = int(req.get("ticket", 0))
            if ticket > 0:
                del_req = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": ticket,
                }
                res = mt5.order_send(del_req)
                if res and res.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
                    resp = {"status": "SUCCESS", "message": f"Cancelled order #{ticket}"}
                else:
                    err = mt5.last_error()
                    resp = {"status": "ERROR", "message": f"Failed to cancel order #{ticket}: {err}"}
            else:
                resp = {"status": "ERROR", "message": "Missing or invalid ticket"}

        elif cmd in ("PLACE_ORDER", "TRADE", "ORDER", "BUY", "SELL"):
            raw_sym = req.get("symbol", "XAUUSD")
            symbol = map_symbol(raw_sym)
            
            # Action can be in 'action' or 'side' or 'order_type' or inferred from cmd
            side_str = req.get("action", req.get("side", req.get("order_type", cmd))).upper()
            if side_str not in ("BUY", "SELL"):
                side_str = "BUY"
                
            lots = float(req.get("lots", req.get("volume", 0.01)))
            sl = float(req.get("sl", 0.0)) if req.get("sl") else 0.0
            tp = float(req.get("tp", 0.0)) if req.get("tp") else 0.0
            comment = req.get("comment", "GoldTrader-AI")
            
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
            tick = mt5.symbol_info_tick(symbol)
            if not tick or not info:
                resp = {"status": "ERROR", "message": f"Cannot get tick/info for symbol {symbol}"}
            else:
                # Clamp lot size to broker specifications
                if lots < info.volume_min:
                    lots = info.volume_min
                elif lots > info.volume_max:
                    lots = info.volume_max
                if info.volume_step > 0:
                    lots = round(round(lots / info.volume_step) * info.volume_step, 2)

                order_type_str = str(req.get("order_type", req.get("type", "MARKET"))).upper()
                req_price = float(req.get("price", 0.0)) if req.get("price") else 0.0
                is_limit = (order_type_str in ("LIMIT", "BUY_LIMIT", "SELL_LIMIT")) and (req_price > 0)

                if is_limit:
                    pending_type = mt5.ORDER_TYPE_BUY_LIMIT if side_str == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT
                    target_price = round(req_price, info.digits)
                    target_sl = round(sl, info.digits) if sl > 0 else 0.0
                    target_tp = round(tp, info.digits) if tp > 0 else 0.0

                    trade_req = {
                        "action": mt5.TRADE_ACTION_PENDING,
                        "symbol": symbol,
                        "volume": lots,
                        "type": pending_type,
                        "price": target_price,
                        "sl": target_sl,
                        "tp": target_tp,
                        "deviation": 20,
                        "magic": 987001,
                        "comment": comment,
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_RETURN,
                    }
                    result = mt5.order_send(trade_req)
                    if result and result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
                        ticket = result.order
                        resp = {
                            "status": "SUCCESS",
                            "order_type": "LIMIT",
                            "ticket": str(ticket),
                            "symbol": symbol,
                            "action": side_str,
                            "lots": lots,
                            "price": target_price,
                            "sl": target_sl if target_sl > 0 else None,
                            "tp": target_tp if target_tp > 0 else None,
                            "message": f"XM Limit Order #{ticket} placed: {side_str} {lots} lots @ {target_price}",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    else:
                        err_code = getattr(result, "retcode", mt5.last_error())
                        err_msg = getattr(result, "comment", str(mt5.last_error()))
                        resp = {"status": "ERROR", "message": f"Limit order failed on XM: {err_msg} (retcode: {err_code})"}
                else:
                    order_type = mt5.ORDER_TYPE_BUY if side_str == "BUY" else mt5.ORDER_TYPE_SELL
                    price = tick.ask if side_str == "BUY" else tick.bid
                    
                    # Determine supported filling type
                    filling_type = mt5.ORDER_FILLING_IOC
                    if hasattr(info, "filling_mode"):
                        if info.filling_mode & 1:  # FOK
                            filling_type = mt5.ORDER_FILLING_FOK
                        elif info.filling_mode & 2:  # IOC
                            filling_type = mt5.ORDER_FILLING_IOC
                        else:
                            filling_type = mt5.ORDER_FILLING_RETURN

                    # Check if market execution mode (trade_exemode == 2; XM brokers forbid SL/TP in initial DEAL request)
                    trade_exemode = getattr(info, "trade_exemode", getattr(info, "execution_mode", 0))
                    is_market_exec = (trade_exemode == 2)
                    send_sl = 0.0 if is_market_exec else sl
                    send_tp = 0.0 if is_market_exec else tp

                    trade_req = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": lots,
                        "type": order_type,
                        "price": price,
                        "sl": send_sl,
                        "tp": send_tp,
                        "deviation": 20,
                        "magic": 987001,
                        "comment": comment,
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": filling_type,
                    }
                    result = mt5.order_send(trade_req)
                    
                    # Fallback: If initial order failed with 10016 (Invalid Stops), retry without SL/TP
                    if (not result or getattr(result, "retcode", None) == 10016) and (send_sl > 0 or send_tp > 0):
                        trade_req["sl"] = 0.0
                        trade_req["tp"] = 0.0
                        result = mt5.order_send(trade_req)

                    if result and result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
                        ticket = result.order
                        filled_price = float(result.price) if result.price else price
                        applied_sl = None
                        applied_tp = None

                        # If SL or TP were requested, apply them via TRADE_ACTION_SLTP with stops_level validation
                        if sl > 0 or tp > 0:
                            time.sleep(0.15)
                            cur_tick = mt5.symbol_info_tick(symbol) or tick
                            cur_info = mt5.symbol_info(symbol) or info
                            # Ensure buffer beyond broker stops_level and spread
                            min_pts = max(cur_info.trade_stops_level, cur_info.spread, 30)
                            min_distance = min_pts * cur_info.point
                            
                            target_sl = sl
                            target_tp = tp
                            
                            # Validate and clamp stop distance against current price
                            if side_str == "BUY":
                                if target_sl > 0 and (cur_tick.bid - target_sl) < min_distance:
                                    target_sl = round(cur_tick.bid - min_distance, cur_info.digits)
                                if target_tp > 0 and (target_tp - cur_tick.bid) < min_distance:
                                    target_tp = round(cur_tick.bid + min_distance, cur_info.digits)
                            else:  # SELL
                                if target_sl > 0 and (target_sl - cur_tick.ask) < min_distance:
                                    target_sl = round(cur_tick.ask + min_distance, cur_info.digits)
                                if target_tp > 0 and (cur_tick.ask - target_tp) < min_distance:
                                    target_tp = round(cur_tick.ask - min_distance, cur_info.digits)

                            # Find open position ticket
                            positions = mt5.positions_get(symbol=symbol)
                            pos_ticket = ticket
                            if positions:
                                matching = [p for p in positions if p.magic == 987001 or p.ticket == ticket]
                                if matching:
                                    pos_ticket = matching[-1].ticket

                            mod_req = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": pos_ticket,
                                "symbol": symbol,
                                "sl": target_sl if target_sl > 0 else 0.0,
                                "tp": target_tp if target_tp > 0 else 0.0,
                            }
                            mod_res = mt5.order_send(mod_req)
                            if mod_res and mod_res.retcode == mt5.TRADE_RETCODE_DONE:
                                applied_sl = target_sl if target_sl > 0 else None
                                applied_tp = target_tp if target_tp > 0 else None
                            else:
                                err_mod = mod_res.comment if mod_res else str(mt5.last_error())
                                print(f"[{datetime.now()}] Position #{pos_ticket} opened, but SL/TP attach: {err_mod}", flush=True)

                        resp = {
                            "status": "SUCCESS",
                            "ticket": str(ticket),
                            "symbol": symbol,
                            "action": side_str,
                            "lots": lots,
                            "price": filled_price,
                            "sl": applied_sl,
                            "tp": applied_tp,
                            "message": f"XM {side_str} {lots} lots of {symbol} executed successfully. Ticket #{ticket}",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    else:
                        err_msg = result.comment if result else str(mt5.last_error())
                        resp = {"status": "ERROR", "message": f"Order failed on XM: {err_msg} (retcode: {getattr(result, 'retcode', None)})"}

        elif cmd in ("CLOSE_POSITION", "CLOSE"):
            ticket = int(req.get("ticket", 0))
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                resp = {"status": "ERROR", "message": f"Position #{ticket} not found on XM"}
            else:
                pos = positions[0]
                close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                tick = mt5.symbol_info_tick(pos.symbol)
                close_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
                
                req_lots = float(req.get("lots", req.get("volume", pos.volume)))
                close_vol = round(min(max(req_lots, 0.01), pos.volume), 2)

                close_req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": close_vol,
                    "type": close_type,
                    "position": pos.ticket,
                    "price": close_price,
                    "deviation": 20,
                    "magic": 987001,
                    "comment": "Close-GoldTrader-AI",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                res = mt5.order_send(close_req)
                if res and res.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
                    resp = {
                        "status": "SUCCESS",
                        "ticket": str(ticket),
                        "symbol": pos.symbol,
                        "action": "CLOSE",
                        "lots": float(close_vol),
                        "remaining_lots": round(pos.volume - close_vol, 2),
                        "price": float(close_price),
                        "message": f"Position #{ticket} closed {close_vol} lots on XM at {close_price}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                else:
                    err_msg = res.comment if res else str(mt5.last_error())
                    resp = {"status": "ERROR", "message": f"Failed to close #{ticket}: {err_msg}"}

        elif cmd in ("MODIFY_POSITION", "MODIFY"):
            ticket = int(req.get("ticket", 0))
            sl = float(req.get("sl", 0.0)) if req.get("sl") else 0.0
            tp = float(req.get("tp", 0.0)) if req.get("tp") else 0.0
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                resp = {"status": "ERROR", "message": f"Position #{ticket} not found on XM"}
            else:
                mod_req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": ticket,
                    "sl": sl,
                    "tp": tp,
                }
                res = mt5.order_send(mod_req)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    resp = {
                        "status": "SUCCESS",
                        "ticket": str(ticket),
                        "sl": sl if sl else None,
                        "tp": tp if tp else None,
                        "message": f"Position #{ticket} modified on XM with SL={sl} TP={tp}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                else:
                    err_msg = res.comment if res else str(mt5.last_error())
                    resp = {"status": "ERROR", "message": f"Failed to modify #{ticket}: {err_msg}"}

        elif cmd in ("HISTORY_DEALS", "DEAL_HISTORY", "GET_DEAL_HISTORY", "GET_DEALS"):
            ticket = int(req.get("ticket") or req.get("position") or 0)
            if ticket > 0:
                deals = mt5.history_deals_get(position=ticket)
                if not deals:
                    deals = mt5.history_deals_get(ticket=ticket)

                if deals:
                    deal_records = []
                    total_profit = 0.0
                    total_commission = 0.0
                    total_swap = 0.0
                    entry_price = None
                    close_price = None
                    close_time = None
                    close_comment = ""

                    for d in deals:
                        deal_dt = datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat()
                        d_comm = float(getattr(d, "commission", 0.0))
                        d_swap = float(getattr(d, "swap", 0.0))
                        d_prof = float(d.profit)
                        total_profit += d_prof
                        total_commission += d_comm
                        total_swap += d_swap

                        deal_records.append({
                            "ticket": str(d.ticket),
                            "order": str(d.order),
                            "position_id": str(d.position_id),
                            "time": deal_dt,
                            "type": "BUY" if d.type == 0 else "SELL",
                            "entry": int(d.entry),
                            "volume": float(d.volume),
                            "price": float(d.price),
                            "profit": d_prof,
                            "commission": d_comm,
                            "swap": d_swap,
                            "comment": d.comment,
                            "symbol": d.symbol,
                        })

                        if d.entry == 0:
                            entry_price = float(d.price)
                        elif d.entry in (1, 2, 3):
                            close_price = float(d.price)
                            close_time = deal_dt
                            close_comment = d.comment

                    resp = {
                        "status": "SUCCESS",
                        "position_id": str(ticket),
                        "total_profit": round(total_profit, 2),
                        "total_commission": round(total_commission, 2),
                        "total_swap": round(total_swap, 2),
                        "net_pnl": round(total_profit + total_commission + total_swap, 2),
                        "entry_price": entry_price,
                        "close_price": close_price,
                        "close_time": close_time,
                        "close_comment": close_comment,
                        "deals": deal_records,
                    }
                else:
                    resp = {"status": "ERROR", "message": f"No deals found for position #{ticket}"}
            else:
                days = int(req.get("days", 7))
                date_from = datetime.now(timezone.utc) - timedelta(days=days)
                date_to = datetime.now(timezone.utc) + timedelta(days=1)
                deals = mt5.history_deals_get(date_from, date_to)
                deal_list = []
                if deals:
                    for d in deals:
                        deal_list.append({
                            "ticket": str(d.ticket),
                            "position_id": str(d.position_id),
                            "order": str(d.order),
                            "time": datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat(),
                            "type": "BUY" if d.type == 0 else "SELL",
                            "entry": int(d.entry),
                            "volume": float(d.volume),
                            "price": float(d.price),
                            "profit": float(d.profit),
                            "commission": float(getattr(d, "commission", 0.0)),
                            "swap": float(getattr(d, "swap", 0.0)),
                            "comment": d.comment,
                            "symbol": d.symbol,
                        })
                resp = {"status": "SUCCESS", "count": len(deal_list), "deals": deal_list}

        else:
            resp = {"status": "ERROR", "message": f"Unknown command: {cmd}"}

        conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
    except Exception as e:
        err_resp = {"status": "ERROR", "message": str(e)}
        try:
            conn.sendall((json.dumps(err_resp) + "\n").encode("utf-8"))
        except:
            pass
    finally:
        conn.close()

def run_server():
    print(f"Starting XM MT5 Bridge Server on {HOST}:{PORT}...", flush=True)
    if not ensure_mt5():
        print("Warning: Initial MT5 connection attempt failed. Will retry on request.", flush=True)
    else:
        info = mt5.account_info()
        print(f"XM MT5 Terminal Connected! Account: {info.login} | Balance: ${info.balance} | Server: {info.server}", flush=True)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(10)
    print(f"Bridge Server listening on TCP port {PORT} 24/7...", flush=True)

    while True:
        try:
            conn, addr = server.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
        except Exception as e:
            print(f"Socket accept error: {e}", flush=True)
            time.sleep(1)

if __name__ == "__main__":
    run_server()
