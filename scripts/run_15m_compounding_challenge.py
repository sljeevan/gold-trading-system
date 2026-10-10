import numpy as np
from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import calc_ema, calc_supertrend, calc_rsi
from gold_trader.core.compounding_ladder import get_ladder_level_info

candles = fetch_gold_candles("XAUUSD", period="60d", interval="15m")
n = len(candles)
highs = np.array([float(c["high"]) for c in candles], dtype=float)
lows = np.array([float(c["low"]) for c in candles], dtype=float)
closes = np.array([float(c["close"]) for c in candles], dtype=float)
opens = np.array([float(c["open"]) for c in candles], dtype=float)
dates = [c["date"] for c in candles]

ema9 = calc_ema(closes.tolist(), 9)
ema21 = calc_ema(closes.tolist(), 21)
ema50 = calc_ema(closes.tolist(), 50)
st = calc_supertrend(highs.tolist(), lows.tolist(), closes.tolist(), 10, 3.0)
rsi = calc_rsi(closes.tolist(), 14)

disc = 0.80 # 8-pip limit discount ($0.80)
level = 1
balance = 20.0
peak_level = 1
peak_balance = 20.0
completions = 0
reloads = 0
wins, losses, scratches = 0, 0, 0
act = -1
completion_history = []

for i in range(200, n - 36):
    if i <= act: continue
    dt = dates[i]
    hour = int(dt.split("T" if "T" in dt else " ")[1].split(":")[0])
    if not (7 <= hour <= 19): continue
    
    e9, e21, e50 = ema9[i], ema21[i], ema50[i]
    st_d = st["direction"][i]
    r = rsi[i]
    if None in (e9, e21, e50, st_d, r): continue
    c, h, l, o = closes[i], highs[i], lows[i], opens[i]
    
    dir_ = None
    if c > e50 and e9 > e21 and (l <= e9 and c >= e21) and c >= o and (45 <= r <= 65):
        dir_ = "BUY"
    elif c < e50 and e9 < e21 and (h >= e9 and c <= e21) and c <= o and (35 <= r <= 55):
        dir_ = "SELL"
    if not dir_: continue
    
    k_fill = i + 1
    limit_p = c - disc if dir_ == "BUY" else c + disc
    if dir_ == "BUY" and lows[k_fill] <= limit_p: fill_p = limit_p
    elif dir_ == "SELL" and highs[k_fill] >= limit_p: fill_p = limit_p
    else: continue
    
    lvl_info = get_ladder_level_info(level)
    eff_entry = fill_p + 0.35 if dir_ == "BUY" else fill_p
    tp = eff_entry + 2.0 if dir_ == "BUY" else eff_entry - 2.0 - 0.35
    sl = eff_entry - 1.54 if dir_ == "BUY" else eff_entry + 1.54 + 0.35
    be_price = eff_entry + 1.0 if dir_ == "BUY" else eff_entry - 1.0
    
    won, lost, scratched = False, False, False
    is_be = False
    bars = 0
    for k in range(k_fill, min(k_fill + 24, n)):
        bars = k - i
        o_k, h_k, l_k, c_k = opens[k], highs[k], lows[k], closes[k]
        curr_sl = eff_entry if is_be else sl
        if dir_ == "BUY":
            if c_k >= o_k:
                if l_k <= curr_sl:
                    if is_be: scratched = True
                    else: lost = True
                    break
                if h_k >= be_price: is_be = True
                curr_sl = eff_entry if is_be else sl
                if h_k >= tp: won = True; break
            else:
                if h_k >= tp: won = True; break
                if h_k >= be_price: is_be = True
                curr_sl = eff_entry if is_be else sl
                if l_k <= curr_sl:
                    if is_be: scratched = True
                    else: lost = True
                    break
        else:
            if c_k <= o_k:
                if h_k >= curr_sl:
                    if is_be: scratched = True
                    else: lost = True
                    break
                if l_k <= be_price: is_be = True
                curr_sl = eff_entry if is_be else sl
                if l_k <= tp: won = True; break
            else:
                if l_k <= tp: won = True; break
                if l_k <= be_price: is_be = True
                curr_sl = eff_entry if is_be else sl
                if h_k >= curr_sl:
                    if is_be: scratched = True
                    else: lost = True
                    break
    act = i + bars
    
    if won:
        wins += 1
        balance += lvl_info["profit_dollars"]
        if level == 30:
            completions += 1
            completion_history.append({"date": dt, "trade_no": wins+losses, "balance": balance})
            level = 1
            balance = 20.0
        else:
            level += 1
            peak_level = max(peak_level, level)
    elif scratched:
        scratches += 1
    elif lost:
        losses += 1
        balance -= lvl_info["risk_dollars"]
        if level > 1:
            level -= 1
        else:
            if balance < 5.0:
                reloads += 1
                balance = 20.0
                level = 1
    peak_balance = max(peak_balance, balance)

print(f"=== 15M HIGH-FREQUENCY 30-LEVEL COMPOUNDING RUN (60 Days) ===")
print(f"Executed: {wins+losses+scratches} | Wins: {wins} | Losses: {losses} | Scratches: {scratches}")
print(f"Win Rate: {wins/(wins+losses)*100:.1f}% | Trade Frequency: {(wins+losses+scratches)/42.0:.1f} trades/day")
print(f"Peak Level: Level {peak_level} | Peak Balance: ${peak_balance:,.2f}")
print(f"Level 30 Completions: {completions} | Reloads at L1: {reloads}")
for c in completion_history:
    print(f"  ⭐ Level 30 Achieved on {c['date']} (Trade #{c['trade_no']}) with ${c['balance']:,.2f}")
print(f"Current Level at End: Level {level} | Balance: ${balance:,.2f}")
