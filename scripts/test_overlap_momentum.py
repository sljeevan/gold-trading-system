"""
Test London/NY Overlap Momentum Strategy on 6-Month Gold Data
"""
import numpy as np
import sys
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/src')
from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import calc_ema, calc_atr, calc_rsi

def test():
    candles = fetch_gold_candles('XAUUSD', period='6mo', interval='1h')
    n = len(candles)
    highs = np.array([float(c['high']) for c in candles], dtype=float)
    lows = np.array([float(c['low']) for c in candles], dtype=float)
    closes = np.array([float(c['close']) for c in candles], dtype=float)
    opens = np.array([float(c['open']) for c in candles], dtype=float)
    dates = [c['date'] for c in candles]

    ema20 = calc_ema(closes.tolist(), 20)
    ema50 = calc_ema(closes.tolist(), 50)
    ema200 = calc_ema(closes.tolist(), 200)
    atr = calc_atr(highs.tolist(), lows.tolist(), closes.tolist(), 14)
    rsi = calc_rsi(closes.tolist(), 14)

    wins, losses = 0, 0
    gross_p, gross_l = 0.0, 0.0
    active_until = -1

    for i in range(200, n - 24):
        if i <= active_until:
            continue
        dt = dates[i]
        hour = int(dt.split('T' if 'T' in dt else ' ')[1].split(':')[0])
        if not (12 <= hour <= 16):
            continue

        e20, e50, e200, a, r = ema20[i], ema50[i], ema200[i], atr[i], rsi[i]
        if None in (e20, e50, e200, a, r):
            continue
        c, h, l, o = closes[i], highs[i], lows[i], opens[i]

        if c > e200 and e20 > e50 and c > e20 and (50 <= r <= 70) and c > o:
            sl = l - 1.2 * a
            tp = c + 1.2 * (c - sl)
            won = False
            bars = 0
            for k in range(i + 1, min(i + 24, n)):
                bars = k - i
                if lows[k] <= sl:
                    won = False
                    break
                if highs[k] >= tp:
                    won = True
                    break
            else:
                won = closes[min(i + 24, n - 1)] > c

            active_until = i + bars
            if won:
                wins += 1
                gross_p += 1.2 * 100
            else:
                losses += 1
                gross_l += 100

        elif c < e200 and e20 < e50 and c < e20 and (30 <= r <= 50) and c < o:
            sl = h + 1.2 * a
            tp = c - 1.2 * (sl - c)
            won = False
            bars = 0
            for k in range(i + 1, min(i + 24, n)):
                bars = k - i
                if (highs[k] + 0.35) >= sl:
                    won = False
                    break
                if (lows[k] + 0.35) <= tp:
                    won = True
                    break
            else:
                won = c > closes[min(i + 24, n - 1)]

            active_until = i + bars
            if won:
                wins += 1
                gross_p += 1.2 * 100
            else:
                losses += 1
                gross_l += 100

    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0
    pf = (gross_p / gross_l) if gross_l > 0 else 0
    print(f'London/NY Overlap Momentum (1.2R Target):')
    print(f'Trades: {total} | Wins: {wins} ({wr:.1f}%) | Losses: {losses} | Profit Factor: {pf:.2f} | Net P&L: +${gross_p - gross_l:,.2f}')

if __name__ == '__main__':
    test()
