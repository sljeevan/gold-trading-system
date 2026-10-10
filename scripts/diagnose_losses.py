import collections
from datetime import datetime, timezone
import MetaTrader5 as mt5

PATH = r"C:\Program Files\XM Global MT5\terminal64.exe"
if not mt5.initialize(path=PATH, login=168987477, password="Shivaya@148", server="XMGlobal-MT5 2"):
    print("Failed to initialize MT5")
    exit(1)

deals = mt5.history_deals_get(datetime(2020, 1, 1), datetime.now())
if not deals:
    print("No deals found")
    mt5.shutdown()
    exit(0)

# Trade deals
trade_deals = [d for d in deals if d.type in (0, 1)]
closing_deals = [d for d in trade_deals if d.entry in (1, 2)]

# Map positions to get durations
pos_entries = {}
durations = []
for d in trade_deals:
    if d.entry == 0:  # ENTRY_IN
        pos_entries[d.position_id] = d.time
    elif d.entry in (1, 2) and d.position_id in pos_entries:
        open_time = pos_entries[d.position_id]
        dur = d.time - open_time
        durations.append(dur)

exit_comments = collections.Counter()
losses = []
wins = []

for cd in closing_deals:
    comm = cd.comment
    if "[sl" in comm.lower():
        tag = "SL Hit"
    elif "[tp" in comm.lower():
        tag = "TP Hit"
    elif "close" in comm.lower():
        tag = "Manual / AI Close"
    else:
        tag = comm if comm else "No Comment"
    exit_comments[tag] += 1

    if cd.profit > 0:
        wins.append(cd.profit)
    elif cd.profit < 0:
        losses.append(cd.profit)

avg_win = sum(wins) / len(wins) if wins else 0
avg_loss = sum(losses) / len(losses) if losses else 0
win_rate = len(wins) / len(closing_deals) * 100 if closing_deals else 0

# Streaks
max_consec_losses = 0
cur_consec_losses = 0
max_consec_wins = 0
cur_consec_wins = 0

for cd in closing_deals:
    if cd.profit < 0:
        cur_consec_losses += 1
        cur_consec_wins = 0
        if cur_consec_losses > max_consec_losses:
            max_consec_losses = cur_consec_losses
    elif cd.profit > 0:
        cur_consec_wins += 1
        cur_consec_losses = 0
        if cur_consec_wins > max_consec_wins:
            max_consec_wins = cur_consec_wins

# Duration stats (in seconds)
avg_dur_sec = sum(durations) / len(durations) if durations else 0
median_dur_sec = sorted(durations)[len(durations)//2] if durations else 0

print("=== LOSS DIAGNOSTICS ===")
print(f"Total Closed Trades : {len(closing_deals)}")
print(f"Win Rate            : {win_rate:.1f}% ({len(wins)} wins, {len(losses)} losses)")
print(f"Average Win         : +${avg_win:.2f}")
print(f"Average Loss        : -${abs(avg_loss):.2f}")
print(f"Reward-to-Risk Ratio: {avg_win / abs(avg_loss):.2f}:1" if avg_loss != 0 else "N/A")
expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
print(f"Expectancy per trade: ${expectancy:.2f} USD")
print(f"Max Win             : +${max(wins):.2f}" if wins else "")
print(f"Max Loss            : -${abs(min(losses)):.2f}" if losses else "")
print(f"Max Consecutive Loss: {max_consec_losses} losses in a row")
print(f"Max Consecutive Win : {max_consec_wins} wins in a row")
print(f"Avg Trade Duration  : {avg_dur_sec:.1f}s ({avg_dur_sec/60:.1f} mins)")
print(f"Median Trade Dur    : {median_dur_sec:.1f}s ({median_dur_sec/60:.1f} mins)")

print("\n=== EXIT TRIGGER BREAKDOWN ===")
for tag, count in exit_comments.most_common():
    pct = count / len(closing_deals) * 100
    print(f"{tag:20s}: {count:4d} ({pct:5.1f}%)")

# Hourly breakdown to check session performance
hourly_pnl = collections.defaultdict(float)
hourly_trades = collections.defaultdict(int)
hourly_wins = collections.defaultdict(int)
for cd in closing_deals:
    hour = datetime.fromtimestamp(cd.time, tz=timezone.utc).hour
    hourly_pnl[hour] += cd.profit
    hourly_trades[hour] += 1
    if cd.profit > 0:
        hourly_wins[hour] += 1

print("\n=== HOURLY UTC PERFORMANCE ===")
for h in range(24):
    if h in hourly_trades:
        cnt = hourly_trades[h]
        pnl = hourly_pnl[h]
        w = hourly_wins[h]
        wr = w / cnt * 100
        sign = "+" if pnl >= 0 else ""
        print(f"{h:02d}:00 UTC : {sign}${pnl:7.2f} | {cnt:3d} trades (WR: {wr:4.1f}%)")

mt5.shutdown()
