import collections
from datetime import datetime, timezone
import MetaTrader5 as mt5

PATH = r"C:\Program Files\XM Global MT5\terminal64.exe"
if not mt5.initialize(path=PATH, login=168987477, password="Shivaya@148", server="XMGlobal-MT5 2"):
    print("Failed to initialize MT5")
    exit(1)

info = mt5.account_info()
print("=== ACCOUNT SUMMARY ===")
print(f"Account Login : {info.login}")
print(f"Server        : {info.server}")
print(f"Currency      : {info.currency}")
print(f"Leverage      : 1:{info.leverage}")
print(f"Balance       : ${info.balance:,.2f}")
print(f"Equity        : ${info.equity:,.2f}")
print(f"Margin        : ${info.margin:,.2f}")
print(f"Free Margin   : ${info.margin_free:,.2f}")
print(f"Floating PnL  : ${info.profit:,.2f}")

deals = mt5.history_deals_get(datetime(2020, 1, 1), datetime.now())
if not deals:
    print("No deals found.")
    mt5.shutdown()
    exit(0)

deposits = 0.0
trade_deals = []
for d in deals:
    if d.type == 2:  # DEAL_TYPE_BALANCE
        deposits += d.profit
    else:
        trade_deals.append(d)

print(f"\nInitial/Total Deposits: ${deposits:,.2f}")

# Closing deals have entry DEAL_ENTRY_OUT (1) or DEAL_ENTRY_OUT_BY (2)
closing_deals = [d for d in trade_deals if d.entry in (1, 2)]

gross_profit = sum(d.profit for d in closing_deals if d.profit > 0)
gross_loss = sum(d.profit for d in closing_deals if d.profit < 0)
net_pnl = sum(d.profit for d in closing_deals)
total_comm = sum(d.commission for d in trade_deals)
total_swap = sum(d.swap for d in trade_deals)
net_with_fees = net_pnl + total_comm + total_swap

wins = [d for d in closing_deals if d.profit > 0]
losses = [d for d in closing_deals if d.profit < 0]
breakevens = [d for d in closing_deals if d.profit == 0]

first_time = datetime.fromtimestamp(deals[0].time, tz=timezone.utc)
last_time = datetime.fromtimestamp(deals[-1].time, tz=timezone.utc)

print("\n=== TRADING PERFORMANCE SUMMARY ===")
print(f"First Activity        : {first_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
print(f"Latest Activity       : {last_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
print(f"Total Deals Recorded  : {len(deals)}")
print(f"Total Closed Positions: {len(closing_deals)}")
if closing_deals:
    print(f"Winning Trades        : {len(wins)} ({len(wins)/len(closing_deals)*100:.1f}%)")
    print(f"Losing Trades         : {len(losses)} ({len(losses)/len(closing_deals)*100:.1f}%)")
    print(f"Breakeven Trades      : {len(breakevens)} ({len(breakevens)/len(closing_deals)*100:.1f}%)")
print(f"Gross Profit          : +${gross_profit:,.2f}")
print(f"Gross Loss            : -${abs(gross_loss):,.2f}")
pf_str = f"{abs(gross_profit/gross_loss):.2f}" if gross_loss != 0 else "N/A"
print(f"Profit Factor         : {pf_str}")
print(f"Net Realized PnL      : ${net_pnl:,.2f}")
print(f"Total Commissions     : ${total_comm:,.2f}")
print(f"Total Swap            : ${total_swap:,.2f}")
print(f"Net Total Return      : ${net_with_fees:,.2f}")

# Daily breakdown
daily_pnl = collections.defaultdict(float)
daily_trades = collections.defaultdict(int)
daily_wins = collections.defaultdict(int)
daily_losses = collections.defaultdict(int)

for d in closing_deals:
    day = datetime.fromtimestamp(d.time, tz=timezone.utc).strftime("%Y-%m-%d")
    daily_pnl[day] += d.profit
    daily_trades[day] += 1
    if d.profit > 0:
        daily_wins[day] += 1
    elif d.profit < 0:
        daily_losses[day] += 1

print("\n=== DAILY PnL BREAKDOWN ===")
for day in sorted(daily_pnl.keys()):
    pnl = daily_pnl[day]
    trades_count = daily_trades[day]
    w = daily_wins[day]
    l = daily_losses[day]
    sign = "+" if pnl >= 0 else ""
    print(f"{day} : {sign}${pnl:8.2f} | {trades_count:2d} trades (W: {w:2d}, L: {l:2d})")

# Symbol breakdown
symbol_pnl = collections.defaultdict(float)
symbol_trades = collections.defaultdict(int)
for d in closing_deals:
    symbol_pnl[d.symbol] += d.profit
    symbol_trades[d.symbol] += 1

print("\n=== SYMBOL BREAKDOWN ===")
for sym in sorted(symbol_pnl.keys()):
    pnl = symbol_pnl[sym]
    sign = "+" if pnl >= 0 else ""
    print(f"{sym} : {sign}${pnl:8.2f} across {symbol_trades[sym]} trades")

mt5.shutdown()
