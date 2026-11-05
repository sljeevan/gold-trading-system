import sys
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/scripts')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/src')

from run_production_backtest_flawless import generate_flawless_candidates, simulate_selection
from collections import defaultdict
import json

configs = [
    {"symbol": "XAUUSD", "sl_atr_mult": 1.5, "min_sl_dist": 8.00, "tsl_buffer": 4.50, "tp2_mult": 3.00, "be_offset": 0.50, "scale_out_r": 1.25, "session_filter": True, "adx_min_threshold": None},
    {"symbol": "BTCUSD", "sl_atr_mult": 1.5, "min_sl_dist": 500.00, "tsl_buffer": 350.00, "tp2_mult": 3.00, "be_offset": 30.00, "scale_out_r": 1.00, "session_filter": False, "adx_min_threshold": None},
    {"symbol": "ETHUSD", "sl_atr_mult": 1.5, "min_sl_dist": 20.00, "tsl_buffer": 10.00, "tp2_mult": 3.00, "be_offset": 2.00, "scale_out_r": 1.00, "session_filter": False, "adx_min_threshold": 22.0},
]

all_cands = {}
for cfg in configs:
    s = cfg["symbol"]
    cands = generate_flawless_candidates(
        symbol=s, sl_atr_mult=cfg["sl_atr_mult"], min_sl_dist=cfg["min_sl_dist"],
        tsl_buffer=cfg["tsl_buffer"], tp2_mult=cfg["tp2_mult"], be_offset=cfg["be_offset"],
        scale_out_r=cfg["scale_out_r"], session_filter=cfg["session_filter"],
        adx_min_threshold=cfg["adx_min_threshold"]
    )
    all_cands[s] = cands

res_noml = []
res_ml50 = []
for s, cands in all_cands.items():
    r0 = simulate_selection(cands, ml_threshold=0.0)
    r5 = simulate_selection(cands, ml_threshold=0.50)
    res_noml.extend(r0["selected_trades"])
    res_ml50.extend(r5["selected_trades"])

def get_mom(trades):
    m_dict = defaultdict(lambda: {"trades": 0, "wins": 0, "net_r": 0.0, "net_pnl": 0.0})
    for t in trades:
        m = t["date"][:7]
        m_dict[m]["trades"] += 1
        if t["pnl_r"] > 0:
            m_dict[m]["wins"] += 1
        m_dict[m]["net_r"] += t["pnl_r"]
        m_dict[m]["net_pnl"] += t["pnl_r"] * 75.44
    return dict(sorted(m_dict.items()))

mom_noml = get_mom(res_noml)
mom_ml50 = get_mom(res_ml50)

all_m = sorted(set(list(mom_noml.keys()) + list(mom_ml50.keys())))
print("=" * 115)
print("ZERO-FLAW MONTH-ON-MONTH PERFORMANCE (NOV 2024 - NOV 2026)")
print("=" * 115)
hdr = "{:<10} | {:<14} | {:<11} | {:<15} | {:<14} | {:<11} | {:<15}".format(
    "Month", "Trades (No ML)", "Win% (No ML)", "Net P&L (No ML)", "Trades (ML>=.5)", "Win% (ML>=.5)", "Net P&L (ML>=.5)"
)
print(hdr)
print("-" * 115)
for m in all_m:
    d0 = mom_noml.get(m, {"trades": 0, "wins": 0, "net_r": 0.0, "net_pnl": 0.0})
    d5 = mom_ml50.get(m, {"trades": 0, "wins": 0, "net_r": 0.0, "net_pnl": 0.0})
    wr0 = (d0["wins"] / d0["trades"] * 100) if d0["trades"] > 0 else 0
    wr5 = (d5["wins"] / d5["trades"] * 100) if d5["trades"] > 0 else 0
    row = "{:<10} | {:<14} | {:>10.1f}% | ${:>13,.2f} | {:<14} | {:>10.1f}% | ${:>13,.2f}".format(
        m, d0["trades"], wr0, d0["net_pnl"], d5["trades"], wr5, d5["net_pnl"]
    )
    print(row)
print("=" * 115)
