import sys
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/scripts')
sys.path.insert(0, '/home/mcpuser/MCP/gold-trading-system/src')

from run_production_backtest_flawless import generate_flawless_candidates
from run_production_backtest import compute_deflated_sharpe_ratio
import numpy as np
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

thresholds = [0.00, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55]

portfolio_res = []
asset_res = {s: [] for s in all_cands}

for th in thresholds:
    combined_trades = []
    for s, cands in all_cands.items():
        # Strictly sequential: max 1 trade at a time per asset!
        seq = []
        last_exit = -1
        for c in cands:
            prob = c.get("oof_prob", 0.0)
            if th > 0.0:
                if np.isnan(prob) or prob < th:
                    continue
            if c["bar_idx"] <= last_exit:
                continue
            seq.append(c)
            last_exit = c["exit_bar_idx"]

        combined_trades.extend(seq)

        tot = len(seq)
        wins = [t for t in seq if t["pnl_r"] > 0]
        wr = (len(wins) / tot * 100) if tot > 0 else 0
        tot_r = sum(t["pnl_r"] for t in seq)
        tot_pnl = tot_r * 75.44
        gross_w = sum(t["pnl_r"] for t in wins)
        gross_l = abs(sum(t["pnl_r"] for t in seq if t["pnl_r"] <= 0))
        pf = gross_w / gross_l if gross_l > 0 else 999.0

        asset_res[s].append({
            "threshold": th, "trades": tot, "trades_per_week": round(tot / 104, 1),
            "win_rate": round(wr, 1), "pf": round(pf, 2), "net_r": round(tot_r, 1),
            "net_pnl": round(tot_pnl, 2)
        })

    # Portfolio combined
    combined_trades.sort(key=lambda x: x["date"])
    tot_p = len(combined_trades)
    wins_p = [t for t in combined_trades if t["pnl_r"] > 0]
    wr_p = (len(wins_p) / tot_p * 100) if tot_p > 0 else 0
    tot_r_p = sum(t["pnl_r"] for t in combined_trades)
    tot_pnl_p = tot_r_p * 75.44

    # Max Drawdown
    bal = 3000.0
    pk = bal
    m_dd = 0.0
    r_list = []
    for t in combined_trades:
        bal += t["pnl_r"] * 75.44
        pk = max(pk, bal)
        dd = pk - bal
        m_dd = max(m_dd, dd)
        r_list.append(t["pnl_r"])

    gross_w_p = sum(t["pnl_r"] for t in wins_p)
    gross_l_p = abs(sum(t["pnl_r"] for t in combined_trades if t["pnl_r"] <= 0))
    pf_p = gross_w_p / gross_l_p if gross_l_p > 0 else 999.0
    dsr = compute_deflated_sharpe_ratio(r_list, num_trials=50)

    portfolio_res.append({
        "threshold": th, "trades": tot_p, "trades_per_week": round(tot_p / 104, 1),
        "win_rate": round(wr_p, 1), "pf": round(pf_p, 2), "net_r": round(tot_r_p, 1),
        "net_pnl": round(tot_pnl_p, 2), "max_dd": round(m_dd, 2), "sharpe": round(dsr["annualized_sharpe"], 2)
    })

print("=" * 115)
print("REALISTIC SEQUENTIAL BACKTEST (1 POSITION AT A TIME PER ASSET, NO OVERLAPPING PYRAMIDING)")
print("=" * 115)
hdr = "{:<12} | {:<8} | {:<10} | {:<10} | {:<6} | {:<9} | {:<14} | {:<11} | {:<6}".format(
    "ML Score", "Trades", "Trades/Wk", "Win Rate", "PF", "Net R", "Net P&L ($)", "Max DD ($)", "Sharpe"
)
print(hdr)
print("-" * 115)
for r in portfolio_res:
    th_label = "NO ML (0.0)" if r["threshold"] == 0.0 else ">= {:.2f}".format(r["threshold"])
    row = "{:<12} | {:<8} | {:<10.1f} | {:>8.1f}% | {:>6.2f} | {:>+8.1f}R | ${:>12,.2f} | ${:>9,.2f} | {:>6.2f}".format(
        th_label, r["trades"], r["trades_per_week"], r["win_rate"], r["pf"],
        r["net_r"], r["net_pnl"], r["max_dd"], r["sharpe"]
    )
    print(row)
print("=" * 115)
