"""
Quick verification script for Gold Trading System.
"""
from gold_trader.core.data_feed import get_live_gold_snapshot
from gold_trader.core.strategy_engine import generate_gold_trade_setup
from gold_trader.core.multi_agent import run_gold_multi_agent_debate
from gold_trader.core.backtester import compare_gold_strategies
from gold_trader.brokers.paper_broker import PaperBroker

def run_test():
    print("=== 1. LIVE GOLD & MACRO SNAPSHOT ===")
    snap = get_live_gold_snapshot()
    for k, v in snap["market_data"].items():
        print(f"  {k}: {v}")

    print("\n=== 2. INSTITUTIONAL TRADE SETUP ===")
    setup = generate_gold_trade_setup("GC", 10000.0, 1.0)
    print("  Action:", setup.get("action"))
    print("  Bias:", setup.get("bias"))
    print("  Entry Price:", setup.get("entry_price"))
    print("  Stop Loss:", setup.get("stop_loss"))
    print("  Target 1:", setup.get("target_1"))
    print("  Suggested Lots:", setup.get("suggested_lot_size"))
    print("  Reasons:", setup.get("reasons"))

    print("\n=== 3. 3-AGENT CONSENSUS DEBATE ===")
    debate = run_gold_multi_agent_debate("GC")
    print("  Consensus Decision:", debate["consensus"]["decision"])
    print("  Confidence:", debate["consensus"]["confidence"])
    print("  Summary:", debate["consensus"]["summary"])

    print("\n=== 4. STRATEGY LEADERBOARD (GC Futures) ===")
    comp = compare_gold_strategies("GC", period="3mo", interval="1h")
    for s in comp["leaderboard"]:
        rank = s["rank"]
        strat = s["strategy"]
        ret = s["total_return_pct"]
        wr = s["win_rate_pct"]
        sharpe = s["sharpe_ratio"]
        print(f"  Rank {rank}: {strat} | Return: {ret}% | Win Rate: {wr}% | Sharpe: {sharpe}")

    print("\n=== 5. PAPER BROKER TEST ===")
    broker = PaperBroker()
    acct = broker.get_account_info()
    print(f"  Broker: {acct['broker']}, Balance: ${acct['balance']}, Equity: ${acct['equity']}")

if __name__ == "__main__":
    run_test()
