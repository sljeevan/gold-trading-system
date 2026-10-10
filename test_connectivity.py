"""
Connectivity Diagnostic Test for XM Broker Integration (MetaAPI, Native MT5, Market Feeds).
"""
import json
import ssl
import urllib.error
import urllib.request
from gold_trader.config import config
from gold_trader.core.data_feed import get_live_gold_snapshot

# Create SSL context that supports system certificates or ignores validation issues if in dev proxy
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def test_metaapi_connectivity():
    print("=" * 65)
    print("1. METAAPI CLOUD CONNECTIVITY & ACCOUNT STATUS")
    print("=" * 65)
    token = config.METAAPI_TOKEN
    demo_acct_id = config.METAAPI_DEMO_ACCOUNT_ID
    live_acct_id = config.METAAPI_LIVE_ACCOUNT_ID

    if not token:
        print("❌ METAAPI_TOKEN is empty in .env")
        return

    print(f"🔑 MetaAPI Token Length: {len(token)} chars")
    
    headers = {
        "auth-token": token,
        "Content-Type": "application/json",
        "User-Agent": "gold-trader-mcp/1.0",
    }

    # 1. Query Provisioning API to list all registered accounts
    prov_url = "https://mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai/users/current/accounts"
    print(f"\n📡 Querying MetaAPI Provisioning API...")
    accounts = []
    try:
        req = urllib.request.Request(prov_url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            accounts = json.loads(resp.read().decode("utf-8"))
            print(f"✅ Authenticated successfully! Found {len(accounts)} registered account(s):")
            for idx, a in enumerate(accounts, 1):
                acc_id = a.get("_id") or a.get("id")
                name = a.get("name")
                login = a.get("login")
                server = a.get("server")
                state = a.get("state")
                conn_status = a.get("connectionStatus")
                platform = a.get("platform")
                acct_type = a.get("type", "cloud")
                print(f"\n   [{idx}] Account ID: {acc_id}")
                print(f"       Account Name : {name}")
                print(f"       Login ID     : {login}")
                print(f"       Broker Server: {server}")
                print(f"       Platform     : {platform}")
                print(f"       State        : {state}")
                print(f"       Conn Status  : {conn_status}")
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP Error {e.code}: {e.read().decode('utf-8')}")
    except Exception as exc:
        print(f"❌ Connection error: {exc}")

    # 2. Check Demo Account State via Client API
    target_id = demo_acct_id
    if not target_id and accounts:
        target_id = accounts[0].get("_id") or accounts[0].get("id")

    if target_id and target_id != "your_metaapi_demo_account_id":
        print(f"\n📡 Testing Account Details for ID ({target_id})...")
        client_url = f"https://mt-client-api-v1.agiliumtrade.agiliumtrade.ai/users/current/accounts/{target_id}/accountInformation"
        try:
            req = urllib.request.Request(client_url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                print(f"✅ Live Account Information Connected & Active!")
                print(f"   Broker Server: {data.get('server')}")
                print(f"   Login        : {data.get('login')}")
                print(f"   Balance      : ${data.get('balance', 0.0):,.2f} {data.get('currency', 'USD')}")
                print(f"   Equity       : ${data.get('equity', 0.0):,.2f}")
                print(f"   Free Margin  : ${data.get('freeMargin', 0.0):,.2f}")
                print(f"   Margin Level : {data.get('marginLevel', 0.0)}%")
                print(f"   Leverage     : 1:{data.get('leverage', 100)}")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8')
            print(f"ℹ️ Client API response ({e.code}): {err_body}")
        except Exception as exc:
            print(f"ℹ️ Account response: {exc}")


def test_socket_bridge():
    print("=" * 65)
    print("1. FREE MT5 SOCKET BRIDGE CHECK (Option 3 - $0 Cost)")
    print("=" * 65)
    from gold_trader.brokers.socket_broker import SocketBroker
    broker = SocketBroker()
    print(f"📡 Target MT5 Bridge: tcp://{broker.host}:{broker.port}")
    info = broker.get_account_info()
    if "OFFLINE" in info["broker"]:
        print(f"ℹ️ Bridge Status: OFFLINE")
        print(f"   Note: To connect, open XM MT5 -> attach 'GoldTraderBridge.mq5' to any chart.")
    else:
        print(f"✅ Bridge Connected & Active!")
        print(f"   Account: {info['account_id']} | Broker: {info['broker']}")
        print(f"   Balance: ${info['balance']:,.2f} | Equity: ${info['equity']:,.2f}")


def test_native_mt5():
    print("\n" + "=" * 65)
    print("2. NATIVE METATRADER 5 LIBRARY CHECK (Windows / VPS)")
    print("=" * 65)
    try:
        import MetaTrader5 as mt5
        print(f"✅ MetaTrader5 package is installed (Version: {mt5.__version__})")
        init_ok = mt5.initialize()
        if init_ok:
            print("✅ MT5 Terminal successfully initialized.")
        else:
            print(f"ℹ️ MT5 initialize returned False (Reason: {mt5.last_error()})")
    except ImportError:
        print("ℹ️ MetaTrader5 native Windows library not installed (Expected on Linux).")
        print("   Use BROKER_MODE=socket with GoldTraderBridge.mq5 for $0 local connectivity.")


def test_market_feeds():
    print("\n" + "=" * 65)
    print("3. LIVE MARKET DATA & MACRO FEED CHECK")
    print("=" * 65)
    try:
        snap = get_live_gold_snapshot()
        print("✅ Market data feeds active:")
        for k, v in snap.get("market_data", {}).items():
            price = v.get("price", "N/A")
            change = v.get("change_pct", "N/A")
            print(f"   {k:20}: Price = {price} (Change: {change}%)")
    except Exception as exc:
        print(f"❌ Market data feed error: {exc}")


if __name__ == "__main__":
    test_socket_bridge()
    test_native_mt5()
    test_market_feeds()

