"""
Configuration settings for the Gold Trading System.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _load_env_file() -> None:
    """Auto-load variables from .env if present in current or project directory."""
    possible_paths = [
        os.path.join(os.getcwd(), ".env"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")),
    ]
    for path in possible_paths:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
                break
            except Exception:
                pass

_load_env_file()



@dataclass(frozen=True)
class Settings:
    # Broker Configuration: "paper" | "xm_mt5" | "metaapi"
    BROKER_MODE: str = os.getenv("BROKER_MODE", "paper")
    
    # Active Account Type: "demo" | "live" (default "demo" for safety)
    XM_ACCOUNT_TYPE: str = os.getenv("XM_ACCOUNT_TYPE", "demo").lower().strip()
    
    # XM Broker MT5 Settings - Generic / Legacy Fallback
    XM_LOGIN: int = int(os.getenv("XM_LOGIN", "0"))
    XM_PASSWORD: str = os.getenv("XM_PASSWORD", "")
    XM_SERVER: str = os.getenv("XM_SERVER", "XMGlobal-MT5")
    XM_PATH: str = os.getenv("XM_PATH", "")

    # XM Demo Account Profile
    XM_DEMO_LOGIN: int = int(os.getenv("XM_DEMO_LOGIN", os.getenv("XM_LOGIN", "0")))
    XM_DEMO_PASSWORD: str = os.getenv("XM_DEMO_PASSWORD", os.getenv("XM_PASSWORD", ""))
    XM_DEMO_SERVER: str = os.getenv("XM_DEMO_SERVER", "XMGlobal-MT5")

    # XM Live Account Profile
    XM_LIVE_LOGIN: int = int(os.getenv("XM_LIVE_LOGIN", os.getenv("XM_LOGIN", "0")))
    XM_LIVE_PASSWORD: str = os.getenv("XM_LIVE_PASSWORD", os.getenv("XM_PASSWORD", ""))
    XM_LIVE_SERVER: str = os.getenv("XM_LIVE_SERVER", "XMGlobal-MT5")

    # Free Open-Source MT5 Socket Bridge Settings (Option 3)
    MT5_BRIDGE_HOST: str = os.getenv("MT5_BRIDGE_HOST", "127.0.0.1")
    MT5_BRIDGE_PORT: int = int(os.getenv("MT5_BRIDGE_PORT", "8888"))
    MT5_BRIDGE_TIMEOUT: float = float(os.getenv("MT5_BRIDGE_TIMEOUT", "5.0"))

    # MetaAPI Credentials (for Cloud / Linux / Docker deployment)
    METAAPI_TOKEN: str = os.getenv("METAAPI_TOKEN", "")
    METAAPI_ACCOUNT_ID: str = os.getenv("METAAPI_ACCOUNT_ID", "")
    METAAPI_DEMO_ACCOUNT_ID: str = os.getenv("METAAPI_DEMO_ACCOUNT_ID", os.getenv("METAAPI_ACCOUNT_ID", ""))
    METAAPI_LIVE_ACCOUNT_ID: str = os.getenv("METAAPI_LIVE_ACCOUNT_ID", os.getenv("METAAPI_ACCOUNT_ID", ""))

    # Telegram Notifications Settings
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    TELEGRAM_NOTIFICATIONS_ENABLED: bool = (
        os.getenv("TELEGRAM_NOTIFICATIONS_ENABLED", "true").lower() in ("true", "1", "yes")
    )

    # Trading & Risk Parameters
    DEFAULT_SYMBOL: str = os.getenv("DEFAULT_SYMBOL", "XAUUSD")
    DEFAULT_RISK_PER_TRADE_PCT: float = float(os.getenv("DEFAULT_RISK_PER_TRADE_PCT", "1.0"))
    DEFAULT_TIMEFRAME: str = os.getenv("DEFAULT_TIMEFRAME", "1h")
    DEFAULT_MAX_SPREAD_PIPS: float = float(os.getenv("DEFAULT_MAX_SPREAD_PIPS", "3.5"))

    # Database
    PAPER_DB_PATH: str = os.path.expanduser(
        os.getenv("PAPER_DB_PATH", "~/.gold_trading_data/paper_trading.db")
    )

    # Gold Symbol Mappings across data sources (Yahoo Finance benchmark tickers)
    SYMBOL_MAP = {
        "XAUUSD": "GC=F",           # COMEX Gold / Spot Proxy on Yahoo
        "GOLD": "GC=F",
        "GC": "GC=F",               # COMEX Gold Futures
        "GLD": "GLD",               # SPDR Gold Shares ETF
        "IAU": "IAU",               # iShares Gold Trust
        "GOLDBEES": "GOLDBEES.NS",  # Nippon India Gold ETF (NSE)
        "DXY": "DX-Y.NYB",          # US Dollar Index
        "US10Y": "^TNX",            # US 10-Year Treasury Yield
        "USDINR": "INR=X",          # USD to INR Exchange Rate
        "BTCUSD": "BTC-USD",        # Bitcoin Spot USD
        "BTC": "BTC-USD",
        "BTC-USD": "BTC-USD",
    }

    def get_xm_credentials(self, account_type: Optional[str] = None) -> tuple[int, str, str]:
        """Returns (login, password, server) for the requested or configured account type."""
        acct_type = (account_type or self.XM_ACCOUNT_TYPE).lower().strip()
        if acct_type == "live":
            login = self.XM_LIVE_LOGIN or self.XM_LOGIN
            password = self.XM_LIVE_PASSWORD or self.XM_PASSWORD
            server = self.XM_LIVE_SERVER or self.XM_SERVER
        else:
            login = self.XM_DEMO_LOGIN or self.XM_LOGIN
            password = self.XM_DEMO_PASSWORD or self.XM_PASSWORD
            server = self.XM_DEMO_SERVER or self.XM_SERVER
        return login, password, server

    def get_metaapi_account_id(self, account_type: Optional[str] = None) -> str:
        """Returns MetaAPI account ID for the requested or configured account type."""
        acct_type = (account_type or self.XM_ACCOUNT_TYPE).lower().strip()
        if acct_type == "live":
            return self.METAAPI_LIVE_ACCOUNT_ID or self.METAAPI_ACCOUNT_ID
        return self.METAAPI_DEMO_ACCOUNT_ID or self.METAAPI_ACCOUNT_ID


config = Settings()
