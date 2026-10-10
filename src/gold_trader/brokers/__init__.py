"""
Brokers package for Gold execution (Paper Trading, XM MT5 Native, Free Socket Bridge, MetaAPI Cloud).
"""
from gold_trader.brokers.base import BaseBroker
from gold_trader.brokers.paper_broker import PaperBroker
from gold_trader.brokers.xm_mt5_broker import XmMT5Broker
from gold_trader.brokers.socket_broker import SocketBroker
from gold_trader.brokers.metaapi_broker import MetaApiBroker

__all__ = ["BaseBroker", "PaperBroker", "XmMT5Broker", "SocketBroker", "MetaApiBroker"]

