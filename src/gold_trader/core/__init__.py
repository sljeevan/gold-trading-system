"""
Core technical analysis, macro data, strategy logic, strategy catalog, and backtesting for Gold.
"""
from gold_trader.core.strategy_catalog import (
    RiskArchetype,
    STRATEGY_CATALOG,
    StrategyProfile,
    StrategyStatus,
    export_catalog_json,
    get_active_strategies,
    get_catalog_summary_table,
    get_high_winrate_candidates,
    get_strategy,
    get_strategy_catalog,
    list_strategies,
)

__all__ = [
    "STRATEGY_CATALOG",
    "StrategyProfile",
    "StrategyStatus",
    "RiskArchetype",
    "get_strategy_catalog",
    "get_strategy",
    "list_strategies",
    "get_active_strategies",
    "get_high_winrate_candidates",
    "get_catalog_summary_table",
    "export_catalog_json",
]
