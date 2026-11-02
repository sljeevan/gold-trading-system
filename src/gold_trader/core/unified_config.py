"""
Unified Configuration & Provenance Schema for Aegis Alpha Quantitative Trading System.
Shared identically across Backtesting, Paper Trading, and Live Daemon Execution.

Enforces:
- Cryptographic configuration hashing (SHA-256)
- Git commit SHA tracking
- Schema versioning
- Fail-fast parameter validation
- Strict rejection of unapproved overrides
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger("UnifiedConfig")

CURRENT_SCHEMA_VERSION = "2.0.0"
CURRENT_STRATEGY_VERSION = "2.1.0"


def resolve_git_sha() -> str:
    """Resolve current Git commit SHA deterministically with safe fallback."""
    try:
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        ).decode("utf-8").strip()
        if sha:
            return sha
    except Exception:
        pass
    return os.getenv("GIT_COMMIT_SHA", "UNKNOWN_GIT_SHA")


@dataclass
class UnifiedRunConfig:
    # Versioning & Provenance
    schema_version: str = CURRENT_SCHEMA_VERSION
    strategy_version: str = CURRENT_STRATEGY_VERSION
    git_sha: str = field(default_factory=resolve_git_sha)
    config_hash: str = ""

    # Instrument & Strategy Routing
    symbol: str = "XAUUSD"
    timeframe: str = "1h"
    strategy_mode: str = "FLAT_RISK_TOUCH"
    trigger_mode: str = "TOUCH"  # "TOUCH" or "CLOSE"
    execution_mode: str = "LIVE"  # "LIVE", "PAPER", "BACKTEST"

    # Core Strategy Parameters
    donchian_period: int = 10
    rr_multiplier: float = 1.25
    trend_filter_ema: int = 50
    require_slope_alignment: bool = True
    adx_min_threshold: Optional[float] = None
    use_pullback_limit: bool = True
    pullback_discount_pips: float = 8.0

    # Risk & Capital Allocation (7.54% / $75.44 test baseline intentionally retained per scope note)
    risk_mode: str = "FLAT_DOLLAR"  # "FLAT_DOLLAR" or "PERCENT"
    fixed_risk_usd: float = 75.44
    risk_pct_per_trade: float = 7.54
    max_open_positions: Optional[int] = 0  # 0 or None = uncapped

    # Institutional Two-Stage Execution & Runner Management
    enable_partial_scale_out: bool = True
    partial_scale_out_pct: float = 0.50
    scale_out_r: float = 1.00
    enable_runner_trailing: bool = True
    runner_trail_buffer_usd: float = 4.50
    runner_trail_buffer_r: Optional[float] = 0.50
    runner_target_2_multiplier: float = 3.00
    trail_to_breakeven_at_t1: bool = True

    # Broker Execution Costs & Fill Model
    spread_usd: float = 0.35
    slippage_usd: float = 0.05
    commission_per_lot_usd: float = 7.00
    max_spread_pips: float = 4.0

    # News Catalyst Guard (OpenBB)
    enable_news_catalyst_guard: bool = True
    news_guard_buffer_minutes: int = 15

    # Machine Learning Meta-Labeling Gatekeeper
    enable_ml_filter: bool = False
    ml_min_probability: float = 0.50
    ml_model_path: Optional[str] = None
    ml_required: bool = False
    ml_failure_policy: Literal["FAIL_CLOSED", "FAIL_OPEN"] = "FAIL_CLOSED"
    require_broker: bool = True

    # Pacing & Session Filters
    one_trade_per_bar: bool = True
    loss_cooldown_minutes: int = 60
    session_filter: bool = True
    session_start_hour_utc: int = 7
    session_end_hour_utc: int = 19
    max_daily_trades: int = 4

    def __post_init__(self):
        if not self.config_hash:
            self.config_hash = self.compute_hash()

    def validate(self) -> None:
        """
        Fail-fast validation enforcing schema boundaries and parameter sanity.
        Raises ValueError on any parameter inconsistency.
        """
        valid_timeframes = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
        if self.timeframe.lower() not in valid_timeframes:
            raise ValueError(f"Invalid timeframe '{self.timeframe}'. Allowed: {sorted(valid_timeframes)}")

        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError(f"Invalid symbol '{self.symbol}'. Must be non-empty string.")

        if self.donchian_period < 2:
            raise ValueError(f"donchian_period must be >= 2, got {self.donchian_period}")

        if self.rr_multiplier <= 0:
            raise ValueError(f"rr_multiplier must be > 0, got {self.rr_multiplier}")

        if self.fixed_risk_usd <= 0:
            raise ValueError(f"fixed_risk_usd must be > 0, got {self.fixed_risk_usd}")

        if self.risk_pct_per_trade <= 0:
            raise ValueError(f"risk_pct_per_trade must be > 0, got {self.risk_pct_per_trade}")

        if self.trigger_mode.upper() not in ("TOUCH", "CLOSE", "BREAKOUT"):
            raise ValueError(f"Invalid trigger_mode '{self.trigger_mode}'. Allowed: TOUCH, CLOSE, BREAKOUT")

        if self.execution_mode.upper() not in ("LIVE", "PAPER", "BACKTEST"):
            raise ValueError(f"Invalid execution_mode '{self.execution_mode}'. Allowed: LIVE, PAPER, BACKTEST")

        if self.ml_failure_policy not in ("FAIL_CLOSED", "FAIL_OPEN"):
            raise ValueError(f"Invalid ml_failure_policy '{self.ml_failure_policy}'. Allowed: FAIL_CLOSED, FAIL_OPEN")

        if self.spread_usd < 0 or self.slippage_usd < 0 or self.commission_per_lot_usd < 0:
            raise ValueError("Cost parameters (spread, slippage, commission) cannot be negative.")

        if not (0.0 < self.partial_scale_out_pct <= 1.0):
            raise ValueError(f"partial_scale_out_pct must be between 0 and 1, got {self.partial_scale_out_pct}")

    def compute_hash(self) -> str:
        """
        Compute deterministic SHA-256 hash of canonical configuration parameters.
        Excludes runtime-generated fields (config_hash, git_sha) to guarantee identical
        hashes for identical trading configurations.
        """
        d = asdict(self)
        d.pop("config_hash", None)
        d.pop("git_sha", None)
        # Canonical JSON string with sorted keys
        canonical_str = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        d = asdict(self)
        d["config_hash"] = self.compute_hash()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UnifiedRunConfig:
        """
        Instantiate UnifiedRunConfig from dictionary with strict unknown parameter rejection.
        """
        valid_fields = set(cls.__dataclass_fields__.keys())
        unknown_fields = set(data.keys()) - valid_fields
        if unknown_fields:
            raise ValueError(f"Unapproved configuration parameter(s) rejected: {sorted(unknown_fields)}")

        cfg = cls(**data)
        cfg.validate()
        return cfg

    def save_json(self, path: str) -> None:
        """Save configuration to JSON file with indentation."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load_json(cls, path: str) -> UnifiedRunConfig:
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
