"""
Data Preprocessor and Feature Engineering Engine for Gold RL Trading Agent.
Extracts stationary, normalized microstructure, session, technical, and macro features
from Gold candles (XAUUSD / GC=F) for Gymnasium environments and PPO agents.
"""
from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import (
    calc_atr,
    calc_bollinger,
    calc_donchian,
    calc_ema,
    calc_rsi,
    calc_supertrend,
)
from gold_trader.core.macro import analyze_macro_drivers

logger = logging.getLogger("GoldRLDataProcessor")


class GoldRLDataProcessor:
    """
    Downloads, cleans, and constructs stationary state matrices for Gold RL environments.
    """

    FEATURE_NAMES = [
        # Returns & price momentum
        "return_1",
        "return_3",
        "return_5",
        "return_10",
        # Candle microstructure
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "range_to_atr",
        # Trend indicators in ATR units
        "dist_ema20_atr",
        "dist_ema50_atr",
        "dist_ema200_atr",
        "spread_ema20_50_atr",
        # Oscillators & Volatility
        "rsi_norm",
        "supertrend_dir",
        "donchian_pos",
        "bollinger_pos",
        "bollinger_width_rel",
        "atr_relative",
        # Session timing & liquidity
        "is_london",
        "is_ny",
        "is_overlap",
        "is_asian",
        "hour_norm",
        "day_norm",
        # Macro confluence
        "macro_score",
    ]

    def __init__(self, scaler_path: Optional[str] = None):
        self.scaler_path = scaler_path
        self.scaler: Optional[StandardScaler] = None

    def build_features_from_candles(
        self,
        candles: List[Dict[str, Any]],
        macro_score: float = 0.0,
        fit_scaler: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Processes a list of OHLCV candle dicts into stationary feature vectors.

        Returns:
            X: np.ndarray of shape (num_samples, num_features)
            prices: np.ndarray of shape (num_samples, 4) with [Open, High, Low, Close]
            valid_candles: List of matching candle dicts corresponding to the rows
        """
        n = len(candles)
        if n < 220:
            raise ValueError(f"Need at least 220 candles to compute 200 EMA and warm up indicators. Got {n}")

        closes = [float(c["close"]) for c in candles]
        highs = [float(c["high"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        opens = [float(c["open"]) for c in candles]

        # Calculate standard technical indicators using gold-trading-system core library
        ema20 = calc_ema(closes, 20)
        ema50 = calc_ema(closes, 50)
        ema200 = calc_ema(closes, 200)
        atr14 = calc_atr(highs, lows, closes, 14)
        rsi14 = calc_rsi(closes, 14)
        st = calc_supertrend(highs, lows, closes, 10, 3.0)
        dc = calc_donchian(highs, lows, 20)
        bb = calc_bollinger(closes, 20, 2.0)

        feature_matrix = []
        price_matrix = []
        valid_candles = []

        start_idx = 200  # Warmup for EMA 200
        for i in range(start_idx, n):
            if (
                ema200[i] is None
                or atr14[i] is None
                or rsi14[i] is None
                or dc["upper"][i] is None
                or bb["upper"][i] is None
            ):
                continue

            c = closes[i]
            o = opens[i]
            h = highs[i]
            l = lows[i]
            rng = max(0.01, h - l)
            atr_val = max(0.10, atr14[i])

            # Returns
            ret_1 = (c - closes[i - 1]) / closes[i - 1]
            ret_3 = (c - closes[i - 3]) / closes[i - 3] if i >= 3 else 0.0
            ret_5 = (c - closes[i - 5]) / closes[i - 5] if i >= 5 else 0.0
            ret_10 = (c - closes[i - 10]) / closes[i - 10] if i >= 10 else 0.0

            # Microstructure
            body = abs(c - o)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            body_ratio = body / rng
            upper_wick_ratio = upper_wick / rng
            lower_wick_ratio = lower_wick / rng
            range_to_atr = rng / atr_val

            # Distance in ATR units
            dist_ema20 = (c - ema20[i]) / atr_val if ema20[i] else 0.0
            dist_ema50 = (c - ema50[i]) / atr_val if ema50[i] else 0.0
            dist_ema200 = (c - ema200[i]) / atr_val if ema200[i] else 0.0
            spread_ema20_50 = (ema20[i] - ema50[i]) / atr_val if (ema20[i] and ema50[i]) else 0.0

            # Oscillators & Bands
            rsi_norm = (rsi14[i] - 50.0) / 50.0  # Center around 0 [-1.0, 1.0]
            st_dir = float(st["direction"][i]) if st["direction"][i] is not None else 0.0

            dc_rng = max(0.01, dc["upper"][i] - dc["lower"][i])
            donchian_pos = ((c - dc["lower"][i]) / dc_rng) * 2.0 - 1.0  # [-1.0, 1.0]

            bb_rng = max(0.01, bb["upper"][i] - bb["lower"][i])
            bollinger_pos = ((c - bb["lower"][i]) / bb_rng) * 2.0 - 1.0  # [-1.0, 1.0]
            bollinger_width_rel = bb_rng / c
            atr_relative = atr_val / c

            # Session extraction
            dt_str = candles[i].get("date", "")
            hour = 12
            day_of_week = 2
            try:
                if "T" in dt_str or " " in dt_str:
                    sep = "T" if "T" in dt_str else " "
                    time_part = dt_str.split(sep)[1]
                    hour = int(time_part.split(":")[0])
                if "-" in dt_str:
                    d_part = dt_str.split("T")[0] if "T" in dt_str else dt_str.split(" ")[0]
                    dt_obj = datetime.strptime(d_part, "%Y-%m-%d")
                    day_of_week = dt_obj.weekday()
            except Exception:
                pass

            is_london = 1.0 if 7 <= hour <= 15 else 0.0
            is_ny = 1.0 if 12 <= hour <= 20 else 0.0
            is_overlap = 1.0 if 12 <= hour <= 16 else 0.0
            is_asian = 1.0 if 0 <= hour < 7 else 0.0
            hour_norm = (hour / 24.0) * 2.0 - 1.0  # [-1, 1]
            day_norm = (day_of_week / 4.0) * 2.0 - 1.0  # [-1, 1]

            norm_macro = np.clip(macro_score / 100.0, -1.0, 1.0)

            row = [
                ret_1,
                ret_3,
                ret_5,
                ret_10,
                body_ratio,
                upper_wick_ratio,
                lower_wick_ratio,
                range_to_atr,
                dist_ema20,
                dist_ema50,
                dist_ema200,
                spread_ema20_50,
                rsi_norm,
                st_dir,
                donchian_pos,
                bollinger_pos,
                bollinger_width_rel,
                atr_relative,
                is_london,
                is_ny,
                is_overlap,
                is_asian,
                hour_norm,
                day_norm,
                norm_macro,
            ]
            feature_matrix.append(row)
            price_matrix.append([o, h, l, c, atr_val])
            valid_candles.append(candles[i])

        X = np.array(feature_matrix, dtype=np.float32)
        prices = np.array(price_matrix, dtype=np.float32)

        # Handle any NaN/inf gracefully
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        if fit_scaler:
            self.scaler = RobustScaler()
            X = self.scaler.fit_transform(X).astype(np.float32)
        elif self.scaler is not None:
            X = self.scaler.transform(X).astype(np.float32)

        return X, prices, valid_candles

    def save_scaler(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {path}")

    def load_scaler(self, path: str):
        with open(path, "rb") as f:
            self.scaler = pickle.load(f)
        logger.info(f"Scaler loaded from {path}")
