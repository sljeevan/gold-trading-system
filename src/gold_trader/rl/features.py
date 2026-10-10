"""
Feature engineering module for Gold Trading RL Environment.
Extracts normalized technical, price action, and macroeconomic state vectors for XAU/USD.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gold_trader.core.indicators import (
    calc_atr,
    calc_bollinger,
    calc_donchian,
    calc_ema,
    calc_macd,
    calc_rsi,
    calc_supertrend,
)


class GoldFeatureExtractor:
    """
    Constructs normalized continuous observation vectors for Gold RL trading.
    Computes multi-horizon momentum, trend alignment, volatility, candle physics,
    and relative channel positioning.
    """

    def __init__(
        self,
        lookback_window: int = 1,
        include_lags: bool = True,
        lag_steps: Tuple[int, ...] = (1, 2, 3, 5),
    ):
        self.lookback_window = lookback_window
        self.include_lags = include_lags
        self.lag_steps = lag_steps
        self.feature_names: List[str] = []

    def extract_features(
        self,
        candles: List[Dict[str, Any]],
        dxy_candles: Optional[List[Dict[str, Any]]] = None,
        tnx_candles: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Process raw OHLCV candles into a clean 2D NumPy array of shape (N, feature_dim).
        Handles warmup periods by forward-filling or trimming initial NaN bars.
        """
        n = len(candles)
        if n < 50:
            raise ValueError(f"Need at least 50 candles to compute indicators, got {n}")

        opens = [float(c["open"]) for c in candles]
        highs = [float(c["high"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        closes = [float(c["close"]) for c in candles]
        volumes = [float(c.get("volume", 0.0) or 0.0) for c in candles]

        # 1. Technical Indicators
        ema9 = calc_ema(closes, 9)
        ema21 = calc_ema(closes, 21)
        ema50 = calc_ema(closes, 50)
        ema200 = calc_ema(closes, 200)
        rsi14 = calc_rsi(closes, 14)
        atr14 = calc_atr(highs, lows, closes, 14)
        st = calc_supertrend(highs, lows, closes, atr_period=10, multiplier=3.0)
        st_dir = st["direction"]
        st_upper = st["upper"]
        st_lower = st["lower"]
        donch = calc_donchian(highs, lows, period=20)
        boll = calc_bollinger(closes, period=20, std_mult=2.0)
        macd = calc_macd(closes, fast=12, slow=26, signal=9)

        # Optional Macro Alignments
        dxy_ret_1 = [0.0] * n
        if dxy_candles and len(dxy_candles) >= n:
            dxy_closes = [float(c["close"]) for c in dxy_candles[-n:]]
            for i in range(1, n):
                if dxy_closes[i - 1] > 0:
                    dxy_ret_1[i] = (dxy_closes[i] - dxy_closes[i - 1]) / dxy_closes[i - 1]

        tnx_ret_1 = [0.0] * n
        if tnx_candles and len(tnx_candles) >= n:
            tnx_closes = [float(c["close"]) for c in tnx_candles[-n:]]
            for i in range(1, n):
                if tnx_closes[i - 1] > 0:
                    tnx_ret_1[i] = (tnx_closes[i] - tnx_closes[i - 1]) / tnx_closes[i - 1]

        feature_matrix: List[List[float]] = []

        base_feature_names = [
            "log_ret_1",
            "log_ret_3",
            "log_ret_5",
            "log_ret_10",
            "atr_pct",
            "high_low_range_pct",
            "rsi_normalized",
            "macd_normalized",
            "macd_signal_normalized",
            "macd_hist_normalized",
            "supertrend_dir",
            "supertrend_dist_atr",
            "donchian_pos",
            "bollinger_pos",
            "bollinger_bandwidth",
            "ema9_dist_atr",
            "ema21_dist_atr",
            "ema50_dist_atr",
            "ema200_dist_atr",
            "ema9_21_cross_atr",
            "candle_body_ratio",
            "upper_shadow_ratio",
            "lower_shadow_ratio",
            "volume_norm",
            "dxy_ret_1",
            "tnx_ret_1",
        ]

        vol_window = 20

        for i in range(n):
            c = closes[i]
            o = opens[i]
            h = highs[i]
            l = lows[i]
            v = volumes[i]
            hl_range = max(h - l, 0.01)

            # ATR normalization
            a = atr14[i] if atr14[i] is not None and atr14[i] > 0 else (hl_range)

            # Log returns
            r1 = math.log(c / closes[i - 1]) if i >= 1 and closes[i - 1] > 0 else 0.0
            r3 = math.log(c / closes[i - 3]) if i >= 3 and closes[i - 3] > 0 else 0.0
            r5 = math.log(c / closes[i - 5]) if i >= 5 and closes[i - 5] > 0 else 0.0
            r10 = math.log(c / closes[i - 10]) if i >= 10 and closes[i - 10] > 0 else 0.0

            # Volatility
            atr_pct = a / c if c > 0 else 0.0
            range_pct = hl_range / c if c > 0 else 0.0

            # RSI: centered to [-1, 1]
            rsi_val = (rsi14[i] - 50.0) / 50.0 if rsi14[i] is not None else 0.0

            # MACD normalized by ATR
            m_val = (macd["macd"][i] or 0.0) / a
            m_sig = (macd["signal"][i] or 0.0) / a
            m_hist = (macd["hist"][i] or 0.0) / a

            # Supertrend
            s_dir = float(st_dir[i] or 0)
            if s_dir == 1 and st_lower[i] is not None:
                s_dist = (c - st_lower[i]) / a
            elif s_dir == -1 and st_upper[i] is not None:
                s_dist = (c - st_upper[i]) / a
            else:
                s_dist = 0.0

            # Donchian channel position [0, 1] -> centered to [-1, 1]
            d_u = donch["upper"][i]
            d_l = donch["lower"][i]
            if d_u is not None and d_l is not None and d_u > d_l:
                d_pos = (2.0 * (c - d_l) / (d_u - d_l)) - 1.0
            else:
                d_pos = 0.0

            # Bollinger position [0, 1] -> centered to [-1, 1]
            b_u = boll["upper"][i]
            b_l = boll["lower"][i]
            if b_u is not None and b_l is not None and b_u > b_l:
                b_pos = (2.0 * (c - b_l) / (b_u - b_l)) - 1.0
            else:
                b_pos = 0.0
            b_bw = boll["bbw"][i] or 0.0

            # EMA distances in units of ATR
            e9_dist = ((c - ema9[i]) / a) if ema9[i] is not None else 0.0
            e21_dist = ((c - ema21[i]) / a) if ema21[i] is not None else 0.0
            e50_dist = ((c - ema50[i]) / a) if ema50[i] is not None else 0.0
            e200_dist = ((c - ema200[i]) / a) if ema200[i] is not None else 0.0
            e9_21_cross = ((ema9[i] - ema21[i]) / a) if (ema9[i] is not None and ema21[i] is not None) else 0.0

            # Candle physics
            body_ratio = (c - o) / hl_range
            upper_shadow = (h - max(o, c)) / hl_range
            lower_shadow = (min(o, c) - l) / hl_range

            # Volume ratio
            start_v = max(0, i - vol_window + 1)
            mean_v = sum(volumes[start_v : i + 1]) / float(i - start_v + 1) if (i - start_v + 1) > 0 else 1.0
            vol_norm = math.log1p(v) / (math.log1p(mean_v) + 1e-8) - 1.0

            row = [
                float(np.clip(r1, -0.1, 0.1)),
                float(np.clip(r3, -0.15, 0.15)),
                float(np.clip(r5, -0.2, 0.2)),
                float(np.clip(r10, -0.25, 0.25)),
                float(np.clip(atr_pct * 100.0, 0.0, 10.0)),
                float(np.clip(range_pct * 100.0, 0.0, 10.0)),
                float(np.clip(rsi_val, -1.0, 1.0)),
                float(np.clip(m_val, -5.0, 5.0)),
                float(np.clip(m_sig, -5.0, 5.0)),
                float(np.clip(m_hist, -5.0, 5.0)),
                float(np.clip(s_dir, -1.0, 1.0)),
                float(np.clip(s_dist, -10.0, 10.0)),
                float(np.clip(d_pos, -2.0, 2.0)),
                float(np.clip(b_pos, -2.0, 2.0)),
                float(np.clip(b_bw * 100.0, 0.0, 20.0)),
                float(np.clip(e9_dist, -10.0, 10.0)),
                float(np.clip(e21_dist, -10.0, 10.0)),
                float(np.clip(e50_dist, -15.0, 15.0)),
                float(np.clip(e200_dist, -25.0, 25.0)),
                float(np.clip(e9_21_cross, -5.0, 5.0)),
                float(np.clip(body_ratio, -1.0, 1.0)),
                float(np.clip(upper_shadow, 0.0, 1.0)),
                float(np.clip(lower_shadow, 0.0, 1.0)),
                float(np.clip(vol_norm, -3.0, 3.0)),
                float(np.clip(dxy_ret_1[i], -0.05, 0.05)),
                float(np.clip(tnx_ret_1[i], -0.1, 0.1)),
            ]
            feature_matrix.append(row)

        mat = np.array(feature_matrix, dtype=np.float32)

        # Build feature lag columns if requested
        if self.include_lags and len(self.lag_steps) > 0:
            lagged_blocks = [mat]
            all_names = list(base_feature_names)
            for lag in self.lag_steps:
                lagged = np.roll(mat, shift=lag, axis=0)
                lagged[:lag, :] = mat[0, :]  # pad head with first observation
                lagged_blocks.append(lagged)
                for name in base_feature_names:
                    all_names.append(f"{name}_lag_{lag}")
            final_features = np.concatenate(lagged_blocks, axis=1)
            self.feature_names = all_names
        else:
            final_features = mat
            self.feature_names = base_feature_names

        # Warmup mask: trim initial bars where 50 EMA is warming up
        return final_features
