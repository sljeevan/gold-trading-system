"""
Machine Learning High-Probability Meta-Labeling & Triple-Barrier Engine for Gold & Crypto.

Employs Marcos López de Prado's Meta-Labeling & Triple-Barrier technique:
1. Takes technical and price-action trade candidates (Donchian Touch, Breakout, Momentum).
2. Extracts stationary microstructure, volatility, session, and macro features.
3. Predicts: P(Trade hits Take Profit before Stop Loss) with dynamic ATR-based clearance.
4. Serves as a high-conviction gatekeeper and dynamic lot size optimizer.
"""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import (
    calc_atr,
    calc_donchian,
    calc_ema,
    calc_macd,
    calc_rsi,
    calc_supertrend,
)
from gold_trader.core.macro import analyze_macro_drivers

logger = logging.getLogger("GoldMLFilter")

DEFAULT_GOLD_MODEL_PATH = os.path.expanduser("~/.gold_trading_data/models/gold_ml_barrier.joblib")
DEFAULT_BTC_MODEL_PATH = os.path.expanduser("~/.gold_trading_data/models/btc_ml_barrier.joblib")


def extract_candle_features(
    candles: List[Dict[str, Any]],
    macro_score: float = 0.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract stationary quantitative features for every bar.
    Zero price-level dependence (uses normalized distances, ratios, cyclical time, and percentages).
    """
    n = len(candles)
    closes = [float(c["close"]) for c in candles]
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    opens = [float(c["open"]) for c in candles]

    # Technical indicators
    ema20 = calc_ema(closes, 20)
    ema50 = calc_ema(closes, 50)
    ema200 = calc_ema(closes, 200)
    atr14 = calc_atr(highs, lows, closes, 14)
    atr50 = calc_atr(highs, lows, closes, 50)
    rsi14 = calc_rsi(closes, 14)
    st = calc_supertrend(highs, lows, closes, 10, 3.0)
    dc10 = calc_donchian(highs, lows, 10)
    dc20 = calc_donchian(highs, lows, 20)
    macd = calc_macd(closes, 12, 26, 9)

    feature_names = [
        "body_ratio",
        "lower_wick_ratio",
        "upper_wick_ratio",
        "range_to_atr",
        "dist_ema20_atr",
        "dist_ema50_atr",
        "dist_ema200_atr",
        "macro_trend_divergence",
        "rsi14",
        "rsi_delta_3",
        "supertrend_dir",
        "donchian10_pos",
        "donchian20_pos",
        "donchian_width_atr",
        "macd_hist_atr",
        "atr_ratio_50",
        "hour_sin",
        "hour_cos",
        "is_london_session",
        "is_ny_session",
        "is_overlap_session",
        "candle_direction",
        "macro_score",
    ]

    feature_matrix = []

    for i in range(n):
        # Skip warm-up period
        if (
            i < 200
            or ema200[i] is None
            or atr14[i] is None
            or atr50[i] is None
            or rsi14[i] is None
            or dc20["upper"][i] is None
            or dc20["lower"][i] is None
            or macd["histogram"][i] is None
        ):
            feature_matrix.append([np.nan] * len(feature_names))
            continue

        h = highs[i]
        l = lows[i]
        o = opens[i]
        c = closes[i]
        rng = max(0.01, h - l)
        atr_val = max(0.10, atr14[i])
        atr50_val = max(0.10, atr50[i])

        body = abs(c - o)
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)

        body_ratio = body / rng
        lower_wick_ratio = lower_wick / rng
        upper_wick_ratio = upper_wick / rng
        range_to_atr = rng / atr_val

        dist_ema20 = (c - ema20[i]) / atr_val if ema20[i] else 0.0
        dist_ema50 = (c - ema50[i]) / atr_val if ema50[i] else 0.0
        dist_ema200 = (c - ema200[i]) / atr_val if ema200[i] else 0.0
        macro_div = (ema50[i] - ema200[i]) / atr_val if (ema50[i] and ema200[i]) else 0.0

        rsi_val = rsi14[i]
        rsi_prev = rsi14[i - 3] if i >= 3 and rsi14[i - 3] is not None else rsi_val
        rsi_delta = rsi_val - rsi_prev

        st_dir = float(st["direction"][i]) if st["direction"][i] is not None else 0.0

        dc10_range = max(0.01, (dc10["upper"][i] or h) - (dc10["lower"][i] or l))
        dc10_pos = (c - (dc10["lower"][i] or l)) / dc10_range

        dc20_range = max(0.01, dc20["upper"][i] - dc20["lower"][i])
        dc20_pos = (c - dc20["lower"][i]) / dc20_range
        dc_width_atr = dc20_range / atr_val

        macd_hist_atr = macd["histogram"][i] / atr_val
        atr_ratio = atr_val / atr50_val

        # Parse hour for session flags and cyclical encoding
        dt_str = candles[i].get("date", "")
        hour = 12
        try:
            if "T" in dt_str or " " in dt_str:
                sep = "T" if "T" in dt_str else " "
                time_part = dt_str.split(sep)[1]
                hour = int(time_part.split(":")[0])
        except Exception:
            hour = 12

        hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
        hour_cos = math.cos(2.0 * math.pi * hour / 24.0)
        is_london = 1.0 if 7 <= hour <= 15 else 0.0
        is_ny = 1.0 if 12 <= hour <= 20 else 0.0
        is_overlap = 1.0 if 12 <= hour <= 16 else 0.0
        candle_dir = 1.0 if c >= o else 0.0

        row = [
            round(body_ratio, 4),
            round(lower_wick_ratio, 4),
            round(upper_wick_ratio, 4),
            round(range_to_atr, 4),
            round(dist_ema20, 4),
            round(dist_ema50, 4),
            round(dist_ema200, 4),
            round(macro_div, 4),
            round(rsi_val, 2),
            round(rsi_delta, 2),
            st_dir,
            round(dc10_pos, 4),
            round(dc20_pos, 4),
            round(dc_width_atr, 4),
            round(macd_hist_atr, 4),
            round(atr_ratio, 4),
            round(hour_sin, 4),
            round(hour_cos, 4),
            is_london,
            is_ny,
            is_overlap,
            candle_dir,
            macro_score,
        ]
        feature_matrix.append(row)

    return np.array(feature_matrix, dtype=float), feature_names


def label_triple_barrier(
    candles: List[Dict[str, Any]],
    direction: Literal["BUY", "SELL"],
    atr_multiplier_sl: float = 1.5,
    rr_ratio: float = 1.25,
    min_sl_pts: float = 8.00,
    max_holding_bars: int = 36,
    tp_points: Optional[float] = None,
    sl_points: Optional[float] = None,
) -> np.ndarray:
    """
    Applies the Triple-Barrier Method to generate labels for directional trades:
    1 = Trade hit Take Profit (+rr_ratio * SL or +tp_points) before Stop Loss (-1.5 * ATR or -sl_points).
    0 = Trade hit Stop Loss first (or ended in loss at time barrier).
    """
    n = len(candles)
    labels = np.full(n, np.nan)
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    closes = [float(c["close"]) for c in candles]
    atr14 = calc_atr(highs, lows, closes, 14)

    for i in range(n - max_holding_bars):
        entry_price = closes[i]
        if tp_points is not None and sl_points is not None:
            tp_dist = tp_points
            sl_dist = sl_points
        else:
            atr_val = atr14[i]
            if atr_val is None or atr_val <= 0:
                continue
            sl_dist = max(atr_multiplier_sl * atr_val, min_sl_pts)
            tp_dist = rr_ratio * sl_dist
        outcome = 0

        if direction == "BUY":
            tp_barrier = entry_price + tp_dist
            sl_barrier = entry_price - sl_dist

            for k in range(i + 1, min(i + max_holding_bars + 1, n)):
                if highs[k] >= tp_barrier:
                    outcome = 1
                    break
                if lows[k] <= sl_barrier:
                    outcome = 0
                    break
            else:
                outcome = 1 if closes[min(i + max_holding_bars, n - 1)] > entry_price else 0

        else:  # SELL
            tp_barrier = entry_price - tp_dist
            sl_barrier = entry_price + sl_dist

            for k in range(i + 1, min(i + max_holding_bars + 1, n)):
                if lows[k] <= tp_barrier:
                    outcome = 1
                    break
                if highs[k] >= sl_barrier:
                    outcome = 0
                    break
            else:
                outcome = 1 if closes[min(i + max_holding_bars, n - 1)] < entry_price else 0

        labels[i] = outcome

    return labels


class GoldMLPredictor:
    """
    Gradient Boosting Meta-Classifier & Probability Estimator for Gold & Bitcoin trades.
    """

    def __init__(self, model_path: Optional[str] = None, symbol: str = "XAUUSD"):
        self.symbol = symbol.upper()
        if model_path:
            self.model_path = os.path.expanduser(model_path)
        else:
            self.model_path = DEFAULT_BTC_MODEL_PATH if ("BTC" in self.symbol) else DEFAULT_GOLD_MODEL_PATH

        self.model: Any = None
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self._load_cached_model()

    def _load_cached_model(self) -> bool:
        if os.path.exists(self.model_path):
            try:
                bundle = joblib.load(self.model_path)
                self.model = bundle["model"]
                self.feature_names = bundle["feature_names"]
                self.metadata = bundle.get("metadata", {})
                logger.info(f"Loaded cached ML model from {self.model_path} (Trained: {self.metadata.get('trained_at')})")
                return True
            except Exception as e:
                logger.warning(f"Could not load cached model from {self.model_path}: {e}")
        return False

    def train_model(
        self,
        symbol: Optional[str] = None,
        period: str = "2y",
        interval: str = "1h",
        atr_multiplier_sl: float = 1.5,
        rr_ratio: float = 1.25,
        min_sl_pts: Optional[float] = None,
        max_holding_bars: int = 36,
    ) -> Dict[str, Any]:
        """
        Trains the Meta-Classifier on historical candles using walk-forward time-series cross-validation.
        """
        sym = (symbol or self.symbol).upper()
        is_btc = "BTC" in sym
        if min_sl_pts is None:
            min_sl_pts = 500.0 if is_btc else 8.00

        logger.info(f"Fetching {period} {interval} data for {sym} ML training...")
        candles = fetch_gold_candles(sym, period=period, interval=interval)
        if len(candles) < 400:
            return {"error": f"Insufficient candles ({len(candles)}) for ML training. Minimum 400 needed."}

        macro_data = analyze_macro_drivers()
        macro_score = float(macro_data.get("macro_gold_score", 0.0))

        X_all, feature_names = extract_candle_features(candles, macro_score=macro_score)

        # Dynamic ATR Triple-Barrier Labeling
        y_buy = label_triple_barrier(
            candles,
            direction="BUY",
            atr_multiplier_sl=atr_multiplier_sl,
            rr_ratio=rr_ratio,
            min_sl_pts=min_sl_pts,
            max_holding_bars=max_holding_bars,
        )
        y_sell = label_triple_barrier(
            candles,
            direction="SELL",
            atr_multiplier_sl=atr_multiplier_sl,
            rr_ratio=rr_ratio,
            min_sl_pts=min_sl_pts,
            max_holding_bars=max_holding_bars,
        )

        # Combine datasets while strictly preserving chronological time order
        rows_X = []
        rows_y = []
        n = len(candles)
        for i in range(n - max_holding_bars):
            if np.isnan(X_all[i]).any():
                continue
            if not np.isnan(y_buy[i]):
                rows_X.append(np.append(X_all[i], 1.0))
                rows_y.append(y_buy[i])
            if not np.isnan(y_sell[i]):
                rows_X.append(np.append(X_all[i], -1.0))
                rows_y.append(y_sell[i])

        X_combined = np.array(rows_X, dtype=float)
        y_combined = np.array(rows_y, dtype=float)

        augmented_feature_names = feature_names + ["trade_direction_sign"]

        # Time-series Walk-Forward Cross-Validation
        tscv = TimeSeriesSplit(n_splits=5)
        auc_scores = []
        precision_scores = []
        brier_scores = []

        if HAS_LIGHTGBM:
            clf = lgb.LGBMClassifier(
                n_estimators=120,
                learning_rate=0.03,
                num_leaves=20,
                min_child_samples=20,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbose=-1,
            )
        else:
            clf = HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.04,
                max_leaf_nodes=25,
                min_samples_leaf=20,
                random_state=42,
            )

        for train_idx, val_idx in tscv.split(X_combined):
            X_tr, X_val = X_combined[train_idx], X_combined[val_idx]
            y_tr, y_val = y_combined[train_idx], y_combined[val_idx]

            clf.fit(X_tr, y_tr)
            preds_proba = clf.predict_proba(X_val)[:, 1]
            preds_class = (preds_proba >= 0.5).astype(int)

            auc_scores.append(roc_auc_score(y_val, preds_proba))
            precision_scores.append(precision_score(y_val, preds_class, zero_division=0))
            brier_scores.append(brier_score_loss(y_val, preds_proba))

        # Final fit on entire dataset
        clf.fit(X_combined, y_combined)

        # High-probability bucket performance (P >= 0.55)
        final_probas = clf.predict_proba(X_combined)[:, 1]
        mask_55 = final_probas >= 0.55
        win_rate_55 = round(float(np.mean(y_combined[mask_55]) * 100), 1) if np.any(mask_55) else 0.0

        mask_60 = final_probas >= 0.60
        win_rate_60 = round(float(np.mean(y_combined[mask_60]) * 100), 1) if np.any(mask_60) else 0.0

        # Persist model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        metadata = {
            "symbol": sym,
            "period": period,
            "interval": interval,
            "samples_count": len(X_combined),
            "atr_multiplier_sl": atr_multiplier_sl,
            "rr_ratio": rr_ratio,
            "min_sl_pts": min_sl_pts,
            "avg_auc": round(float(np.mean(auc_scores)), 3),
            "avg_precision": round(float(np.mean(precision_scores)), 3),
            "avg_brier": round(float(np.mean(brier_scores)), 4),
            "win_rate_p55_pct": win_rate_55,
            "win_rate_p60_pct": win_rate_60,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model_type": "LightGBM" if HAS_LIGHTGBM else "HistGradientBoosting",
        }

        bundle = {
            "model": clf,
            "feature_names": augmented_feature_names,
            "metadata": metadata,
        }
        joblib.dump(bundle, self.model_path)

        self.model = clf
        self.feature_names = augmented_feature_names
        self.metadata = metadata

        logger.info(f"✅ ML Model successfully trained and saved to {self.model_path} (AUC: {metadata['avg_auc']}, WinRate@0.55: {win_rate_55}%)")

        return {
            "status": "SUCCESS",
            "model_path": self.model_path,
            "metrics": metadata,
        }

    def predict_trade_probability(
        self,
        candles: List[Dict[str, Any]],
        direction: Literal["BUY", "SELL"],
        macro_score: Optional[float] = None,
        min_probability_threshold: float = 0.52,
    ) -> Dict[str, Any]:
        """
        Evaluate real-time candle state and return the predicted win probability for the trade.
        """
        if self.model is None:
            logger.info(f"ML model for {self.symbol} not found in cache. Training initial model...")
            self.train_model(period="2y", interval="1h")

        if self.model is None:
            return {"error": "Failed to initialize ML model"}

        if macro_score is None:
            macro_data = analyze_macro_drivers()
            macro_score = float(macro_data.get("macro_gold_score", 0.0))

        X_all, _ = extract_candle_features(candles, macro_score=macro_score)
        latest_features = X_all[-1]

        if np.isnan(latest_features).any():
            return {
                "error": "Insufficient historical candles to compute complete feature vector. Need >= 200 bars.",
                "probability_win": 0.50,
                "is_high_conviction": False,
                "recommendation": "PASS_INSUFFICIENT_DATA",
                "recommended_lot_multiplier": 1.0,
            }

        dir_sign = 1.0 if direction.upper() == "BUY" else -1.0
        feature_vector = np.append(latest_features, [dir_sign]).reshape(1, -1)

        probas = self.model.predict_proba(feature_vector)[0]
        prob_win = round(float(probas[1]), 3)
        prob_loss = round(float(probas[0]), 3)

        is_high_conviction = prob_win >= min_probability_threshold

        # Dynamic Lot Sizing Multiplier based on probability edge
        if prob_win < 0.45:
            lot_mult = 0.0  # Pass entirely
            recommendation = "REJECT_NEGATIVE_EDGE"
        elif prob_win < min_probability_threshold:
            lot_mult = 0.50  # De-risk if forced
            recommendation = "PASS_LOW_PROBABILITY"
        elif prob_win < 0.60:
            lot_mult = 1.00  # Standard risk
            recommendation = "EXECUTE_APPROVED"
        else:
            lot_mult = 1.25  # High-conviction boost
            recommendation = "EXECUTE_HIGH_CONVICTION"

        confidence_score = int(round(prob_win * 100))

        return {
            "direction": direction.upper(),
            "probability_win": prob_win,
            "probability_loss": prob_loss,
            "threshold": min_probability_threshold,
            "is_high_conviction": is_high_conviction,
            "recommendation": recommendation,
            "confidence_score": confidence_score,
            "recommended_lot_multiplier": lot_mult,
            "model_trained_at": self.metadata.get("trained_at", "Unknown"),
            "model_type": self.metadata.get("model_type", "GradientBoosting"),
            "historical_auc": self.metadata.get("avg_auc", 0.60),
        }
