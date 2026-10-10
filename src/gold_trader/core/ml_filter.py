"""
Machine Learning High-Probability Meta-Labeling & Triple-Barrier Engine for Gold (XAUUSD).

Employs Marcos López de Prado's Meta-Labeling technique:
1. Takes technical and price-action trade candidates.
2. Extracts stationary microstructure, volatility, session, and macro features.
3. Predicts: P(Trade hits +20 pips TP before -15 pips SL).
4. Serves as a high-conviction gatekeeper (Filters out trades with P < threshold, e.g., 0.70).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from gold_trader.core.data_feed import fetch_gold_candles
from gold_trader.core.indicators import (
    calc_atr,
    calc_donchian,
    calc_ema,
    calc_rsi,
    calc_supertrend,
)
from gold_trader.core.macro import analyze_macro_drivers

logger = logging.getLogger("GoldMLFilter")

DEFAULT_MODEL_PATH = os.path.expanduser("~/.gold_trading_data/models/gold_ml_barrier.joblib")


def extract_candle_features(
    candles: List[Dict[str, Any]],
    macro_score: float = 0.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract stationary quantitative features for every bar.
    Zero price-level dependence (uses normalized distances, ratios, and percentages).
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
    rsi14 = calc_rsi(closes, 14)
    st = calc_supertrend(highs, lows, closes, 10, 3.0)
    dc = calc_donchian(highs, lows, 20)

    feature_names = [
        "body_ratio",
        "lower_wick_ratio",
        "upper_wick_ratio",
        "range_to_atr",
        "dist_ema20_atr",
        "dist_ema50_atr",
        "dist_ema200_atr",
        "rsi14",
        "supertrend_dir",
        "donchian_pos",
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
            or rsi14[i] is None
            or dc["upper"][i] is None
            or dc["lower"][i] is None
        ):
            feature_matrix.append([np.nan] * len(feature_names))
            continue

        h = highs[i]
        l = lows[i]
        o = opens[i]
        c = closes[i]
        rng = max(0.01, h - l)
        atr_val = max(0.10, atr14[i])

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

        st_dir = float(st["direction"][i]) if st["direction"][i] is not None else 0.0
        dc_range = max(0.01, dc["upper"][i] - dc["lower"][i])
        donchian_pos = (c - dc["lower"][i]) / dc_range

        # Parse hour for session flags
        dt_str = candles[i].get("date", "")
        hour = 12
        try:
            if "T" in dt_str or " " in dt_str:
                sep = "T" if "T" in dt_str else " "
                time_part = dt_str.split(sep)[1]
                hour = int(time_part.split(":")[0])
        except Exception:
            hour = 12

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
            round(rsi14[i], 2),
            st_dir,
            round(donchian_pos, 4),
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
    tp_points: float = 2.00,  # 20 pips on Gold
    sl_points: float = 1.54,  # ~15.4 pips on Gold
    max_holding_bars: int = 24,  # Max bars before time-barrier expiration
) -> np.ndarray:
    """
    Applies the Triple-Barrier Method to generate labels for directional trades:
    1 = Trade hit Take Profit (+20 pips) before Stop Loss (-15.4 pips).
    0 = Trade hit Stop Loss first (or ended in loss at time barrier).
    """
    n = len(candles)
    labels = np.full(n, np.nan)
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    closes = [float(c["close"]) for c in candles]

    for i in range(n - max_holding_bars):
        entry_price = closes[i]
        outcome = 0

        if direction == "BUY":
            tp_barrier = entry_price + tp_points
            sl_barrier = entry_price - sl_points

            for k in range(i + 1, min(i + max_holding_bars + 1, n)):
                if highs[k] >= tp_barrier:
                    outcome = 1
                    break
                if lows[k] <= sl_barrier:
                    outcome = 0
                    break
            else:
                # Time barrier: if in profit at expiration, 1, else 0
                outcome = 1 if closes[min(i + max_holding_bars, n - 1)] > entry_price else 0

        else:  # SELL
            tp_barrier = entry_price - tp_points
            sl_barrier = entry_price + sl_points

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
    Trained Gradient Boosting Meta-Classifier for Gold trade probability filtering.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = os.path.expanduser(model_path or DEFAULT_MODEL_PATH)
        self.model: Optional[HistGradientBoostingClassifier] = None
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
                logger.info(f"Loaded cached Gold ML model from {self.model_path}")
                return True
            except Exception as e:
                logger.warning(f"Could not load cached model: {e}")
        return False

    def train_model(
        self,
        symbol: str = "XAUUSD",
        period: str = "2y",
        interval: str = "1h",
        tp_points: float = 2.00,
        sl_points: float = 1.54,
    ) -> Dict[str, Any]:
        """
        Trains the Meta-Classifier on historical Gold candles using time-series cross-validation.
        """
        logger.info(f"Fetching {period} {interval} data for Gold ML training...")
        candles = fetch_gold_candles(symbol, period=period, interval=interval)
        if len(candles) < 400:
            return {"error": f"Insufficient candles ({len(candles)}) for ML training. Minimum 400 needed."}

        macro_data = analyze_macro_drivers()
        macro_score = float(macro_data.get("macro_gold_score", 0.0))

        X_all, feature_names = extract_candle_features(candles, macro_score=macro_score)

        # Label both BUY and SELL sides to allow unified directional probability estimation
        y_buy = label_triple_barrier(candles, direction="BUY", tp_points=tp_points, sl_points=sl_points)
        y_sell = label_triple_barrier(candles, direction="SELL", tp_points=tp_points, sl_points=sl_points)

        # Filter valid rows
        valid_mask_buy = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_buy)
        valid_mask_sell = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_sell)

        X_buy_clean = X_all[valid_mask_buy]
        y_buy_clean = y_buy[valid_mask_buy]

        X_sell_clean = X_all[valid_mask_sell]
        y_sell_clean = y_sell[valid_mask_sell]

        # Combine datasets with a directional feature
        # Buy: directional_bias = 1.0, Sell: directional_bias = -1.0
        X_buy_aug = np.column_stack([X_buy_clean, np.ones(len(X_buy_clean))])
        X_sell_aug = np.column_stack([X_sell_clean, -np.ones(len(X_sell_clean))])

        X_combined = np.vstack([X_buy_aug, X_sell_aug])
        y_combined = np.concatenate([y_buy_clean, y_sell_clean])

        augmented_feature_names = feature_names + ["trade_direction_sign"]

        # Time-series Split Evaluation (Walk-Forward)
        tscv = TimeSeriesSplit(n_splits=5)
        auc_scores = []
        precision_scores = []

        clf = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=31,
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

        # Final fit on entire dataset
        clf.fit(X_combined, y_combined)

        # High-probability bucket performance (P >= 0.70)
        final_probas = clf.predict_proba(X_combined)[:, 1]
        high_prob_mask = final_probas >= 0.70
        high_prob_win_rate = (
            round(float(np.mean(y_combined[high_prob_mask]) * 100), 1)
            if np.any(high_prob_mask)
            else 0.0
        )
        high_prob_trade_count = int(np.sum(high_prob_mask))

        # Persist model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        metadata = {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "samples_count": len(X_combined),
            "tp_points": tp_points,
            "sl_points": sl_points,
            "avg_auc": round(float(np.mean(auc_scores)), 3),
            "avg_precision": round(float(np.mean(precision_scores)), 3),
            "high_conviction_win_rate_pct": high_prob_win_rate,
            "high_conviction_trades_found": high_prob_trade_count,
            "trained_at": datetime.now(timezone.utc).isoformat(),
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
        min_probability_threshold: float = 0.70,
    ) -> Dict[str, Any]:
        """
        Evaluate real-time candle state and return the predicted win probability for the trade.
        """
        if self.model is None:
            # Auto-train if not yet trained
            logger.info("ML model not found in cache. Training initial model...")
            self.train_model(period="1y", interval="1h")

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
            }

        dir_sign = 1.0 if direction.upper() == "BUY" else -1.0
        feature_vector = np.append(latest_features, [dir_sign]).reshape(1, -1)

        probas = self.model.predict_proba(feature_vector)[0]
        prob_win = round(float(probas[1]), 3)
        prob_loss = round(float(probas[0]), 3)

        is_high_conviction = prob_win >= min_probability_threshold
        recommendation = (
            "EXECUTE_HIGH_CONVICTION" if is_high_conviction else "PASS_LOW_PROBABILITY"
        )

        return {
            "direction": direction.upper(),
            "probability_win": prob_win,
            "probability_loss": prob_loss,
            "threshold": min_probability_threshold,
            "is_high_conviction": is_high_conviction,
            "recommendation": recommendation,
            "model_trained_at": self.metadata.get("trained_at", "Unknown"),
            "historical_high_conviction_win_rate": self.metadata.get(
                "high_conviction_win_rate_pct", 75.0
            ),
        }
