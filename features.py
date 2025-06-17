"""
Feature engineering module for Crypto Forecast Application
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit


class FeatureEngineer:
    """Class for feature engineering and data preparation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = None
        self.feature_columns = []

    def add_technical_indicators(
        self, df: pd.DataFrame, has_volume: bool = True
    ) -> pd.DataFrame:
        """Add technical analysis indicators"""
        self.logger.info("Adding technical indicators")

        # Price-based indicators
        df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=10)
        df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["EMA_10"] = ta.trend.ema_indicator(df["Close"], window=10)
        df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)

        # RSI
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

        # MACD
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_diff"] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["Close"])
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
        df["BB_width"] = bb.bollinger_wband()

        # ATR
        df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])

        # Volume-based indicators (if volume is available)
        if has_volume and "Volume" in df.columns:
            df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
            df["VWAP"] = ta.volume.volume_weighted_average_price(
                df["High"], df["Low"], df["Close"], df["Volume"]
            )

        # Drop NaN values created by indicators
        df = df.dropna()

        return df

    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional price-based features"""
        self.logger.info("Creating price features")

        # Price changes
        df["Price_change"] = df["Close"].pct_change()
        df["High_Low_ratio"] = df["High"] / df["Low"]
        df["Close_Open_ratio"] = df["Close"] / df["Open"]

        # Lagged features
        for lag in [1, 3, 7]:
            df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
            df[f"Volume_lag_{lag}"] = df.get("Volume", 0).shift(lag)

        # Rolling statistics
        for window in [7, 14, 30]:
            df[f"Close_rolling_mean_{window}"] = df["Close"].rolling(window).mean()
            df[f"Close_rolling_std_{window}"] = df["Close"].rolling(window).std()

        # Drop NaN values
        df = df.dropna()

        return df

    def scale_features(
        self, data: np.ndarray, scaler_type: str = "Standard", fit: bool = True
    ) -> np.ndarray:
        """Scale features using specified scaler"""
        if fit:
            if scaler_type == "MinMax":
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()

            return self.scaler.fit_transform(data)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted yet")
            return self.scaler.transform(data)

    def make_windows(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        window_size: int,
        horizon: int,
        stride: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Create sliding windows for time series data"""
        self.logger.info(
            f"Creating windows: size={window_size}, horizon={horizon}, stride={stride}"
        )

        # Select features
        data = df[feature_cols].values
        target_col_idx = feature_cols.index("Close")

        X, y = [], []
        indices = []

        for i in range(0, len(data) - window_size - horizon + 1, stride):
            # Input window
            X.append(data[i : i + window_size])

            # Target values (future Close prices)
            y.append(data[i + window_size : i + window_size + horizon, target_col_idx])

            # Store index for later use
            indices.append(i + window_size)

        return np.array(X), np.array(y), indices

    def prepare_features(
        self, df: pd.DataFrame, scaler_type: str = "Standard", add_ta: bool = True
    ) -> pd.DataFrame:
        """Prepare all features"""
        # Check if volume exists
        has_volume = "Volume" in df.columns and df["Volume"].sum() > 0

        # Add technical indicators
        if add_ta:
            df = self.add_technical_indicators(df, has_volume)

        # Add price features
        df = self.create_price_features(df)

        # Store feature columns (exclude datetime index)
        self.feature_columns = df.columns.tolist()

        return df

    def get_train_val_split(
        self, X: np.ndarray, y: np.ndarray, n_splits: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Create time series train/validation splits"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            splits.append((X_train, X_val, y_train, y_val))

        return splits

    def inverse_transform_predictions(
        self, predictions: np.ndarray, original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Inverse transform scaled predictions"""
        if self.scaler is None:
            return predictions

        # Create dummy array with correct shape for inverse transform
        dummy = np.zeros(original_shape)
        close_idx = self.feature_columns.index("Close")

        # Place predictions in correct position
        dummy[:, close_idx] = predictions.flatten()

        # Inverse transform
        inversed = self.scaler.inverse_transform(dummy)

        return inversed[:, close_idx].reshape(predictions.shape)
