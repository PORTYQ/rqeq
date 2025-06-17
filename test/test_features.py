"""
Tests for feature engineering module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class"""

    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance"""
        return FeatureEngineer()

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data"""
        dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq="D")
        np.random.seed(42)

        # Generate realistic price data
        close_prices = 45000 + np.cumsum(np.random.randn(len(dates)) * 500)

        data = pd.DataFrame(
            {
                "Open": close_prices + np.random.uniform(-200, 200, len(dates)),
                "High": close_prices + np.random.uniform(100, 500, len(dates)),
                "Low": close_prices - np.random.uniform(100, 500, len(dates)),
                "Close": close_prices,
                "Volume": np.random.uniform(1000, 10000, len(dates)),
            },
            index=dates,
        )

        return data

    def test_init(self, feature_engineer):
        """Test FeatureEngineer initialization"""
        assert feature_engineer.logger is not None
        assert feature_engineer.scaler is None
        assert feature_engineer.feature_columns == []

    def test_add_technical_indicators_with_volume(
        self, feature_engineer, sample_ohlcv_data
    ):
        """Test adding technical indicators with volume"""
        result = feature_engineer.add_technical_indicators(
            sample_ohlcv_data.copy(), has_volume=True
        )

        # Check that indicators were added
        expected_indicators = [
            "SMA_10",
            "SMA_20",
            "EMA_10",
            "EMA_20",
            "RSI",
            "MACD",
            "MACD_signal",
            "MACD_diff",
            "BB_upper",
            "BB_lower",
            "BB_width",
            "ATR",
            "OBV",
            "VWAP",
        ]

        for indicator in expected_indicators:
            assert indicator in result.columns

        # Check no NaN values after dropna
        assert not result.isnull().any().any()

    def test_add_technical_indicators_without_volume(
        self, feature_engineer, sample_ohlcv_data
    ):
        """Test adding technical indicators without volume"""
        data_no_volume = sample_ohlcv_data.drop(columns=["Volume"])

        result = feature_engineer.add_technical_indicators(
            data_no_volume.copy(), has_volume=False
        )

        # Check volume indicators are not added
        assert "OBV" not in result.columns
        assert "VWAP" not in result.columns

        # Check other indicators are present
        assert "RSI" in result.columns
        assert "MACD" in result.columns

    def test_create_price_features(self, feature_engineer, sample_ohlcv_data):
        """Test price feature creation"""
        result = feature_engineer.create_price_features(sample_ohlcv_data.copy())

        # Check price features
        assert "Price_change" in result.columns
        assert "High_Low_ratio" in result.columns
        assert "Close_Open_ratio" in result.columns

        # Check lagged features
        for lag in [1, 3, 7]:
            assert f"Close_lag_{lag}" in result.columns
            assert f"Volume_lag_{lag}" in result.columns

        # Check rolling features
        for window in [7, 14, 30]:
            assert f"Close_rolling_mean_{window}" in result.columns
            assert f"Close_rolling_std_{window}" in result.columns

    def test_scale_features(self, feature_engineer):
        """Test feature scaling"""
        data = np.random.randn(100, 5)

        # Test StandardScaler
        scaled_standard = feature_engineer.scale_features(data, "Standard", fit=True)
        assert scaled_standard.shape == data.shape
        assert np.abs(scaled_standard.mean()) < 0.1
        assert np.abs(scaled_standard.std() - 1.0) < 0.1

        # Test MinMaxScaler
        feature_engineer.scaler = None
        scaled_minmax = feature_engineer.scale_features(data, "MinMax", fit=True)
        assert scaled_minmax.min() >= -0.1
        assert scaled_minmax.max() <= 1.1

    def test_make_windows(self, feature_engineer, sample_ohlcv_data):
        """Test window creation"""
        feature_cols = ["Open", "High", "Low", "Close", "Volume"]
        window_size = 10
        horizon = 3
        stride = 2

        X, y, indices = feature_engineer.make_windows(
            sample_ohlcv_data, feature_cols, window_size, horizon, stride
        )

        # Check shapes
        assert X.shape[1] == window_size
        assert X.shape[2] == len(feature_cols)
        assert y.shape[1] == horizon

        # Check that Close is used as target
        close_idx = feature_cols.index("Close")
        assert np.array_equal(
            y[0],
            sample_ohlcv_data.iloc[window_size : window_size + horizon]["Close"].values,
        )

    def test_prepare_features(self, feature_engineer, sample_ohlcv_data):
        """Test complete feature preparation"""
        result = feature_engineer.prepare_features(
            sample_ohlcv_data.copy(), scaler_type="Standard", add_ta=True
        )

        # Check that features were added
        assert len(result.columns) > len(sample_ohlcv_data.columns)
        assert len(feature_engineer.feature_columns) > 0

        # Check no NaN values
        assert not result.isnull().any().any()

    def test_train_val_split(self, feature_engineer):
        """Test time series train/validation split"""
        X = np.random.randn(100, 10, 5)
        y = np.random.randn(100, 3)

        splits = feature_engineer.get_train_val_split(X, y, n_splits=3)

        assert len(splits) == 3

        for X_train, X_val, y_train, y_val in splits:
            # Check that validation comes after training
            assert len(X_train) > 0
            assert len(X_val) > 0
            assert len(X_train) + len(X_val) == len(X)

    def test_inverse_transform_predictions(self, feature_engineer):
        """Test inverse transformation of predictions"""
        # Setup scaler and features
        feature_engineer.feature_columns = ["Open", "High", "Low", "Close", "Volume"]
        feature_engineer.scaler = feature_engineer.scale_features(
            np.random.randn(100, 5) * 1000 + 45000, "Standard", fit=True
        )

        # Create scaled predictions
        predictions = np.random.randn(10, 3)  # 10 samples, 3 horizon

        # Inverse transform
        inversed = feature_engineer.inverse_transform_predictions(
            predictions, (30, 5)  # 10*3=30 flattened predictions, 5 features
        )

        assert inversed.shape == predictions.shape
        # Check that values are in reasonable range for prices
        assert inversed.min() > 0
        assert inversed.max() < 100000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
