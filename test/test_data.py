"""
Tests for data loading module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import DataLoader
from config import config


class TestDataLoader:
    """Test cases for DataLoader class"""

    @pytest.fixture
    def data_loader(self):
        """Create DataLoader instance"""
        return DataLoader()

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        data = pd.DataFrame(
            {
                "Open": np.random.uniform(40000, 45000, len(dates)),
                "High": np.random.uniform(45000, 50000, len(dates)),
                "Low": np.random.uniform(35000, 40000, len(dates)),
                "Close": np.random.uniform(40000, 45000, len(dates)),
                "Volume": np.random.uniform(1000, 10000, len(dates)),
            },
            index=dates,
        )
        return data

    def test_init(self, data_loader):
        """Test DataLoader initialization"""
        assert data_loader.logger is not None
        assert data_loader.binance is None

    @patch("yfinance.download")
    def test_load_crypto_yahoo_success(self, mock_download, data_loader, sample_data):
        """Test successful Yahoo Finance data loading"""
        mock_download.return_value = sample_data

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        result = data_loader.load_crypto_yahoo("BTC-USD", start_date, end_date, "1d")

        assert result is not None
        assert len(result) == len(sample_data)
        assert all(
            col in result.columns for col in ["Open", "High", "Low", "Close", "Volume"]
        )

    @patch("yfinance.download")
    def test_load_crypto_yahoo_empty(self, mock_download, data_loader):
        """Test Yahoo Finance returning empty data"""
        mock_download.return_value = pd.DataFrame()

        result = data_loader.load_crypto_yahoo(
            "BTC-USD", datetime.now(), datetime.now(), "1d"
        )

        assert result is None

    @patch("ccxt.binance")
    def test_load_crypto_binance_success(self, mock_binance, data_loader):
        """Test successful Binance data loading"""
        # Mock Binance API response
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = [
            [1672531200000, 42000, 43000, 41000, 42500, 1000],
            [1672617600000, 42500, 44000, 42000, 43500, 1200],
        ]
        mock_binance.return_value = mock_exchange

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)

        result = data_loader.load_crypto_binance("BTC-USD", start_date, end_date, "1d")

        assert result is not None
        assert len(result) > 0
        assert all(
            col in result.columns for col in ["Open", "High", "Low", "Close", "Volume"]
        )

    @patch("yfinance.download")
    def test_load_macro_indicators(self, mock_download, data_loader, sample_data):
        """Test macro indicators loading"""
        mock_download.return_value = sample_data[["Close"]]

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        result = data_loader.load_macro_indicators(start_date, end_date, "1d")

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_align_calendar(self, data_loader, sample_data):
        """Test calendar alignment"""
        # Remove some dates to create gaps
        data_with_gaps = sample_data.iloc[::2]  # Take every other day

        from utils import align_calendar

        result = align_calendar(data_with_gaps, "D", "TEST")

        # Check that gaps are filled
        assert len(result) > len(data_with_gaps)
        assert result.index.freq == "D"

    def test_check_volume_column(self, data_loader):
        """Test zero volume handling"""
        from utils import check_volume_column

        # Create data with zero volume
        data = pd.DataFrame(
            {"Open": [100, 101, 102], "Close": [101, 102, 103], "Volume": [0, 0, 0]}
        )

        result = check_volume_column(data)

        assert "Volume" not in result.columns

    @patch("data.load_from_cache")
    @patch("data.save_to_cache")
    @patch("yfinance.download")
    def test_caching(
        self, mock_download, mock_save, mock_load, data_loader, sample_data
    ):
        """Test data caching functionality"""
        # First call - cache miss
        mock_load.return_value = None
        mock_download.return_value = sample_data

        result1 = data_loader.load_data(
            "Yahoo Finance", "BTC-USD", 2, "1d", False, use_cache=True
        )

        assert mock_save.called
        assert result1 is not None

        # Second call - cache hit
        mock_load.return_value = sample_data

        result2 = data_loader.load_data(
            "Yahoo Finance", "BTC-USD", 2, "1d", False, use_cache=True
        )

        assert result2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
