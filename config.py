"""
Configuration module for Crypto Forecast Application
"""

from typing import List, Tuple
from pydantic import Field
from pydantic_settings import BaseSettings
from datetime import datetime, timedelta


class AppConfig(BaseSettings):
    """Application configuration using Pydantic BaseSettings"""

    # Data sources
    data_sources: List[str] = ["Yahoo Finance", "Binance"]
    default_source: str = "Yahoo Finance"

    # Cryptocurrencies
    crypto_pairs: List[str] = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD"]
    default_crypto: str = "BTC-USD"

    # Time frames
    time_frames: List[str] = ["1d", "1h", "4h"]
    default_timeframe: str = "1d"

    # Model parameters
    default_window_size: int = 60
    default_horizon: int = 7
    default_stride: int = 1
    default_history_years: int = 2

    # Macro indicators
    macro_indicators: dict = Field(
        default_factory=lambda: {
            "DX-Y.NYB": {"name": "Dollar Index", "freq": "1d"},
            "GC=F": {"name": "Gold Futures", "freq": "1d"},
            "^VIX": {"name": "VIX", "freq": "1d"},
            "^GSPC": {"name": "S&P 500", "freq": "1d"},
        }
    )

    # Model settings
    model_types: List[str] = ["LSTM", "GRU"]
    default_model: str = "LSTM"
    scaler_types: List[str] = ["MinMax", "Standard"]
    default_scaler: str = "Standard"

    # Hyperparameter search ranges
    hp_layers: Tuple[int, int] = (1, 3)
    hp_units: Tuple[int, int] = (64, 256)
    hp_dropout: Tuple[float, float] = (0.0, 0.4)
    hp_lr: Tuple[float, float] = (1e-4, 3e-3)
    hp_max_iterations: int = 20

    # Training settings
    early_stopping_patience: int = 15
    batch_size: int = 32
    max_epochs: int = 100

    # Binance settings
    binance_rate_limit: float = 0.2  # seconds between requests
    binance_max_bars: int = 1000

    # Cache settings
    cache_dir: str = "cache"
    use_cache: bool = True

    # Logging
    log_file: str = "crypto_forecast.log"
    log_level: str = "INFO"

    # SHAP settings
    shap_max_samples: int = 100
    shap_timeout: int = 20  # seconds

    class Config:
        env_prefix = "CRYPTO_"


# Global config instance
config = AppConfig()
