"""
Utility functions for Crypto Forecast Application
"""

import logging
import os
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st


def setup_logging(log_file: str = "crypto_forecast.log", level: str = "INFO") -> None:
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level),
        format=log_format,
        handlers=[logging.FileHandler(f"logs/{log_file}"), logging.StreamHandler()],
    )


def get_cache_key(
    source: str, ticker: str, period: str, interval: str, macro_flag: bool
) -> str:
    """Generate cache key for data"""
    key_string = f"{source}_{ticker}_{period}_{interval}_{macro_flag}"
    return hashlib.md5(key_string.encode()).hexdigest()


def save_to_cache(data: pd.DataFrame, key: str, cache_dir: str = "cache") -> None:
    """Save DataFrame to cache"""
    Path(cache_dir).mkdir(exist_ok=True)
    cache_path = Path(cache_dir) / f"{key}.parquet"
    data.to_parquet(cache_path)
    logging.info(f"Saved data to cache: {cache_path}")


def load_from_cache(key: str, cache_dir: str = "cache") -> Optional[pd.DataFrame]:
    """Load DataFrame from cache"""
    cache_path = Path(cache_dir) / f"{key}.parquet"
    if cache_path.exists():
        logging.info(f"Loading data from cache: {cache_path}")
        return pd.read_parquet(cache_path)
    return None


def save_model_weights(
    model: Any, params: Dict[str, Any], model_name: str = "best_model"
) -> None:
    """Save model weights and parameters"""
    Path("models").mkdir(exist_ok=True)

    # Save model weights
    model.save(f"models/{model_name}.h5")

    # Save parameters
    with open(f"models/{model_name}_params.pkl", "wb") as f:
        pickle.dump(params, f)

    logging.info(f"Saved model to models/{model_name}.h5")


def load_model_weights(
    model_name: str = "best_model",
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Load model weights and parameters"""
    from tensorflow.keras.models import load_model

    model_path = f"models/{model_name}.h5"
    params_path = f"models/{model_name}_params.pkl"

    if not os.path.exists(model_path) or not os.path.exists(params_path):
        return None, None

    model = load_model(model_path)
    with open(params_path, "rb") as f:
        params = pickle.load(f)

    logging.info(f"Loaded model from {model_path}")
    return model, params


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def show_toast(message: str, type: str = "info") -> None:
    """Show toast notification in Streamlit"""
    if type == "error":
        st.error(message)
    elif type == "warning":
        st.warning(message)
    elif type == "success":
        st.success(message)
    else:
        st.info(message)


def format_timeframe(timeframe: str) -> str:
    """Convert timeframe to frequency string"""
    mapping = {"1d": "D", "1h": "H", "4h": "4H"}
    return mapping.get(timeframe, "D")


def align_calendar(df: pd.DataFrame, freq: str, ticker: str) -> pd.DataFrame:
    """Align data to regular calendar frequency"""
    original_len = len(df)

    # Convert to regular frequency
    df = df.asfreq(freq)

    # Fill missing values
    df = df.ffill().bfill()

    missing_count = len(df) - original_len
    if missing_count > 0:
        logging.info(f"Inserted {missing_count} missing timestamps for {ticker}")

    return df


def check_volume_column(df: pd.DataFrame) -> pd.DataFrame:
    """Check and handle zero volume column"""
    if "Volume" in df.columns:
        # Check if all volume values are zero
        try:
            # Check if all values are 0 or NaN
            all_zero = (df["Volume"].fillna(0) == 0).all()
            if all_zero:
                logging.warning("Volume column contains only zeros, removing it")
                df = df.drop(columns=["Volume"])
        except Exception as e:
            logging.warning(f"Error checking volume column: {e}")
    return df


def prepare_data_for_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for scaling - handle missing values"""
    # Forward fill
    df = df.ffill()

    # Backward fill
    df = df.bfill()

    # Interpolate remaining NaN values
    df = df.interpolate(method="linear", limit_direction="both")

    return df
