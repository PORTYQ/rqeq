"""
Data loading module for Crypto Forecast Application
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
from config import config
from utils import (
    get_cache_key,
    save_to_cache,
    load_from_cache,
    show_toast,
    format_timeframe,
    align_calendar,
    check_volume_column,
    prepare_data_for_scaling,
)


class DataLoader:
    """Class for loading cryptocurrency and macro indicator data"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.binance = None

    def _init_binance(self) -> bool:
        """Initialize Binance connection"""
        try:
            if self.binance is None:
                self.binance = ccxt.binance()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance: {e}")
            return False

    def load_crypto_yahoo(
        self, ticker: str, start_date: datetime, end_date: datetime, interval: str
    ) -> Optional[pd.DataFrame]:
        """Load cryptocurrency data from Yahoo Finance"""
        try:
            self.logger.info(f"Loading {ticker} from Yahoo Finance")

            # Convert interval format
            yf_interval = interval.replace("d", "d").replace("h", "h")

            # Download data
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=yf_interval,
                progress=False,
            )

            if data.empty:
                self.logger.warning(f"No data received from Yahoo Finance for {ticker}")
                return None

            # Rename columns to standard format
            data = data.rename(
                columns={
                    "Open": "Open",
                    "High": "High",
                    "Low": "Low",
                    "Close": "Close",
                    "Volume": "Volume",
                }
            )

            # Keep only OHLCV columns
            data = data[["Open", "High", "Low", "Close", "Volume"]]

            return data

        except Exception as e:
            self.logger.error(f"Error loading from Yahoo Finance: {e}")
            return None

    def load_crypto_binance(
        self, ticker: str, start_date: datetime, end_date: datetime, interval: str
    ) -> Optional[pd.DataFrame]:
        """Load cryptocurrency data from Binance"""
        try:
            if not self._init_binance():
                return None

            self.logger.info(f"Loading {ticker} from Binance")

            # Convert ticker format (BTC-USD -> BTC/USDT)
            symbol = ticker.replace("-USD", "/USDT")

            # Convert interval to Binance format
            binance_interval = interval.replace("d", "d").replace("h", "h")

            # Load data in chunks due to API limits
            all_data = []
            current_time = int(start_date.timestamp() * 1000)
            end_time = int(end_date.timestamp() * 1000)

            while current_time < end_time:
                try:
                    ohlcv = self.binance.fetch_ohlcv(
                        symbol,
                        binance_interval,
                        since=current_time,
                        limit=config.binance_max_bars,
                    )

                    if not ohlcv:
                        break

                    all_data.extend(ohlcv)
                    current_time = ohlcv[-1][0] + 1

                    # Rate limiting
                    time.sleep(config.binance_rate_limit)

                except Exception as e:
                    self.logger.error(f"Error fetching Binance data: {e}")
                    break

            if not all_data:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(
                all_data,
                columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
            )

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            return df

        except Exception as e:
            self.logger.error(f"Error loading from Binance: {e}")
            return None

    def load_macro_indicators(
        self, start_date: datetime, end_date: datetime, interval: str
    ) -> Dict[str, pd.DataFrame]:
        """Load macro indicators data"""
        macro_data = {}

        for ticker, info in config.macro_indicators.items():
            self.logger.info(f"Loading macro indicator: {info['name']}")

            try:
                # Check if interval is compatible with indicator frequency
                if interval in ["1h", "4h"] and info["freq"] == "1d":
                    # For intraday intervals, limit to 60 days for some indicators
                    if ticker in ["GC=F", "^GSPC"]:
                        adj_start = max(start_date, end_date - timedelta(days=60))
                    else:
                        self.logger.warning(
                            f"Skipping {info['name']} - not available for {interval}"
                        )
                        continue
                else:
                    adj_start = start_date

                # Download data
                data = yf.download(
                    ticker,
                    start=adj_start,
                    end=end_date,
                    interval=interval if interval != "4h" else "1h",
                    progress=False,
                )

                if not data.empty:
                    # Keep only Close price and rename
                    macro_data[f"Close_{info['name'].replace(' ', '_')}"] = data[
                        "Close"
                    ]
                else:
                    self.logger.warning(f"No data for {info['name']}")

            except Exception as e:
                self.logger.error(f"Error loading {info['name']}: {e}")

        return macro_data

    def load_data(
        self,
        source: str,
        ticker: str,
        history_years: int,
        interval: str,
        add_macro: bool,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Main method to load all data"""
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=history_years * 365)

        # Generate cache key
        cache_key = get_cache_key(
            source, ticker, f"{history_years}y", interval, add_macro
        )

        # Try to load from cache
        if use_cache:
            cached_data = load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

        # Load cryptocurrency data
        if source == "Yahoo Finance":
            crypto_data = self.load_crypto_yahoo(ticker, start_date, end_date, interval)

            # Fallback to Binance if Yahoo fails
            if crypto_data is None:
                show_toast(
                    "Yahoo Finance data not available, switching to Binance", "warning"
                )
                crypto_data = self.load_crypto_binance(
                    ticker, start_date, end_date, interval
                )
        else:
            crypto_data = self.load_crypto_binance(
                ticker, start_date, end_date, interval
            )

            # Fallback to Yahoo if Binance fails
            if crypto_data is None:
                show_toast(
                    "Binance not available, switching to Yahoo Finance (no volume)",
                    "warning",
                )
                crypto_data = self.load_crypto_yahoo(
                    ticker, start_date, end_date, interval
                )

        if crypto_data is None:
            show_toast("Failed to load cryptocurrency data", "error")
            return None

        # Align calendar
        freq = format_timeframe(interval)
        crypto_data = align_calendar(crypto_data, freq, ticker)

        # Check volume column
        crypto_data = check_volume_column(crypto_data)

        # Load macro indicators if requested
        if add_macro:
            macro_data = self.load_macro_indicators(start_date, end_date, interval)

            # Merge macro data with crypto data
            for col_name, series in macro_data.items():
                if len(series) > 0:
                    # Align index
                    series = series.reindex(crypto_data.index, method="ffill")
                    crypto_data[col_name] = series

        # Prepare data for scaling
        crypto_data = prepare_data_for_scaling(crypto_data)

        # Save to cache
        if use_cache:
            save_to_cache(crypto_data, cache_key)

        return crypto_data
