"""
Market data collection module for FinSentinel.

This module provides utilities for collecting financial market data
from various sources like Yahoo Finance, Alpha Vantage, etc.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import quandl
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)

class MarketDataFetcher:
    """Base class for fetching market data from various sources."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the market data fetcher.
        
        Args:
            api_key: API key for the data source (if required)
        """
        self.api_key = api_key
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def fetch_data(self, ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        """
        Fetch market data for a given ticker and time range.
        
        Args:
            ticker: Stock symbol or ticker
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data frequency (e.g., "1d" for daily)
            
        Returns:
            DataFrame with market data
        """
        raise NotImplementedError("Subclasses must implement this method")


class YahooFinanceFetcher(MarketDataFetcher):
    """Fetch market data from Yahoo Finance using yfinance."""
    
    def fetch_data(self, ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        """
        Fetch market data from Yahoo Finance.
        
        Args:
            ticker: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data frequency (1d, 1wk, 1mo)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
                
            # Clean and process data
            data = data.reset_index()
            data.columns = [col if col != "Date" else "date" for col in data.columns]
            data.columns = [col if col != "Adj Close" else "adj_close" for col in data.columns]
            data.columns = [col.lower() for col in data.columns]
            
            logger.info(f"Successfully fetched {len(data)} records for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()


class QuandlFetcher(MarketDataFetcher):
    """Fetch market data from Quandl."""
    
    def __init__(self, api_key: str):
        """
        Initialize Quandl fetcher with API key.
        
        Args:
            api_key: Quandl API key
        """
        super().__init__(api_key)
        quandl.ApiConfig.api_key = api_key
    
    def fetch_data(self, ticker: str, start_date: str, end_date: str, interval: str = "daily") -> pd.DataFrame:
        """
        Fetch market data from Quandl.
        
        Args:
            ticker: Dataset code (e.g., "WIKI/AAPL" for Apple stock)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Frequency (daily, weekly, monthly, quarterly, annual)
            
        Returns:
            DataFrame with market data
        """
        try:
            logger.info(f"Fetching Quandl data for {ticker} from {start_date} to {end_date}")
            data = quandl.get(ticker, start_date=start_date, end_date=end_date, collapse=interval)
            
            if data.empty:
                logger.warning(f"No Quandl data found for {ticker}")
                return pd.DataFrame()
            
            # Process data
            data = data.reset_index()
            data.columns = [col.lower() for col in data.columns]
            
            logger.info(f"Successfully fetched {len(data)} Quandl records for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Quandl data for {ticker}: {str(e)}")
            return pd.DataFrame()


def get_market_data(source: str = "yahoo", 
                   tickers: List[str] = None, 
                   start_date: str = None,
                   end_date: str = None,
                   api_key: str = None) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to fetch market data from the specified source.
    
    Args:
        source: Data source ("yahoo", "quandl", "alpha_vantage", "fmp")
        tickers: List of ticker symbols
        start_date: Start date (defaults to 1 year ago)
        end_date: End date (defaults to today)
        api_key: API key for the data source (if required)
        
    Returns:
        Dictionary mapping tickers to their respective DataFrames
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create appropriate fetcher based on source
    if source.lower() == "yahoo":
        fetcher = YahooFinanceFetcher()
    elif source.lower() == "quandl":
        if api_key is None:
            raise ValueError("API key is required for Quandl")
        fetcher = QuandlFetcher(api_key)
    else:
        raise ValueError(f"Unsupported data source: {source}")
    
    # Fetch data for each ticker
    results = {}
    for ticker in tickers:
        data = fetcher.fetch_data(ticker, start_date, end_date)
        if not data.empty:
            results[ticker] = data
    
    return results
