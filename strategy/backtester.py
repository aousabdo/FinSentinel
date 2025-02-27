"""
Backtesting module for FinSentinel.

This module provides utilities for backtesting trading strategies
based on sentiment analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)


class Position:
    """Represents a trading position."""
    
    def __init__(self, ticker: str, shares: float, entry_price: float, 
                entry_date: datetime, exit_price: Optional[float] = None,
                exit_date: Optional[datetime] = None):
        """
        Initialize a position.
        
        Args:
            ticker: Ticker symbol
            shares: Number of shares (positive for long, negative for short)
            entry_price: Entry price per share
            entry_date: Entry date
            exit_price: Exit price per share (if closed)
            exit_date: Exit date (if closed)
        """
        self.ticker = ticker
        self.shares = shares
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.is_open = exit_price is None or exit_date is None
    
    def close(self, exit_price: float, exit_date: datetime):
        """
        Close the position.
        
        Args:
            exit_price: Exit price per share
            exit_date: Exit date
        """
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.is_open = False
    
    def calculate_profit_loss(self, current_price: Optional[float] = None) -> float:
        """
        Calculate profit/loss for the position.
        
        Args:
            current_price: Current price for open positions
            
        Returns:
            Profit/loss in dollar terms
        """
        price = self.exit_price if not self.is_open else current_price
        if price is None:
            return 0.0
        
        return self.shares * (price - self.entry_price)
    
    def calculate_profit_loss_pct(self, current_price: Optional[float] = None) -> float:
        """
        Calculate profit/loss percentage for the position.
        
        Args:
            current_price: Current price for open positions
            
        Returns:
            Profit/loss as a percentage
        """
        price = self.exit_price if not self.is_open else current_price
        if price is None:
            return 0.0
        
        if self.shares > 0:  # Long position
            return (price / self.entry_price) - 1.0
        else:  # Short position
            return 1.0 - (price / self.entry_price)
    
    def __str__(self) -> str:
        """String representation of the position."""
        position_type = "LONG" if self.shares > 0 else "SHORT"
        status = "OPEN" if self.is_open else "CLOSED"
        pnl = self.calculate_profit_loss() if not self.is_open else "N/A"
        pnl_pct = self.calculate_profit_loss_pct() if not self.is_open else "N/A"
        
        return (f"{position_type} {self.ticker} {abs(self.shares)} shares @ {self.entry_price:.2f} "
                f"[{status}] {'' if self.is_open else f'P&L: ${pnl:.2f} ({pnl_pct:.2%})'}")


class Portfolio:
    """Portfolio for tracking positions and performance."""
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize the portfolio.
        
        Args:
            initial_capital: Initial capital in dollars
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.transactions: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = [initial_capital]
        self.dates: List[datetime] = [datetime.now()]
    
    def open_position(self, ticker: str, shares: float, price: float, date: datetime):
        """
        Open a new position.
        
        Args:
            ticker: Ticker symbol
            shares: Number of shares (positive for long, negative for short)
            price: Entry price per share
            date: Entry date
        """
        cost = abs(shares) * price
        if cost > self.capital:
            logger.warning(f"Insufficient capital to open position: {cost} > {self.capital}")
            # Adjust shares to available capital
            shares = (self.capital / price) * (1 if shares > 0 else -1)
            cost = abs(shares) * price
            logger.info(f"Adjusted shares to {shares} based on available capital")
        
        position = Position(ticker=ticker, shares=shares, entry_price=price, entry_date=date)
        self.positions.append(position)
        self.capital -= cost
        
        self.transactions.append({
            "date": date,
            "ticker": ticker,
            "action": "BUY" if shares > 0 else "SELL",
            "shares": abs(shares),
            "price": price,
            "value": cost
        })
        
        logger.info(f"Opened {position}")
    
    def close_position(self, position_idx: int, price: float, date: datetime):
        """
        Close an existing position.
        
        Args:
            position_idx: Index of the position in the positions list
            price: Exit price per share
            date: Exit date
        """
        if position_idx >= len(self.positions):
            logger.error(f"Invalid position index: {position_idx}")
            return
        
        position = self.positions[position_idx]
        position.close(exit_price=price, exit_date=date)
        
        self.capital += abs(position.shares) * price
        self.closed_positions.append(position)
        self.positions.pop(position_idx)
        
        self.transactions.append({
            "date": date,
            "ticker": position.ticker,
            "action": "SELL" if position.shares > 0 else "BUY",
            "shares": abs(position.shares),
            "price": price,
            "value": abs(position.shares) * price
        })
        
        logger.info(f"Closed {position}")
    
    def close_all_positions(self, prices: Dict[str, float], date: datetime):
        """
        Close all open positions.
        
        Args:
            prices: Dictionary mapping tickers to their current prices
            date: Exit date
        """
        for i in range(len(self.positions) - 1, -1, -1):
            position = self.positions[i]
            if position.ticker in prices:
                self.close_position(i, prices[position.ticker], date)
            else:
                logger.warning(f"No price available for {position.ticker}, position remains open")
    
    def calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate the current portfolio value.
        
        Args:
            prices: Dictionary mapping tickers to their current prices
            
        Returns:
            Current portfolio value (capital + positions)
        """
        positions_value = 0.0
        for position in self.positions:
            if position.ticker in prices:
                positions_value += abs(position.shares) * prices[position.ticker]
        
        return self.capital + positions_value
    
    def update_equity_curve(self, date: datetime, prices: Dict[str, float]):
        """
        Update the equity curve with the current portfolio value.
        
        Args:
            date: Current date
            prices: Dictionary mapping tickers to their current prices
        """
        portfolio_value = self.calculate_portfolio_value(prices)
        self.equity_curve.append(portfolio_value)
        self.dates.append(date)
    
    def calculate_returns(self) -> pd.DataFrame:
        """
        Calculate portfolio returns.
        
        Returns:
            DataFrame with daily returns
        """
        equity_df = pd.DataFrame({
            "date": self.dates,
            "equity": self.equity_curve
        })
        equity_df["daily_return"] = equity_df["equity"].pct_change().fillna(0)
        equity_df["cumulative_return"] = (equity_df["equity"] / self.initial_capital) - 1.0
        
        return equity_df
    
    def calculate_statistics(self) -> Dict[str, float]:
        """
        Calculate performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if len(self.equity_curve) < 2:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0
            }
            
        returns_df = self.calculate_returns()
        daily_returns = returns_df["daily_return"].values
        
        # Calculate total return
        total_return = (self.equity_curve[-1] / self.initial_capital) - 1.0
        
        # Calculate trading days
        trading_days = len(daily_returns) - 1  # Subtract initial day
        if trading_days == 0:
            trading_days = 1  # Avoid division by zero
            
        # Annualized return
        annualized_return = ((1 + total_return) ** (252 / trading_days)) - 1
        
        # Sharpe ratio (assuming risk-free rate of 0)
        daily_std = np.std(daily_returns) if len(daily_returns) > 1 else 1e-10
        sharpe_ratio = (np.mean(daily_returns) / daily_std) * np.sqrt(252) if daily_std > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = returns_df["cumulative_return"].values
        max_drawdown = 0.0
        peak = cumulative_returns[0]
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / (1 + peak)
            max_drawdown = max(max_drawdown, drawdown)
        
        # Win rate
        if len(self.closed_positions) > 0:
            winning_trades = sum(1 for position in self.closed_positions 
                               if position.calculate_profit_loss() > 0)
            win_rate = winning_trades / len(self.closed_positions)
        else:
            win_rate = 0.0
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate
        }
    
    def plot_equity_curve(self, figsize: tuple = (12, 6), 
                         title: str = "Portfolio Equity Curve"):
        """
        Plot the equity curve.
        
        Args:
            figsize: Figure size (width, height)
            title: Plot title
        """
        if len(self.equity_curve) < 2:
            logger.warning("Not enough data to plot equity curve")
            return
            
        returns_df = self.calculate_returns()
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(returns_df["date"], returns_df["equity"], label="Portfolio Value")
        
        # Add buy and sell markers from transactions
        for transaction in self.transactions:
            if transaction["action"] == "BUY":
                ax.scatter(transaction["date"], transaction["value"], 
                          marker="^", color="green", s=100)
            else:  # SELL
                ax.scatter(transaction["date"], transaction["value"],
                          marker="v", color="red", s=100)
        
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.grid(True)
        plt.tight_layout()
        return fig
    
    def __str__(self) -> str:
        """String representation of the portfolio."""
        stats = self.calculate_statistics()
        return (f"Portfolio: ${self.calculate_portfolio_value({}):,.2f}\n"
                f"Cash: ${self.capital:,.2f}\n"
                f"Open Positions: {len(self.positions)}\n"
                f"Closed Positions: {len(self.closed_positions)}\n"
                f"Total Return: {stats['total_return']:.2%}\n"
                f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}\n"
                f"Max Drawdown: {stats['max_drawdown']:.2%}")


class SentimentStrategy:
    """Base class for sentiment-based trading strategies."""
    
    def __init__(self, market_data: Dict[str, pd.DataFrame], 
                sentiment_data: Dict[str, pd.DataFrame],
                initial_capital: float = 100000.0):
        """
        Initialize the strategy.
        
        Args:
            market_data: Dictionary mapping tickers to market data DataFrames
            sentiment_data: Dictionary mapping tickers to sentiment DataFrames
            initial_capital: Initial capital for the portfolio
        """
        self.market_data = market_data
        self.sentiment_data = sentiment_data
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.tickers = list(market_data.keys())
        
        logger.info(f"Initialized {self.__class__.__name__} with {len(self.tickers)} tickers")
    
    def generate_signals(self) -> Dict[str, pd.DataFrame]:
        """
        Generate trading signals based on sentiment data.
        
        Returns:
            Dictionary mapping tickers to signal DataFrames
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def run_backtest(self) -> Portfolio:
        """
        Run the backtest.
        
        Returns:
            The final portfolio after running the backtest
        """
        # Generate signals
        signals = self.generate_signals()
        
        # Merge market data with signals
        merged_data = {}
        for ticker in self.tickers:
            if ticker in signals:
                merged_data[ticker] = pd.merge(
                    self.market_data[ticker], 
                    signals[ticker],
                    on="date", 
                    how="inner"
                )
        
        # Sort all data by date
        for ticker in merged_data:
            merged_data[ticker] = merged_data[ticker].sort_values("date")
        
        # Get a combined list of all dates
        all_dates = []
        for ticker in merged_data:
            all_dates.extend(merged_data[ticker]["date"].tolist())
        all_dates = sorted(list(set(all_dates)))
        
        logger.info(f"Running backtest from {all_dates[0]} to {all_dates[-1]}")
        
        # Run the backtest day by day
        for current_date in all_dates:
            # Get current prices for all tickers
            current_prices = {}
            for ticker in merged_data:
                ticker_data = merged_data[ticker]
                ticker_data_on_date = ticker_data[ticker_data["date"] == current_date]
                
                if not ticker_data_on_date.empty:
                    current_prices[ticker] = ticker_data_on_date["close"].values[0]
            
            # Process signals for this date
            self._process_signals_for_date(current_date, merged_data, current_prices)
            
            # Update portfolio equity curve
            self.portfolio.update_equity_curve(current_date, current_prices)
        
        # Close all positions at the end of the backtest
        self.portfolio.close_all_positions(current_prices, all_dates[-1])
        
        logger.info(f"Backtest completed. Final portfolio value: ${self.portfolio.equity_curve[-1]:,.2f}")
        
        return self.portfolio
    
    def _process_signals_for_date(self, date: datetime, merged_data: Dict[str, pd.DataFrame],
                                 prices: Dict[str, float]):
        """
        Process signals for a specific date.
        
        Args:
            date: Current date
            merged_data: Dictionary mapping tickers to merged market and signal data
            prices: Dictionary mapping tickers to their current prices
        """
        # Check for any open positions that need to be closed
        for i in range(len(self.portfolio.positions) - 1, -1, -1):
            position = self.portfolio.positions[i]
            ticker = position.ticker
            
            if ticker in merged_data:
                ticker_data = merged_data[ticker]
                ticker_data_on_date = ticker_data[ticker_data["date"] == date]
                
                if not ticker_data_on_date.empty and ticker in prices:
                    signal = ticker_data_on_date["signal"].values[0]
                    
                    # Close position if signal is opposite to position or neutral
                    if (position.shares > 0 and signal <= 0) or (position.shares < 0 and signal >= 0):
                        self.portfolio.close_position(i, prices[ticker], date)
        
        # Check for new positions to open
        portfolio_value = self.portfolio.calculate_portfolio_value(prices)
        available_capital = self.portfolio.capital
        
        for ticker in merged_data:
            ticker_data = merged_data[ticker]
            ticker_data_on_date = ticker_data[ticker_data["date"] == date]
            
            if not ticker_data_on_date.empty and ticker in prices:
                price = prices[ticker]
                signal = ticker_data_on_date["signal"].values[0]
                
                # Skip if no clear signal
                if signal == 0:
                    continue
                
                # Check if we already have a position in this ticker
                existing_position = False
                for position in self.portfolio.positions:
                    if position.ticker == ticker:
                        existing_position = True
                        break
                
                if not existing_position:
                    # Calculate position size (e.g., equal weighting)
                    max_positions = 5  # Maximum number of concurrent positions
                    position_value = portfolio_value / max_positions
                    
                    # Ensure we have enough capital
                    if available_capital >= position_value:
                        shares = (position_value / price) * (1 if signal > 0 else -1)
                        self.portfolio.open_position(ticker, shares, price, date)
                        available_capital -= position_value


class SimpleSentimentStrategy(SentimentStrategy):
    """
    Simple sentiment-based trading strategy.
    
    Goes long when sentiment score is above threshold, short when below negative threshold.
    """
    
    def __init__(self, market_data: Dict[str, pd.DataFrame], 
                sentiment_data: Dict[str, pd.DataFrame],
                sentiment_threshold: float = 0.5,
                initial_capital: float = 100000.0,
                sentiment_field: str = "weighted_sentiment"):
        """
        Initialize the strategy.
        
        Args:
            market_data: Dictionary mapping tickers to market data DataFrames
            sentiment_data: Dictionary mapping tickers to sentiment DataFrames
            sentiment_threshold: Threshold for sentiment score to generate signals
            initial_capital: Initial capital for the portfolio
            sentiment_field: Field in sentiment data to use for signals
        """
        super().__init__(market_data, sentiment_data, initial_capital)
        self.sentiment_threshold = sentiment_threshold
        self.sentiment_field = sentiment_field
        
        logger.info(f"Simple Sentiment Strategy with threshold {sentiment_threshold}")
    
    def generate_signals(self) -> Dict[str, pd.DataFrame]:
        """
        Generate trading signals based on sentiment scores.
        
        Returns:
            Dictionary mapping tickers to signal DataFrames
        """
        signals = {}
        
        for ticker in self.tickers:
            if ticker in self.sentiment_data:
                # Get sentiment data for this ticker
                sentiment_df = self.sentiment_data[ticker].copy()
                
                # Generate signals based on sentiment score
                sentiment_df["signal"] = 0  # Neutral by default
                
                # Long signal when sentiment is above threshold
                sentiment_df.loc[sentiment_df[self.sentiment_field] >= self.sentiment_threshold, "signal"] = 1
                
                # Short signal when sentiment is below negative threshold
                sentiment_df.loc[sentiment_df[self.sentiment_field] <= -self.sentiment_threshold, "signal"] = -1
                
                signals[ticker] = sentiment_df[["date", "signal"]]
                
                logger.info(f"Generated {sum(sentiment_df['signal'] != 0)} signals for {ticker}")
        
        return signals


def backtest_strategy(strategy_cls: Any, market_data: Dict[str, pd.DataFrame],
                     sentiment_data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
    """
    Convenience function to backtest a strategy.
    
    Args:
        strategy_cls: Strategy class to instantiate
        market_data: Dictionary mapping tickers to market data DataFrames
        sentiment_data: Dictionary mapping tickers to sentiment DataFrames
        **kwargs: Additional parameters for the strategy
        
    Returns:
        Dictionary with backtest results
    """
    # Instantiate the strategy
    strategy = strategy_cls(market_data=market_data, sentiment_data=sentiment_data, **kwargs)
    
    # Run the backtest
    portfolio = strategy.run_backtest()
    
    # Calculate performance metrics
    stats = portfolio.calculate_statistics()
    
    # Generate equity curve plot
    fig = portfolio.plot_equity_curve()
    
    # Prepare results
    results = {
        "portfolio": portfolio,
        "stats": stats,
        "equity_curve": portfolio.calculate_returns(),
        "transactions": portfolio.transactions,
        "plot": fig
    }
    
    return results
