"""
Dashboard module for FinSentinel.

This module provides utilities for creating interactive dashboards
to visualize market data, sentiment analysis, and backtesting results.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback, Input, Output
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)


def create_price_chart(market_data: pd.DataFrame, ticker: str = None, 
                      figsize: tuple = (900, 500)) -> go.Figure:
    """
    Create an interactive price chart with volume bars.
    
    Args:
        market_data: DataFrame with market data (date, open, high, low, close, volume)
        ticker: Ticker symbol for the chart title
        figsize: Figure size (width, height)
        
    Returns:
        Plotly Figure object
    """
    # Create subplot with 2 rows (price and volume)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       row_heights=[0.8, 0.2],
                       subplot_titles=(f"{ticker} Price" if ticker else "Price", "Volume"))
    
    # Make sure the DataFrame is sorted by date
    market_data = market_data.sort_values("date")
    
    # Add price candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=market_data["date"],
            open=market_data["open"],
            high=market_data["high"],
            low=market_data["low"],
            close=market_data["close"],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ["red" if row["close"] < row["open"] else "green" for _, row in market_data.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=market_data["date"],
            y=market_data["volume"],
            marker_color=colors,
            name="Volume"
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        title=f"{ticker} Historical Data" if ticker else "Historical Price Data",
        xaxis_rangeslider_visible=False,
        yaxis_title="Price",
        yaxis2_title="Volume"
    )
    
    return fig


def create_sentiment_chart(sentiment_data: pd.DataFrame, ticker: str = None,
                         figsize: tuple = (900, 500)) -> go.Figure:
    """
    Create an interactive sentiment chart.
    
    Args:
        sentiment_data: DataFrame with sentiment data (date, sentiment_score, etc.)
        ticker: Ticker symbol for the chart title
        figsize: Figure size (width, height)
        
    Returns:
        Plotly Figure object
    """
    # Make sure the DataFrame is sorted by date
    sentiment_data = sentiment_data.sort_values("date")
    
    # Create figure
    fig = go.Figure()
    
    # Add sentiment line
    fig.add_trace(
        go.Scatter(
            x=sentiment_data["date"],
            y=sentiment_data["sentiment_score"],
            mode="lines+markers",
            name="Sentiment Score",
            line=dict(color="blue", width=2),
            marker=dict(
                size=8,
                color=sentiment_data["sentiment_score"],
                colorscale="RdYlGn",
                cmin=-3,
                cmax=3,
                colorbar=dict(title="Score")
            )
        )
    )
    
    # Add confidence bands if available
    if "confidence" in sentiment_data.columns:
        confidence = sentiment_data["confidence"]
        upper_bound = sentiment_data["sentiment_score"] + confidence
        lower_bound = sentiment_data["sentiment_score"] - confidence
        
        fig.add_trace(
            go.Scatter(
                x=sentiment_data["date"],
                y=upper_bound,
                mode="lines",
                line=dict(width=0),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=sentiment_data["date"],
                y=lower_bound,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(0, 0, 255, 0.2)",
                name="Confidence"
            )
        )
    
    # Add zero line
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")
    
    # Add threshold areas if provided in the data
    if "threshold" in sentiment_data.columns:
        threshold = sentiment_data["threshold"].iloc[0]
        
        # Add positive threshold line
        fig.add_hline(y=threshold, line_width=1, line_color="green", line_dash="dash")
        
        # Add negative threshold line
        fig.add_hline(y=-threshold, line_width=1, line_color="red", line_dash="dash")
    
    # Update layout
    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        title=f"{ticker} Sentiment Analysis" if ticker else "Sentiment Analysis",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-3.5, 3.5])
    )
    
    return fig


def create_combined_chart(market_data: pd.DataFrame, sentiment_data: pd.DataFrame,
                         ticker: str = None, figsize: tuple = (1000, 700)) -> go.Figure:
    """
    Create a combined price and sentiment chart.
    
    Args:
        market_data: DataFrame with market data
        sentiment_data: DataFrame with sentiment data
        ticker: Ticker symbol for the chart title
        figsize: Figure size (width, height)
        
    Returns:
        Plotly Figure object
    """
    # Make sure both DataFrames are sorted by date
    market_data = market_data.sort_values("date")
    sentiment_data = sentiment_data.sort_values("date")
    
    # Create subplot with 3 rows (price, volume, sentiment)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                       vertical_spacing=0.05,
                       row_heights=[0.5, 0.15, 0.35],
                       subplot_titles=("Price", "Volume", "Sentiment"))
    
    # Add price candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=market_data["date"],
            open=market_data["open"],
            high=market_data["high"],
            low=market_data["low"],
            close=market_data["close"],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ["red" if row["close"] < row["open"] else "green" for _, row in market_data.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=market_data["date"],
            y=market_data["volume"],
            marker_color=colors,
            name="Volume"
        ),
        row=2, col=1
    )
    
    # Add sentiment line
    fig.add_trace(
        go.Scatter(
            x=sentiment_data["date"],
            y=sentiment_data["sentiment_score"],
            mode="lines+markers",
            name="Sentiment",
            line=dict(color="blue", width=2),
            marker=dict(
                size=6,
                color=sentiment_data["sentiment_score"],
                colorscale="RdYlGn",
                cmin=-3,
                cmax=3
            )
        ),
        row=3, col=1
    )
    
    # Add zero line to sentiment plot
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black", row=3, col=1)
    
    # Add threshold lines if available
    if "threshold" in sentiment_data.columns:
        threshold = sentiment_data["threshold"].iloc[0]
        
        # Add positive threshold line
        fig.add_hline(y=threshold, line_width=1, line_color="green", line_dash="dash", row=3, col=1)
        
        # Add negative threshold line
        fig.add_hline(y=-threshold, line_width=1, line_color="red", line_dash="dash", row=3, col=1)
    
    # Update layout
    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        title=f"{ticker} Price and Sentiment Analysis" if ticker else "Price and Sentiment Analysis",
        xaxis_rangeslider_visible=False,
        yaxis3=dict(range=[-3.5, 3.5])
    )
    
    return fig


def create_backtest_results_chart(equity_curve: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None,
                                figsize: tuple = (900, 600)) -> go.Figure:
    """
    Create a chart showing backtest results.
    
    Args:
        equity_curve: DataFrame with equity curve data (date, equity, daily_return, cumulative_return)
        benchmark_data: Optional DataFrame with benchmark data for comparison
        figsize: Figure size (width, height)
        
    Returns:
        Plotly Figure object
    """
    # Create subplot with 2 rows (equity and drawdown)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.08,
                       row_heights=[0.7, 0.3],
                       subplot_titles=("Portfolio Value", "Drawdown"))
    
    # Add equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_curve["date"],
            y=equity_curve["equity"],
            mode="lines",
            name="Strategy",
            line=dict(color="blue", width=2)
        ),
        row=1, col=1
    )
    
    # Add benchmark if provided
    if benchmark_data is not None:
        # Calculate benchmark returns
        if "cumulative_return" not in benchmark_data.columns:
            benchmark_data["daily_return"] = benchmark_data["close"].pct_change().fillna(0)
            benchmark_data["cumulative_return"] = (1 + benchmark_data["daily_return"]).cumprod() - 1
        
        # Scale benchmark to match initial portfolio value
        initial_equity = equity_curve["equity"].iloc[0]
        benchmark_values = initial_equity * (1 + benchmark_data["cumulative_return"])
        
        fig.add_trace(
            go.Scatter(
                x=benchmark_data["date"],
                y=benchmark_values,
                mode="lines",
                name="Benchmark",
                line=dict(color="gray", width=2)
            ),
            row=1, col=1
        )
    
    # Calculate and add drawdown
    equity_curve["rolling_max"] = equity_curve["equity"].cummax()
    equity_curve["drawdown"] = (equity_curve["rolling_max"] - equity_curve["equity"]) / equity_curve["rolling_max"]
    
    fig.add_trace(
        go.Scatter(
            x=equity_curve["date"],
            y=equity_curve["drawdown"] * -100,  # Convert to percentage and invert
            mode="lines",
            name="Drawdown",
            line=dict(color="red", width=2),
            fill="tozeroy"
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        title="Backtest Results",
        xaxis_rangeslider_visible=False,
        yaxis_title="Portfolio Value ($)",
        yaxis2_title="Drawdown (%)"
    )
    
    return fig


class SentimentDashboard:
    """Interactive dashboard for sentiment analysis and backtesting results."""
    
    def __init__(self, market_data: Dict[str, pd.DataFrame], 
                sentiment_data: Dict[str, pd.DataFrame],
                backtest_results: Optional[Dict[str, Any]] = None,
                title: str = "FinSentinel Dashboard"):
        """
        Initialize the dashboard.
        
        Args:
            market_data: Dictionary mapping tickers to market data DataFrames
            sentiment_data: Dictionary mapping tickers to sentiment DataFrames
            backtest_results: Optional backtest results
            title: Dashboard title
        """
        self.market_data = market_data
        self.sentiment_data = sentiment_data
        self.backtest_results = backtest_results
        self.title = title
        self.tickers = list(market_data.keys())
        
        logger.info(f"Initialized dashboard with {len(self.tickers)} tickers")
    
    def create_layout(self) -> html.Div:
        """
        Create the dashboard layout.
        
        Returns:
            Dash HTML layout
        """
        return html.Div([
            # Header
            html.H1(self.title, style={"textAlign": "center", "marginBottom": "30px"}),
            
            # Ticker selection
            html.Div([
                html.Label("Select Ticker:"),
                dcc.Dropdown(
                    id="ticker-dropdown",
                    options=[{"label": ticker, "value": ticker} for ticker in self.tickers],
                    value=self.tickers[0] if self.tickers else None,
                    style={"width": "200px"}
                ),
            ], style={"marginBottom": "20px"}),
            
            # Date range selection
            html.Div([
                html.Label("Select Date Range:"),
                dcc.DatePickerRange(
                    id="date-range",
                    min_date_allowed=min(df["date"].min() for df in self.market_data.values()),
                    max_date_allowed=max(df["date"].max() for df in self.market_data.values()),
                    start_date=min(df["date"].min() for df in self.market_data.values()),
                    end_date=max(df["date"].max() for df in self.market_data.values())
                ),
            ], style={"marginBottom": "20px"}),
            
            # Combined chart
            html.Div([
                html.H2("Price and Sentiment Analysis", style={"textAlign": "center"}),
                dcc.Graph(id="combined-chart")
            ], style={"marginBottom": "40px"}),
            
            # Backtest results (if available)
            html.Div([
                html.H2("Backtest Results", style={"textAlign": "center"}),
                dcc.Graph(id="backtest-chart")
            ], style={"marginBottom": "40px"}) if self.backtest_results else [],
            
            # Performance metrics table (if available)
            html.Div([
                html.H2("Performance Metrics", style={"textAlign": "center"}),
                html.Div(id="performance-metrics")
            ]) if self.backtest_results else []
        ])
    
    def register_callbacks(self, app: dash.Dash):
        """
        Register dashboard callbacks.
        
        Args:
            app: Dash application instance
        """
        # Callback to update combined chart
        @app.callback(
            Output("combined-chart", "figure"),
            [Input("ticker-dropdown", "value"),
             Input("date-range", "start_date"),
             Input("date-range", "end_date")]
        )
        def update_combined_chart(ticker, start_date, end_date):
            if not ticker or ticker not in self.market_data or ticker not in self.sentiment_data:
                return go.Figure()
            
            # Filter market data by date range
            market_df = self.market_data[ticker]
            market_df = market_df[(market_df["date"] >= start_date) & (market_df["date"] <= end_date)]
            
            # Filter sentiment data by date range
            sentiment_df = self.sentiment_data[ticker]
            sentiment_df = sentiment_df[(sentiment_df["date"] >= start_date) & (sentiment_df["date"] <= end_date)]
            
            # Create combined chart
            return create_combined_chart(market_df, sentiment_df, ticker)
        
        # Callback to update backtest chart (if available)
        if self.backtest_results:
            @app.callback(
                Output("backtest-chart", "figure"),
                [Input("date-range", "start_date"),
                 Input("date-range", "end_date")]
            )
            def update_backtest_chart(start_date, end_date):
                equity_curve = self.backtest_results.get("equity_curve")
                if equity_curve is None:
                    return go.Figure()
                
                # Filter equity curve by date range
                equity_curve = equity_curve[(equity_curve["date"] >= start_date) & (equity_curve["date"] <= end_date)]
                
                # Get benchmark data for comparison
                benchmark_data = None
                if self.tickers and self.tickers[0] in self.market_data:
                    benchmark_data = self.market_data[self.tickers[0]]
                    benchmark_data = benchmark_data[(benchmark_data["date"] >= start_date) & 
                                                  (benchmark_data["date"] <= end_date)]
                
                # Create backtest results chart
                return create_backtest_results_chart(equity_curve, benchmark_data)
            
            # Callback to update performance metrics table
            @app.callback(
                Output("performance-metrics", "children"),
                [Input("date-range", "start_date"),
                 Input("date-range", "end_date")]
            )
            def update_performance_metrics(start_date, end_date):
                stats = self.backtest_results.get("stats")
                if stats is None:
                    return html.Div()
                
                # Create metrics table
                return html.Table([
                    html.Tr([html.Th("Metric"), html.Th("Value")]),
                    html.Tr([html.Td("Total Return"), html.Td(f"{stats['total_return']:.2%}")]),
                    html.Tr([html.Td("Annualized Return"), html.Td(f"{stats['annualized_return']:.2%}")]),
                    html.Tr([html.Td("Sharpe Ratio"), html.Td(f"{stats['sharpe_ratio']:.2f}")]),
                    html.Tr([html.Td("Max Drawdown"), html.Td(f"{stats['max_drawdown']:.2%}")]),
                    html.Tr([html.Td("Win Rate"), html.Td(f"{stats['win_rate']:.2%}")])
                ], style={"margin": "0 auto", "border": "1px solid black", "borderCollapse": "collapse"})
    
    def run_server(self, host: str = "0.0.0.0", port: int = 8050, debug: bool = True):
        """
        Run the dashboard server.
        
        Args:
            host: Server host
            port: Server port
            debug: Whether to run in debug mode
        """
        app = dash.Dash(__name__)
        app.layout = self.create_layout()
        self.register_callbacks(app)
        
        logger.info(f"Starting dashboard server on {host}:{port}")
        app.run_server(host=host, port=port, debug=debug)


def create_dashboard(market_data: Dict[str, pd.DataFrame], 
                    sentiment_data: Dict[str, pd.DataFrame],
                    backtest_results: Optional[Dict[str, Any]] = None,
                    title: str = "FinSentinel Dashboard",
                    host: str = "0.0.0.0", 
                    port: int = 8050) -> SentimentDashboard:
    """
    Convenience function to create and run a sentiment dashboard.
    
    Args:
        market_data: Dictionary mapping tickers to market data DataFrames
        sentiment_data: Dictionary mapping tickers to sentiment DataFrames
        backtest_results: Optional backtest results
        title: Dashboard title
        host: Server host
        port: Server port
        
    Returns:
        SentimentDashboard instance
    """
    dashboard = SentimentDashboard(
        market_data=market_data,
        sentiment_data=sentiment_data,
        backtest_results=backtest_results,
        title=title
    )
    
    dashboard.run_server(host=host, port=port)
    
    return dashboard
