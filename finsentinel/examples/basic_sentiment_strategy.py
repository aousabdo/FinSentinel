#!/usr/bin/env python
"""
Basic Sentiment Strategy Example

This script demonstrates a simple sentiment-based trading strategy
using the FinSentinel library.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import FinSentinel modules
from finsentinel.data.market_data import get_market_data
from finsentinel.data.text_data import get_text_data, clean_text
from finsentinel.sentiment.llm_analyzer import analyze_sentiment, calculate_aggregate_sentiment
from finsentinel.strategy.backtester import SimpleSentimentStrategy, backtest_strategy
from finsentinel.visualization.dashboard import create_dashboard

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_sample_data():
    """Load sample data for demonstration purposes."""
    # Load sample Reddit posts
    sample_posts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sample_reddit_posts.csv')
    posts_df = pd.read_csv(sample_posts_path)
    
    # Convert created_utc to datetime
    posts_df['created_utc'] = pd.to_datetime(posts_df['created_utc'])
    posts_df['date'] = posts_df['created_utc'].dt.date
    
    # Generate sample market data
    tickers = ['AAPL', 'TSLA', 'MSFT', 'NVDA', 'AMZN', 'META']
    start_date = datetime(2023, 7, 15)
    end_date = datetime(2023, 8, 15)
    
    # Generate synthetic market data for demonstration
    market_data = {}
    for ticker in tickers:
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Initialize with base price
        base_price = {
            'AAPL': 180.0, 'TSLA': 250.0, 'MSFT': 320.0, 
            'NVDA': 430.0, 'AMZN': 130.0, 'META': 290.0
        }.get(ticker, 100.0)
        
        # Generate synthetic price data with random walk
        import numpy as np
        np.random.seed(42 + ord(ticker[0]))  # Different seed for each ticker
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': date_range,
            'close': base_price * (1 + np.cumsum(np.random.normal(0.0005, 0.015, len(date_range))))
        })
        
        # Add other price columns
        df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, 0.005, len(df)))
        df.loc[0, 'open'] = base_price  # Set first open price
        
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.005, len(df))))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.005, len(df))))
        
        # Add volume
        df['volume'] = np.random.randint(1000000, 10000000, len(df))
        
        market_data[ticker] = df
    
    return posts_df, market_data


def preprocess_text_data(posts_df):
    """Preprocess the text data for sentiment analysis."""
    # Clean text data
    posts_df['clean_text'] = posts_df['title'] + ' ' + posts_df['text'].fillna('')
    posts_df['clean_text'] = posts_df['clean_text'].apply(clean_text)
    
    # Group posts by ticker and date
    ticker_mentions = {}
    for ticker in ['AAPL', 'TSLA', 'MSFT', 'NVDA', 'AMZN', 'META']:
        # Filter posts that mention the ticker
        ticker_posts = posts_df[posts_df['clean_text'].str.contains(ticker, case=False)]
        
        if not ticker_posts.empty:
            # Group by date and aggregate
            ticker_posts = ticker_posts.groupby('date').agg({
                'id': 'count',
                'score': 'sum',
                'num_comments': 'sum',
                'clean_text': lambda x: ' '.join(x)
            }).reset_index()
            
            ticker_posts.rename(columns={'id': 'post_count'}, inplace=True)
            ticker_mentions[ticker] = ticker_posts
    
    return ticker_mentions


def run_sentiment_analysis(ticker_mentions):
    """Run sentiment analysis on the text data."""
    # Mock sentiment analysis (in a real scenario, you would use the LLM-based analyzer)
    sentiment_data = {}
    
    for ticker, mentions_df in ticker_mentions.items():
        # Create synthetic sentiment scores for demonstration
        mentions_df['sentiment_score'] = None
        
        # For each day's text, assign a sentiment score based on keywords
        for idx, row in mentions_df.iterrows():
            text = row['clean_text'].lower()
            
            # Simple keyword-based sentiment (for demonstration only)
            positive_keywords = ['beat', 'bullish', 'growth', 'exceeded', 'valuable', 'upside']
            negative_keywords = ['concerned', 'risk', 'slowing', 'issues', 'layoffs', 'bubble']
            
            positive_count = sum(1 for word in positive_keywords if word in text)
            negative_count = sum(1 for word in negative_keywords if word in text)
            
            # Calculate a simple sentiment score between -3 and 3
            score = (positive_count - negative_count) * 0.75
            score = max(min(score, 3), -3)  # Clamp between -3 and 3
            
            mentions_df.loc[idx, 'sentiment_score'] = score
            mentions_df.loc[idx, 'confidence'] = 0.7  # Mock confidence score
        
        # Format the DataFrame for the backtester
        sentiment_df = pd.DataFrame({
            'date': mentions_df['date'],
            'sentiment_score': mentions_df['sentiment_score'],
            'confidence': mentions_df['confidence'],
            'post_count': mentions_df['post_count'],
            'weighted_sentiment': mentions_df['sentiment_score'] * mentions_df['confidence']
        })
        
        # Convert date strings to datetime
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        sentiment_data[ticker] = sentiment_df
    
    return sentiment_data


def run_backtest(market_data, sentiment_data):
    """Run a backtest of the sentiment strategy."""
    # Run backtest
    results = backtest_strategy(
        strategy_cls=SimpleSentimentStrategy,
        market_data=market_data,
        sentiment_data=sentiment_data,
        sentiment_threshold=0.5,  # Signal threshold
        initial_capital=100000.0,
        sentiment_field='weighted_sentiment'
    )
    
    # Print performance metrics
    stats = results['stats']
    print("\nBacktest Results:")
    print(f"Total Return: {stats['total_return']:.2%}")
    print(f"Annualized Return: {stats['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {stats['max_drawdown']:.2%}")
    print(f"Win Rate: {stats['win_rate']:.2%}")
    
    # Plot equity curve
    results['plot'].suptitle('Sentiment Strategy Backtest Results')
    plt.tight_layout()
    plt.show()
    
    return results


def main():
    """Main function to run the example."""
    logger.info("Starting basic sentiment strategy example")
    
    # Load sample data
    logger.info("Loading sample data")
    posts_df, market_data = load_sample_data()
    
    # Preprocess text data
    logger.info("Preprocessing text data")
    ticker_mentions = preprocess_text_data(posts_df)
    
    # Run sentiment analysis
    logger.info("Running sentiment analysis")
    sentiment_data = run_sentiment_analysis(ticker_mentions)
    
    # Run backtest
    logger.info("Running backtest")
    backtest_results = run_backtest(market_data, sentiment_data)
    
    # Create dashboard (uncomment to run)
    # logger.info("Creating dashboard")
    # create_dashboard(market_data, sentiment_data, backtest_results)
    
    logger.info("Example completed successfully")


if __name__ == "__main__":
    main()
