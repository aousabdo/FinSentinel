"""
Text data collection module for FinSentinel.

This module provides utilities for collecting financial text data
from various sources like Reddit, Twitter/X, and news APIs.
"""

import pandas as pd
import numpy as np
import praw
import requests
import logging
import time
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# Set up logging
logger = logging.getLogger(__name__)

class TextDataFetcher:
    """Base class for fetching text data from various sources."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize the text data fetcher.
        
        Args:
            api_key: API key for the data source
            api_secret: API secret for the data source (if required)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def fetch_data(self, query: str, start_date: str, end_date: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch text data for a given query and time range.
        
        Args:
            query: Search query (e.g., ticker symbol or keyword)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of results to fetch
            
        Returns:
            DataFrame with text data
        """
        raise NotImplementedError("Subclasses must implement this method")


class RedditFetcher(TextDataFetcher):
    """Fetch text data from Reddit using PRAW."""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        Initialize Reddit fetcher with API credentials.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string for API requests
        """
        super().__init__(client_id, client_secret)
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        logger.info("Initialized Reddit connection")
    
    def fetch_data(self, query: str, start_date: str, end_date: str, limit: int = 100, 
                  subreddits: List[str] = None) -> pd.DataFrame:
        """
        Fetch posts from Reddit matching the query.
        
        Args:
            query: Search query (ticker symbol or keyword)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of posts to fetch
            subreddits: List of subreddits to search (defaults to financial subreddits)
            
        Returns:
            DataFrame with Reddit posts
        """
        if subreddits is None:
            subreddits = ["wallstreetbets", "investing", "stocks"]
        
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        
        all_posts = []
        
        try:
            for subreddit_name in subreddits:
                logger.info(f"Fetching posts from r/{subreddit_name} with query '{query}'")
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for posts
                search_results = subreddit.search(query, limit=limit)
                
                for post in search_results:
                    post_time = post.created_utc
                    
                    # Check if the post is within the date range
                    if start_timestamp <= post_time <= end_timestamp:
                        all_posts.append({
                            "id": post.id,
                            "title": post.title,
                            "text": post.selftext,
                            "score": post.score,
                            "num_comments": post.num_comments,
                            "created_utc": datetime.fromtimestamp(post.created_utc),
                            "subreddit": subreddit_name,
                            "url": post.url,
                            "author": str(post.author),
                        })
            
            if all_posts:
                logger.info(f"Fetched {len(all_posts)} Reddit posts matching '{query}'")
                return pd.DataFrame(all_posts)
            else:
                logger.warning(f"No Reddit posts found matching '{query}'")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching Reddit data: {str(e)}")
            return pd.DataFrame()


class NewsAPIFetcher(TextDataFetcher):
    """Fetch news articles from NewsAPI."""
    
    def __init__(self, api_key: str):
        """
        Initialize NewsAPI fetcher with API key.
        
        Args:
            api_key: NewsAPI key
        """
        super().__init__(api_key)
        self.base_url = "https://newsapi.org/v2/everything"
        logger.info("Initialized NewsAPI connection")
    
    def fetch_data(self, query: str, start_date: str, end_date: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch news articles from NewsAPI matching the query.
        
        Args:
            query: Search query (ticker symbol or keyword)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of articles to fetch
            
        Returns:
            DataFrame with news articles
        """
        try:
            logger.info(f"Fetching news articles for '{query}' from {start_date} to {end_date}")
            
            params = {
                "q": query,
                "from": start_date,
                "to": end_date,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(limit, 100),  # API limit is 100 per request
                "apiKey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            data = response.json()
            
            if data.get("status") != "ok":
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return pd.DataFrame()
            
            articles = data.get("articles", [])
            
            if not articles:
                logger.warning(f"No news articles found for '{query}'")
                return pd.DataFrame()
            
            # Process articles
            processed_articles = []
            for article in articles:
                processed_articles.append({
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "content": article.get("content"),
                    "source": article.get("source", {}).get("name"),
                    "author": article.get("author"),
                    "published_at": article.get("publishedAt"),
                    "url": article.get("url")
                })
            
            logger.info(f"Fetched {len(processed_articles)} news articles for '{query}'")
            return pd.DataFrame(processed_articles)
            
        except Exception as e:
            logger.error(f"Error fetching news data: {str(e)}")
            return pd.DataFrame()


def get_text_data(source: str = "reddit", 
                 query: str = "AAPL", 
                 start_date: str = None,
                 end_date: str = None,
                 limit: int = 100,
                 api_key: str = None,
                 api_secret: str = None,
                 **kwargs) -> pd.DataFrame:
    """
    Convenience function to fetch text data from the specified source.
    
    Args:
        source: Data source ("reddit", "twitter", "news_api", "sec_edgar")
        query: Search query (ticker symbol or keyword)
        start_date: Start date (defaults to 1 month ago)
        end_date: End date (defaults to today)
        limit: Maximum number of results to fetch
        api_key: API key for the data source
        api_secret: API secret for the data source (if required)
        **kwargs: Additional source-specific parameters
        
    Returns:
        DataFrame with text data
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create appropriate fetcher based on source
    if source.lower() == "reddit":
        if api_key is None or api_secret is None:
            raise ValueError("API key and secret are required for Reddit")
        
        fetcher = RedditFetcher(
            client_id=api_key,
            client_secret=api_secret,
            user_agent=kwargs.get("user_agent", f"FinSentinel/0.1.0 (by /u/{kwargs.get('username', 'YourUsername')})")
        )
        return fetcher.fetch_data(
            query=query,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            subreddits=kwargs.get("subreddits")
        )
        
    elif source.lower() == "news_api":
        if api_key is None:
            raise ValueError("API key is required for NewsAPI")
        
        fetcher = NewsAPIFetcher(api_key=api_key)
        return fetcher.fetch_data(
            query=query,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
    else:
        raise ValueError(f"Unsupported data source: {source}")


def clean_text(text: str) -> str:
    """
    Clean text data by removing URLs, special characters, etc.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text
