# API Integration Guide for FinSentinel

This guide provides instructions for setting up and using the various APIs required for the FinSentinel project.

## Table of Contents

1. [Market Data APIs](#market-data-apis)
   - [Yahoo Finance](#yahoo-finance)
   - [Alpha Vantage](#alpha-vantage)
   - [Financial Modeling Prep](#financial-modeling-prep)
   - [Quandl](#quandl)

2. [Text Data APIs](#text-data-apis)
   - [Reddit API](#reddit-api)
   - [Twitter/X API](#twitterx-api)
   - [NewsAPI](#newsapi)
   - [SEC Edgar Database](#sec-edgar-database)

3. [LLM APIs](#llm-apis)
   - [OpenAI API](#openai-api)
   - [Anthropic Claude API](#anthropic-claude-api)
   - [Ollama](#ollama)

4. [Environment Setup](#environment-setup)
   - [API Key Management](#api-key-management)
   - [Using .env Files](#using-env-files)

## Market Data APIs

### Yahoo Finance

Yahoo Finance is accessible via the [yfinance](https://github.com/ranaroussi/yfinance) Python package and does not require an API key.

**Installation:**
```bash
pip install yfinance
```

**Example Usage:**
```python
import yfinance as yf

# Get historical market data
data = yf.download("AAPL", start="2022-01-01", end="2022-12-31")
print(data.head())

# Get stock info
stock_info = yf.Ticker("AAPL").info
print(stock_info["longBusinessSummary"])
```

### Alpha Vantage

Alpha Vantage provides stock market data, forex, and cryptocurrency data via a simple REST API.

**Registration:**
1. Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Sign up for a free API key

**Installation:**
```bash
pip install alpha_vantage
```

**Example Usage:**
```python
from alpha_vantage.timeseries import TimeSeries

# Initialize TimeSeries with your API key
ts = TimeSeries(key='YOUR_API_KEY')

# Get daily adjusted data
data, meta_data = ts.get_daily_adjusted(symbol='AAPL', outputsize='full')
print(data.head())
```

### Financial Modeling Prep

Financial Modeling Prep provides financial data including stock prices, financial statements, and company profiles.

**Registration:**
1. Visit [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs/)
2. Sign up for an API key

**Installation:**
```bash
pip install requests  # FMP doesn't have an official Python package
```

**Example Usage:**
```python
import requests
import pandas as pd

# Define the API base URL and your API key
base_url = "https://financialmodelingprep.com/api/v3"
api_key = "YOUR_API_KEY"

# Get historical price data
endpoint = f"{base_url}/historical-price-full/AAPL?apikey={api_key}"
response = requests.get(endpoint)
data = response.json()

# Convert to DataFrame
historical_data = pd.DataFrame(data["historical"])
print(historical_data.head())
```

### Quandl

Quandl provides financial, economic, and alternative datasets.

**Registration:**
1. Visit [Quandl](https://www.quandl.com/)
2. Sign up for an API key

**Installation:**
```bash
pip install quandl
```

**Example Usage:**
```python
import quandl
import pandas as pd

# Set your API key
quandl.ApiConfig.api_key = 'YOUR_API_KEY'

# Get historical data
data = quandl.get('EOD/AAPL', start_date='2022-01-01', end_date='2022-12-31')
print(data.head())
```

## Text Data APIs

### Reddit API

The Reddit API allows you to access posts, comments, and other data from Reddit.

**Registration:**
1. Go to [Reddit Developer Portal](https://www.reddit.com/prefs/apps)
2. Click "create app" or "create another app"
3. Fill in the required information:
   - Name: Your app name
   - App type: Script
   - Description: Brief description of your app
   - About URL: Optional
   - Redirect URI: Use `http://localhost:8000` for a script app
4. Click "create app"
5. Note your client ID (below the app name) and client secret

**Installation:**
```bash
pip install praw
```

**Example Usage:**
```python
import praw
import pandas as pd

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="YOUR_APP_NAME/1.0 (by /u/YOUR_USERNAME)"
)

# Get hot posts from a subreddit
hot_posts = reddit.subreddit("wallstreetbets").hot(limit=10)

# Convert to DataFrame
posts_data = []
for post in hot_posts:
    posts_data.append({
        "title": post.title,
        "score": post.score,
        "id": post.id,
        "url": post.url,
        "created_utc": pd.to_datetime(post.created_utc, unit='s'),
        "num_comments": post.num_comments
    })

posts_df = pd.DataFrame(posts_data)
print(posts_df)
```

### Twitter/X API

The Twitter (now X) API allows you to access tweets and other Twitter data.

**Registration:**
1. Go to the [Twitter Developer Platform](https://developer.twitter.com/en)
2. Sign up for a developer account
3. Create a project and app
4. Generate API keys and tokens

**Installation:**
```bash
pip install tweepy
```

**Example Usage:**
```python
import tweepy
import pandas as pd

# Set up API keys and tokens
consumer_key = "YOUR_CONSUMER_KEY"
consumer_secret = "YOUR_CONSUMER_SECRET"
access_token = "YOUR_ACCESS_TOKEN"
access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"

# Authenticate with Twitter API
auth = tweepy.OAuth1UserHandler(
    consumer_key, consumer_secret, access_token, access_token_secret
)
api = tweepy.API(auth)

# Search for tweets containing a cashtag
tweets = api.search_tweets(q="$AAPL", count=10)

# Convert to DataFrame
tweets_data = []
for tweet in tweets:
    tweets_data.append({
        "text": tweet.text,
        "created_at": tweet.created_at,
        "user": tweet.user.screen_name,
        "retweet_count": tweet.retweet_count,
        "favorite_count": tweet.favorite_count
    })

tweets_df = pd.DataFrame(tweets_data)
print(tweets_df)
```

### NewsAPI

NewsAPI provides access to news articles from various sources.

**Registration:**
1. Visit [NewsAPI](https://newsapi.org/)
2. Sign up for an API key

**Installation:**
```bash
pip install newsapi-python
```

**Example Usage:**
```python
from newsapi import NewsApiClient
import pandas as pd

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key='YOUR_API_KEY')

# Get news articles about Apple
articles = newsapi.get_everything(
    q='Apple',
    language='en',
    sort_by='publishedAt',
    page=1
)

# Convert to DataFrame
articles_data = []
for article in articles['articles']:
    articles_data.append({
        "title": article['title'],
        "source": article['source']['name'],
        "published_at": article['publishedAt'],
        "url": article['url'],
        "description": article['description']
    })

articles_df = pd.DataFrame(articles_data)
print(articles_df)
```

### SEC Edgar Database

The SEC Edgar Database provides access to company filings.

**Installation:**
```bash
pip install sec-edgar-downloader
```

**Example Usage:**
```python
from sec_edgar_downloader import Downloader
import os

# Initialize downloader
downloader = Downloader("./edgar_data")

# Download 10-K filings for Apple
downloader.get("10-K", "AAPL", limit=5)
print(f"Files downloaded to {os.path.abspath('./edgar_data')}")
```

## LLM APIs

### OpenAI API

The OpenAI API provides access to models like GPT-4o.

**Registration:**
1. Visit [OpenAI API](https://platform.openai.com/)
2. Sign up and create an API key

**Installation:**
```bash
pip install openai
```

**Example Usage:**
```python
from openai import OpenAI
import json

# Initialize OpenAI client
client = OpenAI(api_key="YOUR_API_KEY")

# Define function to analyze sentiment
def analyze_sentiment(text):
    system_prompt = """
    You are a financial sentiment analyzer. Analyze the text and provide a sentiment score from -3 (very negative) to +3 (very positive).
    Respond in JSON format with the following fields:
    {
        "sentiment_score": float,
        "confidence": float,
        "sentiment_drivers": [string],
        "entities": [string],
        "summary": string
    }
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.1
    )
    
    # Parse JSON response
    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except json.JSONDecodeError:
        # If response is not valid JSON, return raw response
        return {"raw_response": response.choices[0].message.content}

# Example usage
text = "Apple's quarterly results exceeded analyst expectations, with strong iPhone sales and growing services revenue."
result = analyze_sentiment(text)
print(result)
```

### Anthropic Claude API

The Anthropic Claude API provides access to the Claude family of models.

**Registration:**
1. Visit [Anthropic Claude](https://www.anthropic.com/claude)
2. Sign up and create an API key

**Installation:**
```bash
pip install anthropic
```

**Example Usage:**
```python
import anthropic
import json

# Initialize Anthropic client
client = anthropic.Anthropic(api_key="YOUR_API_KEY")

# Define function to analyze sentiment
def analyze_sentiment(text):
    system_prompt = """
    You are a financial sentiment analyzer. Analyze the text and provide a sentiment score from -3 (very negative) to +3 (very positive).
    Respond in JSON format with the following fields:
    {
        "sentiment_score": float,
        "confidence": float,
        "sentiment_drivers": [string],
        "entities": [string],
        "summary": string
    }
    """
    
    response = client.messages.create(
        model="claude-3-opus-20240229",
        system=system_prompt,
        messages=[
            {"role": "user", "content": text}
        ],
        temperature=0.1,
        max_tokens=1000
    )
    
    # Parse JSON response
    try:
        # Extract JSON from the response
        content = response.content[0].text
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        json_str = content[json_start:json_end]
        
        result = json.loads(json_str)
        return result
    except (json.JSONDecodeError, ValueError):
        # If response is not valid JSON, return raw response
        return {"raw_response": response.content[0].text}

# Example usage
text = "Tesla reported manufacturing delays at their new factory, which could impact quarterly delivery targets."
result = analyze_sentiment(text)
print(result)
```

### Ollama

Ollama allows you to run open-source LLMs locally.

**Installation:**
1. Visit [Ollama](https://ollama.ai/)
2. Follow the installation instructions for your operating system

**Example Usage:**
```python
import json
import requests

# Define function to analyze sentiment using Ollama
def analyze_sentiment(text, model="llama3"):
    system_prompt = """
    You are a financial sentiment analyzer. Analyze the text and provide a sentiment score from -3 (very negative) to +3 (very positive).
    Respond in JSON format with the following fields:
    {
        "sentiment_score": float,
        "confidence": float,
        "sentiment_drivers": [string],
        "entities": [string],
        "summary": string
    }
    """
    
    # Build the prompt
    full_prompt = f"{system_prompt}\n\nText to analyze: {text}"
    
    # Call Ollama API
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": full_prompt,
            "temperature": 0.1,
            "stream": False
        }
    )
    
    # Parse response
    response_json = response.json()
    response_text = response_json.get("response", "")
    
    # Extract JSON from the response
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        json_str = response_text[json_start:json_end]
        
        result = json.loads(json_str)
        return result
    except (json.JSONDecodeError, ValueError):
        # If response is not valid JSON, return raw response
        return {"raw_response": response_text}

# Example usage
text = "Microsoft's cloud services revenue grew by 25% year-over-year, though slightly below analyst expectations."
result = analyze_sentiment(text)
print(result)
```

## Environment Setup

### API Key Management

It's important to keep your API keys secure and not hardcode them in your application code. Here are some best practices:

1. Store API keys in environment variables
2. Use a `.env` file for local development
3. Use secure key management services for production (e.g., AWS Secrets Manager, HashiCorp Vault)
4. Never commit API keys to version control

### Using .env Files

A `.env` file is a simple text file that contains environment variables in the format `KEY=value`.

**Create a `.env` file:**
```
# Market Data API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
QUANDL_API_KEY=your_quandl_key
FMP_API_KEY=your_fmp_key

# Text Data API Keys
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_reddit_user_agent
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
NEWS_API_KEY=your_news_api_key

# LLM API Keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

**Loading `.env` file in Python:**
```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

print(f"OpenAI API Key: {openai_api_key[:5]}..." if openai_api_key else "OpenAI API Key not found")
print(f"Alpha Vantage API Key: {alpha_vantage_api_key[:5]}..." if alpha_vantage_api_key else "Alpha Vantage API Key not found")
```

**Installation for `.env` support:**
```bash
pip install python-dotenv
```

**Important:** Add `.env` to your `.gitignore` file to prevent it from being committed to version control.
