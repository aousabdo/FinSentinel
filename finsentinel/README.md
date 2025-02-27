# FinSentinel: Financial Market Sentiment Analysis & Trading Strategy Project

## Project Overview
This project combines the power of Large Language Models (LLMs) with financial market analysis to create a sentiment-based trading strategy. You'll build a system that collects financial market data and related discussions, analyzes sentiment using LLMs, and develops trading signals based on this analysis.

## Learning Objectives
- Apply LLMs to real-world financial analysis problems
- Understand the relationship between market sentiment and price movements
- Develop and test algorithmic trading strategies
- Build data processing pipelines for financial information
- Create meaningful visualizations of financial insights

## Data Resources

### Market Data Options:
1. **Yahoo Finance API** (via [yfinance package](https://github.com/ranaroussi/yfinance))
   - Offers historical price data for stocks, ETFs, indices
   - Simple Python interface with no authentication required
   - Includes OHLC prices, volume, dividends, and splits
   - [Documentation](https://pypi.org/project/yfinance/)

2. **Alpha Vantage**
   - [Official Website](https://www.alphavantage.co/)
   - Free API tier with 5 calls per minute, 500 per day
   - Comprehensive financial data including stocks, forex, crypto
   - Requires simple API key registration
   - [Documentation](https://www.alphavantage.co/documentation/)

3. **Financial Modeling Prep**
   - [Official Website](https://financialmodelingprep.com/)
   - Free tier includes historical data, company financials
   - Stock prices, dividends, and market caps
   - Basic registration for API access
   - [Documentation](https://site.financialmodelingprep.com/developer/docs)

4. **Quandl**
   - [Official Website](https://www.quandl.com/)
   - Free tier with limited daily calls
   - Extensive financial and economic datasets
   - Registration required
   - [Documentation](https://docs.quandl.com/)
   - [Python Package](https://github.com/quandl/quandl-python)

### Text Data Options:
1. **Reddit API** (via [PRAW library](https://praw.readthedocs.io/))
   - Access to [r/wallstreetbets](https://www.reddit.com/r/wallstreetbets/), [r/investing](https://www.reddit.com/r/investing/), [r/stocks](https://www.reddit.com/r/stocks/)
   - Rich source of retail investor sentiment
   - Requires simple [Reddit developer account](https://www.reddit.com/prefs/apps)
   - [PRAW Documentation](https://praw.readthedocs.io/en/stable/)

2. **Twitter/X API**
   - [Developer Platform](https://developer.twitter.com/en)
   - Limited free access
   - Financial discussions with cashtags ($AAPL, etc.)
   - Developer account required
   - [Documentation](https://developer.twitter.com/en/docs)

3. **NewsAPI**
   - [Official Website](https://newsapi.org/)
   - Free tier with 100 requests/day
   - Financial news headlines from multiple sources
   - Simple registration
   - [Documentation](https://newsapi.org/docs)

4. **SEC Edgar Database**
   - [Official SEC Edgar Website](https://www.sec.gov/edgar.shtml)
   - Company filings (10-K, 10-Q)
   - Python packages available for access ([sec-edgar-downloader](https://github.com/jadchaar/sec-edgar-downloader))
   - No authentication required
   - [SEC API Documentation](https://www.sec.gov/edgar/sec-api-documentation)

### LLM Access Options:
1. **OpenAI API**
   - [Official Website](https://openai.com/api/)
   - Free credits for new users
   - GPT-4o models suitable for analysis
   - [Documentation](https://platform.openai.com/docs/)

2. **Anthropic Claude API**
   - [Official Website](https://www.anthropic.com/claude)
   - Academic access programs available
   - Strong reasoning capabilities for analysis
   - [Documentation](https://docs.anthropic.com/claude/docs)

3. **Ollama**
   - [Official Website](https://ollama.ai/)
   - Completely free, locally hosted open-source LLMs
   - No usage limits or API costs
   - Requires more computing resources
   - [Documentation](https://github.com/ollama/ollama)

## Implementation Plan

### Phase 1: Data Collection & Exploration (2-3 weeks)
- Set up your Python environment and project structure
- Select 3-5 stocks to focus on (preferably with active social discussion)
- Implement data collection from your chosen market data source
- Gather relevant social media posts/news using selected text source
- Create basic visualizations of price movements
- Explore correlations between posting volume and market movements

### Phase 2: LLM-Based Sentiment Analysis (2-3 weeks)
- Design prompts for your chosen LLM to analyze financial texts
- Create a scoring framework (e.g., -3 to +3 scale for sentiment)
- Process collected texts through the LLM to generate sentiment scores
- Analyze sentiment trends over time
- Visualize sentiment against price movements
- Document interesting patterns or correlations

### Phase 3: Trading Signal Development (2-3 weeks)
- Develop a methodology to convert sentiment scores into trading signals
- Create a simple backtesting framework
- Implement your sentiment-based trading strategy
- Test strategy performance on historical data
- Calculate key metrics (returns, Sharpe ratio, max drawdown)
- Compare performance to buy-and-hold

### Phase 4: Enhancement & Optimization (2-3 weeks)
- Add traditional technical indicators to complement sentiment signals
- Implement a simple machine learning model using sentiment as a feature
- Optimize strategy parameters (entry/exit thresholds, holding periods)
- Conduct sensitivity analysis on different parameters
- Test on out-of-sample data periods
- Document performance improvements

### Phase 5: Final Analysis & Presentation (1-2 weeks)
- Create comprehensive visualizations of your strategy
- Prepare a final report documenting methodology and findings
- Build a simple dashboard for ongoing sentiment monitoring
- Document limitations and potential improvements
- Prepare a presentation of your project

## Deliverables
1. Python codebase with documented modules
2. Data collection and processing pipeline
3. LLM sentiment analysis implementation
4. Trading strategy backtest results
5. Final report with visualizations
6. Project presentation

## Extension Ideas
- Compare sentiment analysis across different LLMs
- Expand to sector-wide analysis rather than individual stocks
- Incorporate earnings call transcripts for deeper sentiment analysis
- Develop a real-time monitoring system for ongoing sentiment tracking
- Explore different trading timeframes (daily vs. weekly signals)

## Implementation Tips
- Start small with manageable datasets before scaling up
- Save LLM responses to avoid repeated API calls (and costs)
- Use consistent prompting templates for reliable sentiment scoring
- Document all assumptions and methodology decisions
- Consider market hours and announcement timing in your analysis
- Be mindful of look-ahead bias in backtesting

## Helpful Libraries and Tools

### Data Analysis & Processing
- [pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [numpy](https://numpy.org/) - Numerical computing
- [scipy](https://scipy.org/) - Scientific computing

### Visualization
- [matplotlib](https://matplotlib.org/) - Basic plotting library
- [seaborn](https://seaborn.pydata.org/) - Statistical data visualization
- [plotly](https://plotly.com/python/) - Interactive visualizations
- [dash](https://dash.plotly.com/) - Web applications for visualization

### Financial Analysis
- [pandas-ta](https://github.com/twopirllc/pandas-ta) - Technical analysis indicators
- [pyfolio](https://github.com/quantopian/pyfolio) - Portfolio and risk analytics
- [backtrader](https://www.backtrader.com/) - Trading strategy backtesting

### Machine Learning
- [scikit-learn](https://scikit-learn.org/) - Machine learning algorithms
- [tensorflow](https://www.tensorflow.org/) or [pytorch](https://pytorch.org/) - Deep learning

### API Integration
- [requests](https://requests.readthedocs.io/) - HTTP requests
- [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/) - Web scraping
- [fastapi](https://fastapi.tiangolo.com/) - API development

## System Workflow Diagrams

To help understand the FinSentinel architecture and workflows, we've created a set of visual diagrams. These diagrams illustrate the key components and processes of the system.

### Overall Workflow
![Overall Workflow](docs/images/overall_workflow.png)

The FinSentinel workflow consists of five main stages:
1. **Data Collection** - Gathering market and text data from various sources
2. **Data Processing** - Cleaning and preprocessing the collected data
3. **Sentiment Analysis** - Using LLMs to analyze sentiment in financial texts
4. **Trading Strategy** - Generating trading signals based on sentiment and market data
5. **Analysis & Visualization** - Backtesting strategies and visualizing the results

### System Architecture
![System Architecture](docs/images/system_architecture.png)

The system consists of the following components:
- **External Data Sources** - APIs for market data, text data, and LLM providers
- **Core Modules** - Data, Sentiment, Strategy, and Visualization components
- **Application Layer** - Examples, notebooks, and user interfaces

### Data Pipeline
![Data Pipeline](docs/images/data_pipeline.png)

### Sentiment Analysis Process
![Sentiment Analysis](docs/images/sentiment_analysis.png)

### Backtesting Workflow
![Backtest Workflow](docs/images/backtest_workflow.png)

### User Workflow
![User Workflow](docs/images/user_workflow.png)

For more detailed explanations of these diagrams and workflows, see [Workflow Guide](docs/workflow_guide.md).
