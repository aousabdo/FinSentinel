"""
LLM-based sentiment analysis module for FinSentinel.

This module provides utilities for analyzing sentiment in financial texts
using various Large Language Models (LLMs).
"""

import pandas as pd
import numpy as np
import logging
import time
import json
import os
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

# Optional LLM integrations
try:
    import openai
except ImportError:
    logger.warning("OpenAI package not installed. OpenAI models will not be available.")

try:
    import anthropic
except ImportError:
    logger.warning("Anthropic package not installed. Claude models will not be available.")


class LLMSentimentAnalyzer:
    """Base class for LLM-based sentiment analysis."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "default"):
        """
        Initialize the LLM sentiment analyzer.
        
        Args:
            api_key: API key for the LLM service
            model: Model identifier to use
        """
        self.api_key = api_key
        self.model = model
        logger.info(f"Initialized {self.__class__.__name__} with model {model}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a text using the LLM.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def batch_analyze(self, texts: List[str], batch_size: int = 10, 
                     delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process in each batch
            delay: Delay between batches (in seconds)
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            batch_results = []
            for text in batch:
                try:
                    result = self.analyze_sentiment(text)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing text: {str(e)}")
                    batch_results.append({"error": str(e)})
            
            results.extend(batch_results)
            
            if i + batch_size < len(texts):
                logger.info(f"Sleeping for {delay} seconds before next batch")
                time.sleep(delay)
        
        return results


class OpenAIAnalyzer(LLMSentimentAnalyzer):
    """Sentiment analysis using OpenAI models."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize OpenAI analyzer.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (default: gpt-4o)
        """
        super().__init__(api_key, model)
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI client with model {model}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using OpenAI models.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Prepare the prompt
            prompt = self._create_prompt(text)
            
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more consistent results
                max_tokens=500
            )
            
            # Parse the response
            result = self._parse_response(response.choices[0].message.content)
            result["text"] = text[:200] + "..." if len(text) > 200 else text
            result["timestamp"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in OpenAI sentiment analysis: {str(e)}")
            return {"error": str(e), "text": text[:100], "timestamp": datetime.now().isoformat()}
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for sentiment analysis."""
        return """
        You are a financial sentiment analyzer. Your task is to analyze the sentiment of financial texts.
        For each text, provide:
        1. A sentiment score from -3 (extremely negative) to +3 (extremely positive), with 0 being neutral
        2. A confidence score from 0 to 1
        3. Identified sentiment drivers (key factors influencing the sentiment)
        4. Key financial entities mentioned (stocks, companies, markets, etc.)
        
        Respond in a structured JSON format with the following fields:
        {
            "sentiment_score": float,
            "confidence": float,
            "sentiment_drivers": [list of strings],
            "entities": [list of strings],
            "summary": string
        }
        """
    
    def _create_prompt(self, text: str) -> str:
        """Create a prompt for sentiment analysis."""
        return f"""
        Please analyze the sentiment of the following financial text:

        "{text}"

        Provide your analysis in the structured JSON format described in the system prompt.
        Focus only on financial sentiment, not general sentiment.
        """
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the response from OpenAI API."""
        try:
            # Extract JSON from the response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > 0:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            
            # Fallback to a basic structure if JSON can't be extracted
            return {
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "sentiment_drivers": [],
                "entities": [],
                "summary": "Failed to parse structured response",
                "raw_response": response_text
            }
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse response as JSON: {response_text}")
            return {
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "sentiment_drivers": [],
                "entities": [],
                "summary": "Failed to parse response as JSON",
                "raw_response": response_text
            }


class AnthropicAnalyzer(LLMSentimentAnalyzer):
    """Sentiment analysis using Anthropic Claude models."""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        """
        Initialize Anthropic analyzer.
        
        Args:
            api_key: Anthropic API key
            model: Anthropic model to use
        """
        super().__init__(api_key, model)
        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Initialized Anthropic client with model {model}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using Anthropic Claude models.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Prepare the prompt
            system_prompt = self._get_system_prompt()
            user_prompt = self._create_prompt(text)
            
            # Call the Anthropic API
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse the response
            result = self._parse_response(response.content[0].text)
            result["text"] = text[:200] + "..." if len(text) > 200 else text
            result["timestamp"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Anthropic sentiment analysis: {str(e)}")
            return {"error": str(e), "text": text[:100], "timestamp": datetime.now().isoformat()}
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for sentiment analysis."""
        return """
        You are a financial sentiment analyzer. Your task is to analyze the sentiment of financial texts.
        For each text, provide:
        1. A sentiment score from -3 (extremely negative) to +3 (extremely positive), with 0 being neutral
        2. A confidence score from 0 to 1
        3. Identified sentiment drivers (key factors influencing the sentiment)
        4. Key financial entities mentioned (stocks, companies, markets, etc.)
        
        Respond in a structured JSON format with the following fields:
        {
            "sentiment_score": float,
            "confidence": float,
            "sentiment_drivers": [list of strings],
            "entities": [list of strings],
            "summary": string
        }
        """
    
    def _create_prompt(self, text: str) -> str:
        """Create a prompt for sentiment analysis."""
        return f"""
        Please analyze the sentiment of the following financial text:

        "{text}"

        Provide your analysis in the structured JSON format described in the system instructions.
        Focus only on financial sentiment, not general sentiment.
        """
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the response from Anthropic API."""
        return self._extract_json(response_text)
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from the response text."""
        try:
            # Find JSON in the response
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > 0:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
            
            # Fallback if JSON can't be extracted
            return {
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "sentiment_drivers": [],
                "entities": [],
                "summary": "Failed to parse structured response",
                "raw_response": text
            }
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse response as JSON: {text}")
            return {
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "sentiment_drivers": [],
                "entities": [],
                "summary": "Failed to parse response as JSON",
                "raw_response": text
            }


def analyze_sentiment(texts: Union[str, List[str]], 
                     provider: str = "openai",
                     api_key: Optional[str] = None,
                     model: Optional[str] = None,
                     **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Convenience function to analyze sentiment using the specified LLM provider.
    
    Args:
        texts: Text or list of texts to analyze
        provider: LLM provider ("openai", "anthropic", "ollama")
        api_key: API key for the LLM provider
        model: Model identifier
        **kwargs: Additional provider-specific parameters
        
    Returns:
        Dictionary or list of dictionaries with sentiment analysis results
    """
    # Check if API key is in environment variables
    if api_key is None:
        env_var_name = f"{provider.upper()}_API_KEY"
        api_key = os.environ.get(env_var_name)
        if api_key is None:
            raise ValueError(f"API key not provided and {env_var_name} not found in environment variables")
    
    # Choose default models if not specified
    if model is None:
        if provider.lower() == "openai":
            model = "gpt-4o"
        elif provider.lower() == "anthropic":
            model = "claude-3-opus-20240229"
        # Add other providers' default models as needed
    
    # Create the appropriate analyzer
    if provider.lower() == "openai":
        analyzer = OpenAIAnalyzer(api_key=api_key, model=model)
    elif provider.lower() == "anthropic":
        analyzer = AnthropicAnalyzer(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    
    # Handle single text or list of texts
    if isinstance(texts, str):
        return analyzer.analyze_sentiment(texts)
    else:
        batch_size = kwargs.get("batch_size", 10)
        delay = kwargs.get("delay", 1.0)
        return analyzer.batch_analyze(texts, batch_size=batch_size, delay=delay)


def calculate_aggregate_sentiment(sentiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate aggregate sentiment from a list of sentiment analysis results.
    
    Args:
        sentiment_results: List of sentiment analysis results
        
    Returns:
        Dictionary with aggregate sentiment metrics
    """
    # Filter out results with errors
    valid_results = [r for r in sentiment_results if "error" not in r]
    
    if not valid_results:
        return {
            "mean_sentiment": 0.0,
            "median_sentiment": 0.0,
            "weighted_sentiment": 0.0,
            "sentiment_std": 0.0,
            "sample_size": 0,
            "sentiment_distribution": {}
        }
    
    # Extract sentiment scores and confidences
    scores = [r.get("sentiment_score", 0.0) for r in valid_results]
    confidences = [r.get("confidence", 0.5) for r in valid_results]
    
    # Calculate basic statistics
    mean_sentiment = np.mean(scores)
    median_sentiment = np.median(scores)
    
    # Calculate confidence-weighted sentiment
    weighted_sentiment = np.average(scores, weights=confidences) if confidences else mean_sentiment
    
    # Calculate standard deviation
    sentiment_std = np.std(scores)
    
    # Create sentiment distribution
    sentiment_bins = {
        "very_negative": 0,
        "negative": 0,
        "slightly_negative": 0,
        "neutral": 0,
        "slightly_positive": 0,
        "positive": 0,
        "very_positive": 0
    }
    
    for score in scores:
        if score <= -2.0:
            sentiment_bins["very_negative"] += 1
        elif score <= -1.0:
            sentiment_bins["negative"] += 1
        elif score < 0:
            sentiment_bins["slightly_negative"] += 1
        elif score == 0:
            sentiment_bins["neutral"] += 1
        elif score < 1.0:
            sentiment_bins["slightly_positive"] += 1
        elif score < 2.0:
            sentiment_bins["positive"] += 1
        else:
            sentiment_bins["very_positive"] += 1
    
    # Convert counts to percentages
    total = len(scores)
    sentiment_distribution = {k: v / total for k, v in sentiment_bins.items()}
    
    # Aggregate common sentiment drivers
    all_drivers = []
    for result in valid_results:
        drivers = result.get("sentiment_drivers", [])
        all_drivers.extend(drivers)
    
    # Count driver frequencies
    driver_counts = {}
    for driver in all_drivers:
        driver_counts[driver] = driver_counts.get(driver, 0) + 1
    
    # Get top 5 drivers
    top_drivers = sorted(driver_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "mean_sentiment": mean_sentiment,
        "median_sentiment": median_sentiment,
        "weighted_sentiment": weighted_sentiment,
        "sentiment_std": sentiment_std,
        "sample_size": len(valid_results),
        "sentiment_distribution": sentiment_distribution,
        "top_sentiment_drivers": dict(top_drivers)
    }
