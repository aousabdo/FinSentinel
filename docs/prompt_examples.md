# LLM Prompt Examples for Financial Sentiment Analysis

This document provides example prompts for using LLMs to analyze financial sentiment in text data.

## Basic Sentiment Analysis Prompt

```
You are an expert financial analyst with deep knowledge of stock markets, trading, and investment sentiment.

Please analyze the following text from a financial discussion and provide:
1. A sentiment score from -3 (extremely bearish) to +3 (extremely bullish), with 0 being neutral
2. A confidence score from 0.0 to 1.0 indicating your confidence in this assessment
3. Key sentiment drivers identified in the text
4. Specific financial entities mentioned (companies, stocks, sectors, etc.)

Text to analyze:
"{text}"

Format your response as a JSON object with the following fields:
{
  "sentiment_score": float,
  "confidence": float,
  "sentiment_drivers": ["driver1", "driver2", ...],
  "entities": ["entity1", "entity2", ...],
  "summary": "One sentence summary of the sentiment"
}
```

## Structured Sentiment Analysis Prompt

```
You are an expert financial analyst assessing market sentiment in text data. Analyze the following text 
from an investment forum post and provide a structured assessment.

Text to analyze:
"{text}"

Your assessment should include:

1. SENTIMENT SCORE: Assign a score from -3 to +3 where:
   * -3: Extremely bearish, strong conviction of significant downside
   * -2: Clearly bearish outlook
   * -1: Somewhat bearish or cautious
   *  0: Neutral or mixed sentiment
   * +1: Somewhat bullish or cautiously optimistic
   * +2: Clearly bullish outlook
   * +3: Extremely bullish, strong conviction of significant upside

2. CONFIDENCE: Rate your confidence in this sentiment score (0.0-1.0)
   Consider factors like:
   * Specificity of claims (specific metrics vs. general feelings)
   * Presence of factual statements vs. opinions
   * Author's apparent expertise
   * Consistency of sentiment throughout

3. KEY DRIVERS: List 2-5 key factors driving the sentiment

4. ENTITIES: Identify all financial entities mentioned (stocks, companies, sectors, indices, etc.)

5. SUMMARY: One-sentence synopsis of the overall sentiment

Respond in JSON format:
{
  "sentiment_score": float,
  "confidence": float,
  "sentiment_drivers": ["driver1", "driver2", ...],
  "entities": ["entity1", "entity2", ...],
  "summary": "string"
}
```

## Entity-Specific Sentiment Analysis

```
You are a financial sentiment analyzer focusing on specific entities mentioned in text. 
For the following text, analyze the sentiment for each mentioned company, stock, or financial entity.

Text to analyze:
"{text}"

For each detected entity:
1. Assign a sentiment score from -3 (extremely negative) to +3 (extremely positive)
2. Note key passages that informed your sentiment assessment
3. Assess the confidence of your sentiment rating (0.0-1.0)

Then provide an overall assessment of the text's financial sentiment.

Format your response as a JSON object:
{
  "entities": [
    {
      "name": "Entity name (e.g., AAPL, Tesla, cryptocurrency)",
      "sentiment_score": float,
      "key_passages": ["passage1", "passage2"],
      "confidence": float
    },
    ...
  ],
  "overall_sentiment": float,
  "confidence": float,
  "summary": "One sentence summary"
}
```

## Time-Sensitive Sentiment Analysis

```
You are an expert financial analyst assessing the time-horizon of sentiment in financial discussions.

Analyze the following text:
"{text}"

Provide a sentiment analysis across different time horizons:
1. SHORT-TERM (days to weeks)
2. MEDIUM-TERM (months to quarters)
3. LONG-TERM (1+ years)

For each time horizon, provide:
- Sentiment score (-3 to +3)
- Confidence (0.0-1.0)
- Key factors influencing this time-specific outlook

Then provide an overall assessment and summary.

Format your response as JSON:
{
  "short_term": {
    "sentiment_score": float,
    "confidence": float,
    "factors": ["factor1", "factor2", ...]
  },
  "medium_term": {
    "sentiment_score": float,
    "confidence": float,
    "factors": ["factor1", "factor2", ...]
  },
  "long_term": {
    "sentiment_score": float,
    "confidence": float,
    "factors": ["factor1", "factor2", ...]
  },
  "entities": ["entity1", "entity2", ...],
  "overall_sentiment": float,
  "summary": "One sentence summary"
}
```

## Comparative Sentiment Analysis

```
You are a financial analyst comparing sentiment between different entities mentioned in a text.

Text to analyze:
"{text}"

Identify the primary financial entities (companies, stocks, sectors, etc.) mentioned and compare the sentiment:

1. For each entity:
   - Sentiment score (-3 to +3)
   - Confidence in this assessment (0.0-1.0)
   - Key sentiment drivers

2. Provide a comparison of the entities:
   - Which has the most positive sentiment
   - Which has the most negative sentiment
   - Any notable contrasts in sentiment drivers

Format your response as JSON:
{
  "entities": [
    {
      "name": "Entity name",
      "sentiment_score": float,
      "confidence": float,
      "sentiment_drivers": ["driver1", "driver2", ...]
    },
    ...
  ],
  "comparison": {
    "most_positive": "Entity name",
    "most_negative": "Entity name",
    "notable_contrasts": ["contrast1", "contrast2", ...]
  },
  "summary": "One sentence summary of the comparative sentiment"
}
```

## Implementation Tips

When implementing these prompts:

1. **Variable Substitution**: Replace `{text}` with the actual text to be analyzed.
2. **Output Parsing**: Set up robust JSON parsing with error handling.
3. **Prompt Tuning**: You may need to adjust the prompt based on the specific LLM you're using.
4. **Context Length**: Be mindful of the token limits of your LLM when analyzing longer texts.
5. **Batching**: When analyzing multiple texts, consider batching to reduce API costs.
6. **Caching**: Implement a caching mechanism to avoid reanalyzing the same text multiple times.

## Example Use in Python

```python
import json
from finsentinel.sentiment.llm_analyzer import analyze_sentiment

# Example text
text = "Apple's recent earnings report shows impressive growth in services, which could offset any weakness in iPhone sales. I'm bullish on their long-term prospects given their strong ecosystem and customer loyalty."

# Load prompt template
with open("prompt_templates/basic_sentiment.txt", "r") as f:
    prompt_template = f.read()

# Format prompt with text
prompt = prompt_template.format(text=text)

# Call LLM with prompt
result = analyze_sentiment(text, provider="openai", model="gpt-4o")

# Parse result
print(f"Sentiment Score: {result['sentiment_score']}")
print(f"Confidence: {result['confidence']}")
print(f"Sentiment Drivers: {', '.join(result['sentiment_drivers'])}")
print(f"Entities: {', '.join(result['entities'])}")
print(f"Summary: {result['summary']}")
```
