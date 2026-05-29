from typing import Optional

from pydantic import BaseModel

from prompts.daily_sentiment.config import DailyStockSentimentResult
from prompts.retrieve_articles.config import ArticlesResult


class OrchestratorResult(BaseModel):
    articles_result: ArticlesResult
    daily_sentiment_result: Optional[DailyStockSentimentResult]