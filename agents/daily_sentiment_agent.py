from dotenv import load_dotenv
from openai import OpenAI

from prompts.daily_sentiment.config import (
    SYSTEM_MESSAGE,
    USER_MESSAGE_TEMPLATE,
    DailyStockSentimentResult,
)
from prompts.retrieve_articles.config import ArticlesResult

load_dotenv()
client = OpenAI()

MODEL = "gpt-5.4-mini"


def classify_daily_stock_sentiment(
    articles_result: ArticlesResult,
    sector: str,
) -> DailyStockSentimentResult:
    articles_text = "\n\n".join(
        f"Title: {article.title}\n"
        f"Published date: {article.published_date}\n"
        f"Source: {article.source}\n"
        f"Category: {article.category}\n"
        f"Summary: {article.summary}\n"
        f"Relevance reason: {article.relevance_reason}\n"
        f"URL: {article.url}"
        for article in articles_result.articles
    )

    user_message = USER_MESSAGE_TEMPLATE.format(
        date=articles_result.date,
        sector=sector,
        articles_text=articles_text,
    )

    response = client.responses.parse(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        text_format=DailyStockSentimentResult,
    )

    return response.output_parsed