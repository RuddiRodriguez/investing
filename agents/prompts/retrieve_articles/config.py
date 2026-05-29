"""Prompts and schemas for retrieve_articles_agent"""

from typing import List

from pydantic import BaseModel, Field


# Prompts
SYSTEM_MESSAGE = (
    "You are a financial news retrieval agent. "
    "Your job is to find real news articles that could affect stock prices. "
    "Use web search. "
    "Return only factual news articles from , not opinion columns, blog posts, transcripts, "
    "live pages, tag pages, search pages, market data pages, or generic explainers. "
    "Every article must be directly relevant to publicly traded stocks or the sector's "
    "market conditions. "
    "Return structured output only."
)

USER_MESSAGE_TEMPLATE = (
    "Find up to {max_articles} factual news articles published on {date_to_analyze} "
    "about the {sector} sector and its impact on stocks or financial markets.\n\n"
    "Search using these concepts:\n"
    "- {sector} stocks\n"
    "- {sector} sector market news\n"
    "- earnings, revenue, guidance, oil prices, natural gas prices, electricity demand, "
    "renewables, utilities, regulation, OPEC, supply, demand, mergers, acquisitions, "
    "analyst outlook, investment, capital expenditure\n\n"
    "Preferred sources: Reuters, Bloomberg, CNBC, Financial Times, Wall Street Journal, "
    "MarketWatch, AP News, Yahoo Finance, Investing.com, Nasdaq, official company investor relations, "
    "or official government/regulator sources.\n\n"
    "Reject articles if they are only weakly related, only opinion, or only mention the sector indirectly. "
    "Do not rank the articles. "
    "For each article return title, published date, source, category, summary, URL, and relevance reason."
)

TOOLS = [
    {
        "type": "web_search",
        "search_context_size": "high",
    }
]


# Schemas
class Article(BaseModel):
    title: str = Field(description="Article headline")
    published_date: str = Field(description="Published date in YYYY-MM-DD format")
    source: str = Field(description="Publisher or news source")
    category: str = Field(description="Article category")
    summary: str = Field(description="Short factual summary")
    url: str = Field(description="Article URL")
    relevance_reason: str = Field(description="Why this article is relevant to the sector and stocks")


class ArticlesResult(BaseModel):
    date: str = Field(description="Date searched in YYYY-MM-DD format")
    sector: str = Field(description="Sector searched")
    articles: List[Article] = Field(description="Relevant articles for the date")
    notes: str = Field(description="Limitations, missing information, or rejected result types")
