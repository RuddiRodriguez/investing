from dotenv import load_dotenv
from openai import OpenAI

from prompts.retrieve_articles.config import (
    SYSTEM_MESSAGE,
    TOOLS,
    USER_MESSAGE_TEMPLATE,
    ArticlesResult,
)

load_dotenv()
client = OpenAI()

MODEL = "gpt-5.4-mini"


def retrieve_articles(
    date_to_analyze: str,
    sector: str,
    max_articles: int,
) -> ArticlesResult:
    user_message = USER_MESSAGE_TEMPLATE.format(
        max_articles=max_articles,
        date_to_analyze=date_to_analyze,
        sector=sector,
    )

    response = client.responses.parse(
        model=MODEL,
        tools=TOOLS,
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
        text_format=ArticlesResult,
    )

    return response.output_parsed