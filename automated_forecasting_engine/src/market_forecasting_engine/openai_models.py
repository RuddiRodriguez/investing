from __future__ import annotations

from enum import Enum


class ModelName(str, Enum):
    GPT_3_5_LATEST = "gpt-3.5-turbo"
    GPT_3_5_0613 = "gpt-3.5-turbo-0613"
    GPT_3_5_1106 = "gpt-3.5-turbo-1106"
    GPT_3_5_0125 = "gpt-3.5-turbo-0125"
    GPT_3_5_LATEST_16K = "gpt-3.5-turbo-16k"
    GPT_3_5_16K_0613 = "gpt-3.5-turbo-16k-0613"
    GPT_4_0125_PREVIEW = "gpt-4-0125-preview"
    GPT_4_2024_04_09 = "gpt-4-turbo-2024-04-09"
    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"
    GPT_4O_2024_08_06 = "gpt-4o-2024-08-06"
    GPT_4O_2024_11_20 = "gpt-4o-2024-11-20"
    GPT_4O_MINI_2024_07_18 = "gpt-4o-mini-2024-07-18"
    GPT_4_1_NANO_2025_04_14 = "gpt-4.1-nano-2025-04-14"
    GPT_4_1_MINI_2025_04_14 = "gpt-4.1-mini-2025-04-14"
    GPT_4_1_2025_04_14 = "gpt-4.1-2025-04-14"
    GPT_5_NANO_2025_08_07 = "gpt-5-nano-2025-08-07"
    GPT_5_MINI_2025_08_07 = "gpt-5-mini-2025-08-07"
    GPT_5_2025_08_07 = "gpt-5-2025-08-07"
    GPT_5_2_2025_12_11 = "gpt-5.2-2025-12-11"
    GPT_5_4 = "gpt-5.4"
    GPT_5_4_MINI = "gpt-5.4-mini"
    GPT_5_4_NANO = "gpt-5.4-nano"
    GPT_5_4_2026_03_05 = "gpt-5.4-2026-03-05"
    GPT_5_4_MINI_2026_03_17 = "gpt-5.4-mini-2026-03-17"
    GPT_5_4_NANO_2026_03_17 = "gpt-5.4-nano-2026-03-17"
    BEDROCK_OPENAI_GPT_5_4 = "openai.gpt-5.4"
    BEDROCK_OPENAI_GPT_5_5 = "openai.gpt-5.5"
    O3_DEEP_RESEARCH_2025_06_26 = "o3-deep-research-2025-06-26"
    O4_MINI_DEEP_RESEARCH_2025_06_26 = "o4-mini-deep-research-2025-06-26"


REASONING_MODELS = {
    ModelName.GPT_5_2025_08_07,
    ModelName.GPT_5_NANO_2025_08_07,
    ModelName.GPT_5_MINI_2025_08_07,
    ModelName.GPT_5_2_2025_12_11,
    ModelName.GPT_5_4,
    ModelName.GPT_5_4_MINI,
    ModelName.GPT_5_4_NANO,
    ModelName.GPT_5_4_2026_03_05,
    ModelName.GPT_5_4_MINI_2026_03_17,
    ModelName.GPT_5_4_NANO_2026_03_17,
}


DEFAULT_OPENAI_MODEL = ModelName.GPT_5_4_MINI_2026_03_17.value
DEFAULT_BEDROCK_OPENAI_MODEL = ModelName.BEDROCK_OPENAI_GPT_5_4.value
BEDROCK_OPENAI_MODELS = {
    ModelName.BEDROCK_OPENAI_GPT_5_4.value,
    ModelName.BEDROCK_OPENAI_GPT_5_5.value,
}
BEDROCK_OPENAI_BASE_URL = "https://bedrock-mantle.us-east-2.api.aws/openai/v1"
DEFAULT_REASONING_EFFORT = "none"


def is_reasoning_model(model: str) -> bool:
    try:
        return ModelName(model) in REASONING_MODELS
    except ValueError:
        return model.startswith(("gpt-5", "o3", "o4"))
