from __future__ import annotations


class HuggingFaceModel:
    KIMI_K2_INSTRUCT = "moonshotai/Kimi-K2-Instruct"
    KIMI_K2_THINKING = "moonshotai/Kimi-K2-Thinking"
    GPT_OSS_20B_CEREBRAS = "openai/gpt-oss-20b:cerebras"
    QWEN3_32B = "Qwen/Qwen3-32B"
    LLAMA_3_1_8B_INSTRUCT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    MISTRAL_NEMO_INSTRUCT = "mistralai/Mistral-Nemo-Instruct-2407"


class LocalModel:
    LLAMA_3_1_8B_INSTRUCT = "llama-3.1-8b-instruct"
    QWEN2_5_7B_INSTRUCT = "qwen2.5-7b-instruct"
    QWEN3_5_9B = "qwen3.5-9b"
    NEMOTRON_MINI_4B_INSTRUCT = "nemotron-mini-4b-instruct"
    GEMMA_4_E4B_IT = "gemma-4-e4b-it"
    MISTRAL_7B_INSTRUCT = "mistral-7b-instruct"


DEFAULT_HUGGINGFACE_TRADER_MODEL = HuggingFaceModel.KIMI_K2_INSTRUCT
DEFAULT_LOCAL_TRADER_MODEL = LocalModel.QWEN3_5_9B
DEFAULT_LLM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_LLM_STUDIO_REMOTE_DEVICE = "DESKTOP-89TCN67"
DEFAULT_LLM_STUDIO_REMOTE_MODEL_PATH = "1db783ee9370e4b499419e97c1cdf0be:lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"


ALTERNATIVE_TRADER_MODEL_PROFILES = {
    "hf_router_fast": {
        "provider": "huggingface",
        "model": DEFAULT_HUGGINGFACE_TRADER_MODEL,
        "notes": "Hosted Hugging Face router profile using Kimi for non-OpenAI trader experiments.",
    },
    "hf_kimi_thinking": {
        "provider": "huggingface",
        "model": HuggingFaceModel.KIMI_K2_THINKING,
        "notes": "Hosted Hugging Face Kimi thinking profile for deeper trader reasoning experiments.",
    },
    "local_llm_studio": {
        "provider": "llm_studio",
        "model": DEFAULT_LOCAL_TRADER_MODEL,
        "base_url": DEFAULT_LLM_STUDIO_BASE_URL,
        "notes": "Local OpenAI-compatible chat-completions profile, intended for LM Studio or similar local runtimes.",
    },
}
