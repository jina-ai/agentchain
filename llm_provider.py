"""LLM provider registry for AgentChain.

The default construction path preserves AgentChain's existing behavior. Set
``LLM_PROVIDER=minimax`` to use MiniMax through its OpenAI-compatible API, and
select a regional endpoint with ``MINIMAX_REGION``.
"""

import os


MINIMAX_MODELS = {
    "MiniMax-M3": {
        "context_window": 1_000_000,
        "pricing_usd_per_million_tokens": {
            "input": 0.6,
            "output": 2.4,
            "cache_read": 0.12,
            "cache_write": None,
        },
        "input_modalities": ("text", "image", "video"),
        "thinking": ("adaptive", "disabled"),
    },
    "MiniMax-M2.7": {
        "context_window": 204_800,
        "pricing_usd_per_million_tokens": {
            "input": 0.3,
            "output": 1.2,
            "cache_read": 0.06,
            "cache_write": 0.375,
        },
        "input_modalities": ("text",),
        "thinking": ("always_on",),
    },
}

DEFAULT_MINIMAX_MODEL = "MiniMax-M3"

MINIMAX_REGIONAL_ENDPOINTS = {
    "global_en": {
        "openai": "https://api.minimax.io/v1",
        "anthropic": "https://api.minimax.io/anthropic",
    },
    "cn_zh": {
        "openai": "https://api.minimaxi.com/v1",
        "anthropic": "https://api.minimaxi.com/anthropic",
    },
}

DEFAULT_MINIMAX_REGION = "global_en"
SUPPORTED_PROVIDERS = ("openai", "minimax")


def _clamp_temperature(temperature, provider):
    """Clamp temperature to the accepted range for a provider."""
    if provider == "minimax":
        if temperature <= 0:
            return 0.01
        if temperature > 1.0:
            return 1.0
    return temperature


def get_llm(provider=None, temperature=0):
    """Create the central LLM from explicit or environment configuration."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "openai")).lower().strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    if provider == "minimax":
        return _create_minimax_llm(temperature)
    return _create_openai_llm(temperature)


def _create_openai_llm(temperature):
    """Create the default LLM used by AgentChain."""
    from langchain.llms.openai import OpenAI

    return OpenAI(temperature=temperature)


def _resolve_minimax_base_url(api_style="openai"):
    """Resolve an OpenAI- or Anthropic-compatible regional base URL."""
    region = os.environ.get("MINIMAX_REGION", DEFAULT_MINIMAX_REGION).lower().strip()
    if region not in MINIMAX_REGIONAL_ENDPOINTS:
        raise ValueError(
            f"Unsupported MiniMax region: '{region}'. "
            f"Supported regions: {', '.join(MINIMAX_REGIONAL_ENDPOINTS.keys())}"
        )

    endpoints = MINIMAX_REGIONAL_ENDPOINTS[region]
    if api_style not in endpoints:
        raise ValueError(
            f"Unsupported MiniMax API style: '{api_style}'. "
            f"Supported styles: {', '.join(endpoints.keys())}"
        )
    return endpoints[api_style]


def _create_minimax_llm(temperature):
    """Create a MiniMax LLM through the OpenAI-compatible transport."""
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError("MINIMAX_API_KEY environment variable is required")

    model = os.environ.get("MINIMAX_MODEL", DEFAULT_MINIMAX_MODEL)
    if model not in MINIMAX_MODELS:
        raise ValueError(
            f"Unsupported MiniMax model: '{model}'. "
            f"Supported models: {', '.join(MINIMAX_MODELS.keys())}"
        )

    from langchain.chat_models import ChatOpenAI

    return ChatOpenAI(
        model_name=model,
        openai_api_key=api_key,
        openai_api_base=_resolve_minimax_base_url("openai"),
        temperature=_clamp_temperature(temperature, "minimax"),
    )
