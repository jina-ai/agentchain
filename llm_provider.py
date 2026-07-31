"""LLM provider registry for AgentChain.

A single construction point for the central LLM used by ``ConversationBot``.
The default provider is OpenAI (mirroring the original behavior of
``OpenAI(temperature=0)``). Setting the ``LLM_PROVIDER`` environment variable
to ``minimax`` routes the agent through MiniMax's OpenAI-compatible endpoint.

MiniMax exposes both global and China OpenAI-compatible endpoints. The region is
selected through ``MINIMAX_REGION`` (``global_en`` by default, ``cn_zh`` for the
China endpoint). See the ``MINIMAX_REGIONAL_ENDPOINTS`` mapping below.

Usage:
    export LLM_PROVIDER=openai        # default
    export LLM_PROVIDER=minimax       # MiniMax M3 / M2.7

    # For MiniMax, also set:
    export MINIMAX_API_KEY=your_key
    export MINIMAX_MODEL=MiniMax-M3        # optional, default: MiniMax-M3
    export MINIMAX_REGION=global_en        # optional, default: global_en
"""

import os

# MiniMax text models currently supported by AgentChain.
# Each entry maps a model id to its context window (in tokens). Context windows
# are sourced from the MiniMax model configuration and must not be hardcoded
# to a single value across models.
MINIMAX_MODELS = {
    "MiniMax-M3": 1_000_000,
    "MiniMax-M2.7": 204_800,
}

DEFAULT_MINIMAX_MODEL = "MiniMax-M3"

# Regional OpenAI-compatible endpoints for MiniMax.
# - global_en: https://api.minimax.io
# - cn_zh:     https://api.minimaxi.com (China)
MINIMAX_REGIONAL_ENDPOINTS = {
    "global_en": "https://api.minimax.io/v1",
    "cn_zh": "https://api.minimaxi.com/v1",
}

DEFAULT_MINIMAX_REGION = "global_en"

SUPPORTED_PROVIDERS = ("openai", "minimax")


def _clamp_temperature(temperature, provider):
    """Clamp temperature to a valid range for the given provider.

    MiniMax requires the sampling temperature to be strictly greater than 0,
    so a temperature of 0 is bumped to a small positive value. Values above
    1.0 are clamped to 1.0, the top of MiniMax's accepted range.
    """
    if provider == "minimax":
        if temperature <= 0:
            return 0.01
        if temperature > 1.0:
            return 1.0
    return temperature


def get_llm(provider=None, temperature=0):
    """Create an LLM instance based on the provider configuration.

    Args:
        provider: LLM provider name. Defaults to the ``LLM_PROVIDER`` env var
            or ``openai`` when unset. Matching is case-insensitive.
        temperature: Sampling temperature. Clamped per provider before use.

    Returns:
        A LangChain LLM or ChatModel instance.

    Raises:
        ValueError: If the provider is unsupported, the region is unknown, or a
            required environment variable is missing.
    """
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
    """Create an OpenAI LLM instance (the default provider)."""
    from langchain.llms.openai import OpenAI

    return OpenAI(temperature=temperature)


def _resolve_minimax_base_url():
    """Resolve the MiniMax OpenAI-compatible base URL from the selected region."""
    region = os.environ.get("MINIMAX_REGION", DEFAULT_MINIMAX_REGION).lower().strip()
    if region not in MINIMAX_REGIONAL_ENDPOINTS:
        raise ValueError(
            f"Unsupported MiniMax region: '{region}'. "
            f"Supported regions: {', '.join(MINIMAX_REGIONAL_ENDPOINTS.keys())}"
        )
    return MINIMAX_REGIONAL_ENDPOINTS[region]


def _create_minimax_llm(temperature):
    """Create a MiniMax LLM instance via the OpenAI-compatible API."""
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError(
            "MINIMAX_API_KEY environment variable is required for the MiniMax "
            "provider. Get your API key at https://platform.minimax.io/"
        )

    model = os.environ.get("MINIMAX_MODEL", DEFAULT_MINIMAX_MODEL)
    if model not in MINIMAX_MODELS:
        raise ValueError(
            f"Unsupported MiniMax model: '{model}'. "
            f"Supported models: {', '.join(MINIMAX_MODELS.keys())}"
        )

    base_url = _resolve_minimax_base_url()

    from langchain.chat_models import ChatOpenAI

    temperature = _clamp_temperature(temperature, "minimax")

    return ChatOpenAI(
        model_name=model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=temperature,
    )
