"""LLM provider factory for AgentChain.

Supports multiple LLM providers:
- OpenAI (default): Uses the OpenAI completions API via langchain.llms.openai.OpenAI
- MiniMax: Uses MiniMax's OpenAI-compatible API via langchain.chat_models.ChatOpenAI

Usage:
    Set the LLM_PROVIDER environment variable to select a provider:
        export LLM_PROVIDER=openai     # default
        export LLM_PROVIDER=minimax    # MiniMax M2.7

    For MiniMax, also set:
        export MINIMAX_API_KEY=your_key
        export MINIMAX_MODEL=MiniMax-M2.7  # optional, default: MiniMax-M2.7
"""

import os

# Supported MiniMax models with their context windows (tokens)
MINIMAX_MODELS = {
    "MiniMax-M2.7": 204_000,
    "MiniMax-M2.7-highspeed": 204_000,
    "MiniMax-M2.5": 204_000,
    "MiniMax-M2.5-highspeed": 204_000,
}

SUPPORTED_PROVIDERS = ("openai", "minimax")


def _clamp_temperature(temperature, provider):
    """Clamp temperature to valid range for the given provider.

    MiniMax requires temperature in (0.0, 1.0].
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
        provider: LLM provider name. Defaults to LLM_PROVIDER env var or 'openai'.
        temperature: Sampling temperature. Will be clamped for providers with
            restricted ranges (e.g., MiniMax requires > 0).

    Returns:
        A LangChain LLM or ChatModel instance.

    Raises:
        ValueError: If the provider is not supported or required config is missing.
    """
    provider = (provider or os.environ.get("LLM_PROVIDER", "openai")).lower().strip()

    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    if provider == "minimax":
        return _create_minimax_llm(temperature)

    # Default: OpenAI
    return _create_openai_llm(temperature)


def _create_openai_llm(temperature):
    """Create an OpenAI LLM instance."""
    from langchain.llms.openai import OpenAI

    return OpenAI(temperature=temperature)


def _create_minimax_llm(temperature):
    """Create a MiniMax LLM instance via OpenAI-compatible API."""
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError(
            "MINIMAX_API_KEY environment variable is required for MiniMax provider. "
            "Get your API key at https://platform.minimaxi.com/"
        )

    model = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7")
    if model not in MINIMAX_MODELS:
        raise ValueError(
            f"Unsupported MiniMax model: '{model}'. "
            f"Supported models: {', '.join(MINIMAX_MODELS.keys())}"
        )

    from langchain.chat_models import ChatOpenAI

    temperature = _clamp_temperature(temperature, "minimax")

    return ChatOpenAI(
        model_name=model,
        openai_api_key=api_key,
        openai_api_base="https://api.minimax.io/v1",
        temperature=temperature,
    )
