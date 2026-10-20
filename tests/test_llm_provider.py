"""Unit tests for the AgentChain LLM provider registry."""

import os
import unittest
from unittest.mock import patch

from llm_provider import (
    DEFAULT_MINIMAX_MODEL,
    MINIMAX_MODELS,
    _clamp_temperature,
    _resolve_minimax_base_url,
    get_llm,
)


class TestProviderSelection(unittest.TestCase):
    def test_default_provider_preserves_existing_llm(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("llm_provider._create_openai_llm") as create:
                get_llm(temperature=0.4)
                create.assert_called_once_with(0.4)

    def test_environment_selects_minimax(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "minimax"}, clear=True):
            with patch("llm_provider._create_minimax_llm") as create:
                get_llm(temperature=0.4)
                create.assert_called_once_with(0.4)

    def test_unknown_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported LLM provider"):
            get_llm(provider="unknown")

    def test_missing_api_key_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "MINIMAX_API_KEY"):
                get_llm(provider="minimax")

    def test_blank_model_is_rejected(self):
        env = {"MINIMAX_API_KEY": "fixture", "MINIMAX_MODEL": "  "}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "MINIMAX_MODEL"):
                get_llm(provider="minimax")


class TestMiniMaxRegistry(unittest.TestCase):
    def test_ordinary_catalog_is_flash_first(self):
        self.assertEqual(list(MINIMAX_MODELS), ["MiniMax-M3.1-Flash", "MiniMax-M3"])
        self.assertEqual(DEFAULT_MINIMAX_MODEL, "MiniMax-M3.1-Flash")
        for model in MINIMAX_MODELS.values():
            self.assertEqual(model["context_window"], 1_000_000)
            self.assertEqual(model["input_modalities"], ("text", "image", "video"))

    def test_flash_contract(self):
        model = MINIMAX_MODELS[DEFAULT_MINIMAX_MODEL]
        self.assertEqual(model["max_output_tokens"], 524_288)
        self.assertEqual(model["thinking"], ("adaptive",))
        self.assertEqual(model["reasoning_effort"], ("low", "medium", "high", "xhigh", "max"))
        self.assertEqual(
            model["pricing_usd_per_million_tokens"],
            {"input": 0.15, "output": 0.6, "cache_read": 0.015, "cache_write": None},
        )

    def test_m3_contract(self):
        model = MINIMAX_MODELS["MiniMax-M3"]
        self.assertEqual(model["thinking"], ("adaptive", "disabled"))
        self.assertEqual(
            model["pricing_usd_per_million_tokens"],
            {"input": 0.3, "output": 1.2, "cache_read": 0.06, "cache_write": None},
        )
        self.assertEqual(
            model["pricing_above_512k_usd_per_million_tokens"],
            {"input": 0.6, "output": 2.4, "cache_read": 0.12, "cache_write": None},
        )
        self.assertEqual(model["priority_price_multiplier"], 1.5)

    def test_temperature_clamping_preserves_legacy_behavior(self):
        self.assertEqual(_clamp_temperature(0, "minimax"), 0.01)
        self.assertEqual(_clamp_temperature(2, "minimax"), 1)
        self.assertEqual(_clamp_temperature(0.5, "minimax"), 0.5)
        self.assertEqual(_clamp_temperature(0, "openai"), 0)


class TestMiniMaxEndpoints(unittest.TestCase):
    def test_regional_endpoint_metadata(self):
        for region, host in [("global_en", "api.minimax.io"), ("cn_zh", "api.minimaxi.com")]:
            with self.subTest(region=region):
                with patch.dict(os.environ, {"MINIMAX_REGION": region}, clear=True):
                    self.assertEqual(_resolve_minimax_base_url(), f"https://{host}/v1")
                    self.assertEqual(_resolve_minimax_base_url("anthropic"), f"https://{host}/anthropic")

    def test_unknown_region_is_rejected(self):
        with patch.dict(os.environ, {"MINIMAX_REGION": "unknown"}, clear=True):
            with self.assertRaisesRegex(ValueError, "Unsupported MiniMax region"):
                _resolve_minimax_base_url()

    def test_unknown_api_style_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Unsupported MiniMax API style"):
                _resolve_minimax_base_url("unknown")


if __name__ == "__main__":
    unittest.main()
