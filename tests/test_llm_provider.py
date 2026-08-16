"""Unit tests for the AgentChain LLM provider registry."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestProviderSelection(unittest.TestCase):
    def test_default_provider_preserves_existing_llm(self):
        from llm_provider import get_llm

        with patch("llm_provider._create_openai_llm") as create_default:
            create_default.return_value = MagicMock()
            get_llm(temperature=0.4)
            create_default.assert_called_once_with(0.4)

    def test_environment_selects_minimax(self):
        from llm_provider import get_llm

        with patch.dict(os.environ, {"LLM_PROVIDER": "minimax"}):
            with patch("llm_provider._create_minimax_llm") as create_minimax:
                create_minimax.return_value = MagicMock()
                get_llm(temperature=0.4)
                create_minimax.assert_called_once_with(0.4)

    def test_unknown_provider_is_rejected(self):
        from llm_provider import get_llm

        with self.assertRaisesRegex(ValueError, "Unsupported LLM provider"):
            get_llm(provider="unknown")


class TestMiniMaxRegistry(unittest.TestCase):
    def test_models_have_current_context_windows(self):
        from llm_provider import MINIMAX_MODELS

        self.assertEqual(MINIMAX_MODELS["MiniMax-M3"]["context_window"], 1_000_000)
        self.assertEqual(MINIMAX_MODELS["MiniMax-M2.7"]["context_window"], 204_800)

    def test_models_have_current_pricing(self):
        from llm_provider import MINIMAX_MODELS

        self.assertEqual(
            MINIMAX_MODELS["MiniMax-M3"]["pricing_usd_per_million_tokens"],
            {"input": 0.6, "output": 2.4, "cache_read": 0.12, "cache_write": None},
        )
        self.assertEqual(
            MINIMAX_MODELS["MiniMax-M2.7"]["pricing_usd_per_million_tokens"],
            {"input": 0.3, "output": 1.2, "cache_read": 0.06, "cache_write": 0.375},
        )

    def test_m3_declares_multimodal_inputs_and_thinking_modes(self):
        from llm_provider import MINIMAX_MODELS

        model = MINIMAX_MODELS["MiniMax-M3"]
        self.assertEqual(model["input_modalities"], ("text", "image", "video"))
        self.assertEqual(model["thinking"], ("adaptive", "disabled"))

    def test_m27_declares_text_input_and_always_on_thinking(self):
        from llm_provider import MINIMAX_MODELS

        model = MINIMAX_MODELS["MiniMax-M2.7"]
        self.assertEqual(model["input_modalities"], ("text",))
        self.assertEqual(model["thinking"], ("always_on",))


class TestMiniMaxEndpoints(unittest.TestCase):
    def test_global_endpoints(self):
        from llm_provider import _resolve_minimax_base_url

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_resolve_minimax_base_url(), "https://api.minimax.io/v1")
            self.assertEqual(
                _resolve_minimax_base_url("anthropic"),
                "https://api.minimax.io/anthropic",
            )

    def test_china_endpoints(self):
        from llm_provider import _resolve_minimax_base_url

        with patch.dict(os.environ, {"MINIMAX_REGION": "cn_zh"}, clear=True):
            self.assertEqual(_resolve_minimax_base_url(), "https://api.minimaxi.com/v1")
            self.assertEqual(
                _resolve_minimax_base_url("anthropic"),
                "https://api.minimaxi.com/anthropic",
            )

    def test_unknown_region_is_rejected(self):
        from llm_provider import _resolve_minimax_base_url

        with patch.dict(os.environ, {"MINIMAX_REGION": "unknown"}, clear=True):
            with self.assertRaisesRegex(ValueError, "Unsupported MiniMax region"):
                _resolve_minimax_base_url()

    def test_unknown_api_style_is_rejected(self):
        from llm_provider import _resolve_minimax_base_url

        with self.assertRaisesRegex(ValueError, "Unsupported MiniMax API style"):
            _resolve_minimax_base_url("unknown")


class TestMiniMaxConstruction(unittest.TestCase):
    def setUp(self):
        self.chat_module = MagicMock()
        self.chat_openai = MagicMock()
        self.chat_module.ChatOpenAI = self.chat_openai

    def test_missing_api_key_is_rejected(self):
        from llm_provider import _create_minimax_llm

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "MINIMAX_API_KEY"):
                _create_minimax_llm(0.4)

    def test_unknown_model_is_rejected(self):
        from llm_provider import _create_minimax_llm

        env = {"MINIMAX_API_KEY": "stub", "MINIMAX_MODEL": "unknown"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "Unsupported MiniMax model"):
                _create_minimax_llm(0.4)

    def test_default_model_uses_global_openai_endpoint(self):
        from llm_provider import _create_minimax_llm

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "stub"}, clear=True):
            with patch.dict(sys.modules, {"langchain.chat_models": self.chat_module}):
                _create_minimax_llm(0)

        self.chat_openai.assert_called_once_with(
            model_name="MiniMax-M3",
            openai_api_key="stub",
            openai_api_base="https://api.minimax.io/v1",
            temperature=0.01,
        )

    def test_china_region_and_custom_model_are_forwarded(self):
        from llm_provider import _create_minimax_llm

        env = {
            "MINIMAX_API_KEY": "stub",
            "MINIMAX_MODEL": "MiniMax-M2.7",
            "MINIMAX_REGION": "cn_zh",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.dict(sys.modules, {"langchain.chat_models": self.chat_module}):
                _create_minimax_llm(0.5)

        self.chat_openai.assert_called_once_with(
            model_name="MiniMax-M2.7",
            openai_api_key="stub",
            openai_api_base="https://api.minimaxi.com/v1",
            temperature=0.5,
        )


if __name__ == "__main__":
    unittest.main()
