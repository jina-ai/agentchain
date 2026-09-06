"""Unit tests for the llm_provider registry."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class TestClampTemperature(unittest.TestCase):
    def setUp(self):
        from llm_provider import _clamp_temperature
        self.clamp = _clamp_temperature

    def test_minimax_zero_temp_clamped(self):
        self.assertEqual(self.clamp(0, "minimax"), 0.01)

    def test_minimax_negative_temp_clamped(self):
        self.assertEqual(self.clamp(-1, "minimax"), 0.01)

    def test_minimax_valid_temp_unchanged(self):
        self.assertEqual(self.clamp(0.5, "minimax"), 0.5)

    def test_minimax_temp_above_one_clamped(self):
        self.assertEqual(self.clamp(1.5, "minimax"), 1.0)

    def test_minimax_exact_one_unchanged(self):
        self.assertEqual(self.clamp(1.0, "minimax"), 1.0)

    def test_openai_zero_temp_unchanged(self):
        self.assertEqual(self.clamp(0, "openai"), 0)

    def test_openai_high_temp_unchanged(self):
        self.assertEqual(self.clamp(2.0, "openai"), 2.0)


class TestProviderSelection(unittest.TestCase):
    def test_invalid_provider_raises(self):
        from llm_provider import get_llm
        with self.assertRaises(ValueError) as ctx:
            get_llm(provider="unsupported_provider")
        self.assertIn("Unsupported LLM provider", str(ctx.exception))

    def test_env_var_selects_minimax(self):
        from llm_provider import get_llm
        with patch.dict(os.environ, {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "k"}):
            with patch("llm_provider._create_minimax_llm") as mock_create:
                mock_create.return_value = MagicMock()
                get_llm()
                mock_create.assert_called_once_with(0)

    def test_provider_arg_overrides_env(self):
        from llm_provider import get_llm
        with patch.dict(os.environ, {"LLM_PROVIDER": "minimax"}):
            with patch("llm_provider._create_openai_llm") as mock_openai:
                mock_openai.return_value = MagicMock()
                get_llm(provider="openai")
                mock_openai.assert_called_once()

    def test_default_provider_is_openai(self):
        from llm_provider import get_llm
        env = {k: v for k, v in os.environ.items() if k != "LLM_PROVIDER"}
        with patch.dict(os.environ, env, clear=True):
            with patch("llm_provider._create_openai_llm") as mock_openai:
                mock_openai.return_value = MagicMock()
                get_llm()
                mock_openai.assert_called_once()

    def test_provider_case_insensitive(self):
        from llm_provider import get_llm
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "k"}):
            with patch("llm_provider._create_minimax_llm") as mock_create:
                mock_create.return_value = MagicMock()
                get_llm(provider="MiniMax")
                mock_create.assert_called_once()

    def test_temperature_passed_to_minimax(self):
        from llm_provider import get_llm
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "k"}):
            with patch("llm_provider._create_minimax_llm") as mock_create:
                mock_create.return_value = MagicMock()
                get_llm(provider="minimax", temperature=0.5)
                mock_create.assert_called_once_with(0.5)


class TestMiniMaxProvider(unittest.TestCase):
    def _setup_langchain_mocks(self):
        mock_chat_module = MagicMock()
        self._mock_ChatOpenAI = MagicMock()
        mock_chat_module.ChatOpenAI = self._mock_ChatOpenAI
        return {"langchain.chat_models": mock_chat_module}

    def test_missing_api_key_raises(self):
        from llm_provider import _create_minimax_llm
        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                _create_minimax_llm(0.5)
            self.assertIn("MINIMAX_API_KEY", str(ctx.exception))

    def test_invalid_model_raises(self):
        from llm_provider import _create_minimax_llm
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "k", "MINIMAX_MODEL": "bad-model"}):
            with self.assertRaises(ValueError) as ctx:
                _create_minimax_llm(0.5)
            self.assertIn("Unsupported MiniMax model", str(ctx.exception))

    def test_default_model_is_m3(self):
        from llm_provider import _create_minimax_llm
        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_MODEL"}
        env["MINIMAX_API_KEY"] = "test-key"
        mocks = self._setup_langchain_mocks()
        with patch.dict(os.environ, env, clear=True):
            with patch.dict(sys.modules, mocks):
                _create_minimax_llm(0.5)
                call_kwargs = self._mock_ChatOpenAI.call_args[1]
                self.assertEqual(call_kwargs["model_name"], "MiniMax-M3")

    def test_custom_model_from_env(self):
        from llm_provider import _create_minimax_llm
        mocks = self._setup_langchain_mocks()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "k", "MINIMAX_MODEL": "MiniMax-M2.7"}):
            with patch.dict(sys.modules, mocks):
                _create_minimax_llm(0.5)
                call_kwargs = self._mock_ChatOpenAI.call_args[1]
                self.assertEqual(call_kwargs["model_name"], "MiniMax-M2.7")

    def test_default_region_is_global(self):
        from llm_provider import _create_minimax_llm
        mocks = self._setup_langchain_mocks()
        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_REGION"}
        env["MINIMAX_API_KEY"] = "test-key"
        with patch.dict(os.environ, env, clear=True):
            with patch.dict(sys.modules, mocks):
                _create_minimax_llm(0.5)
                call_kwargs = self._mock_ChatOpenAI.call_args[1]
                self.assertEqual(call_kwargs["openai_api_base"], "https://api.minimax.io/v1")

    def test_china_region_base_url(self):
        from llm_provider import _create_minimax_llm
        mocks = self._setup_langchain_mocks()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "k", "MINIMAX_REGION": "cn_zh"}):
            with patch.dict(sys.modules, mocks):
                _create_minimax_llm(0.5)
                call_kwargs = self._mock_ChatOpenAI.call_args[1]
                self.assertEqual(call_kwargs["openai_api_base"], "https://api.minimaxi.com/v1")

    def test_unknown_region_raises(self):
        from llm_provider import _create_minimax_llm
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "k", "MINIMAX_REGION": "eu-west"}):
            with self.assertRaises(ValueError) as ctx:
                _create_minimax_llm(0.5)
            self.assertIn("Unsupported MiniMax region", str(ctx.exception))

    def test_anthropic_endpoint_can_be_resolved(self):
        from llm_provider import _resolve_minimax_base_url
        with patch.dict(os.environ, {"MINIMAX_REGION": "global_en"}):
            self.assertEqual(
                _resolve_minimax_base_url("anthropic"),
                "https://api.minimax.io/anthropic",
            )

    def test_unknown_api_format_raises(self):
        from llm_provider import _resolve_minimax_base_url
        with self.assertRaises(ValueError) as ctx:
            _resolve_minimax_base_url("unsupported")
        self.assertIn("Unsupported MiniMax API format", str(ctx.exception))

    def test_temperature_clamped(self):
        from llm_provider import _create_minimax_llm
        mocks = self._setup_langchain_mocks()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            with patch.dict(sys.modules, mocks):
                _create_minimax_llm(0)
                call_kwargs = self._mock_ChatOpenAI.call_args[1]
                self.assertEqual(call_kwargs["temperature"], 0.01)

    def test_api_key_passed(self):
        from llm_provider import _create_minimax_llm
        mocks = self._setup_langchain_mocks()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "dummy-key-value"}):
            with patch.dict(sys.modules, mocks):
                _create_minimax_llm(0.5)
                call_kwargs = self._mock_ChatOpenAI.call_args[1]
                self.assertEqual(call_kwargs["openai_api_key"], "dummy-key-value")


class TestOpenAIProvider(unittest.TestCase):
    def _setup_langchain_mocks(self):
        mock_llm_module = MagicMock()
        self._mock_OpenAI = MagicMock()
        mock_llm_module.OpenAI = self._mock_OpenAI
        return {"langchain.llms.openai": mock_llm_module}

    def test_openai_temperature_passed(self):
        from llm_provider import _create_openai_llm
        mocks = self._setup_langchain_mocks()
        with patch.dict(sys.modules, mocks):
            _create_openai_llm(0.7)
            self._mock_OpenAI.assert_called_once_with(temperature=0.7)


class TestRegistryConstants(unittest.TestCase):
    def test_minimax_models_defined(self):
        from llm_provider import MINIMAX_MODELS
        self.assertEqual(set(MINIMAX_MODELS.keys()), {"MiniMax-M3", "MiniMax-M2.7"})

    def test_m3_context_window(self):
        from llm_provider import MINIMAX_MODELS
        self.assertEqual(MINIMAX_MODELS["MiniMax-M3"]["context_window"], 1_000_000)
        self.assertEqual(
            MINIMAX_MODELS["MiniMax-M3"]["input_modalities"],
            ("text", "image", "video"),
        )
        self.assertEqual(
            MINIMAX_MODELS["MiniMax-M3"]["thinking_modes"],
            ("adaptive", "disabled"),
        )

    def test_m27_context_window(self):
        from llm_provider import MINIMAX_MODELS
        self.assertEqual(MINIMAX_MODELS["MiniMax-M2.7"]["context_window"], 204_800)
        self.assertEqual(MINIMAX_MODELS["MiniMax-M2.7"]["input_modalities"], ("text",))
        self.assertEqual(
            MINIMAX_MODELS["MiniMax-M2.7"]["thinking_modes"],
            ("always-on",),
        )

    def test_regional_endpoints(self):
        from llm_provider import MINIMAX_REGIONAL_ENDPOINTS
        self.assertEqual(
            MINIMAX_REGIONAL_ENDPOINTS["global_en"],
            {
                "openai": "https://api.minimax.io/v1",
                "anthropic": "https://api.minimax.io/anthropic",
            },
        )
        self.assertEqual(
            MINIMAX_REGIONAL_ENDPOINTS["cn_zh"],
            {
                "openai": "https://api.minimaxi.com/v1",
                "anthropic": "https://api.minimaxi.com/anthropic",
            },
        )

    def test_supported_providers(self):
        from llm_provider import SUPPORTED_PROVIDERS
        self.assertIn("openai", SUPPORTED_PROVIDERS)
        self.assertIn("minimax", SUPPORTED_PROVIDERS)


if __name__ == "__main__":
    unittest.main()
