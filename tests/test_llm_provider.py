"""Unit tests for llm_provider module."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch


class TestClampTemperature(unittest.TestCase):
    """Tests for _clamp_temperature helper."""

    def setUp(self):
        from llm_provider import _clamp_temperature

        self.clamp = _clamp_temperature

    def test_minimax_zero_temp_clamped(self):
        """MiniMax does not accept temperature=0, should be clamped to 0.01."""
        self.assertEqual(self.clamp(0, "minimax"), 0.01)

    def test_minimax_negative_temp_clamped(self):
        """Negative temperature should be clamped to 0.01 for MiniMax."""
        self.assertEqual(self.clamp(-1, "minimax"), 0.01)

    def test_minimax_valid_temp_unchanged(self):
        """Valid MiniMax temperature (0 < t <= 1) should pass through."""
        self.assertEqual(self.clamp(0.5, "minimax"), 0.5)

    def test_minimax_temp_above_one_clamped(self):
        """Temperature above 1.0 should be clamped for MiniMax."""
        self.assertEqual(self.clamp(1.5, "minimax"), 1.0)

    def test_minimax_exact_one_unchanged(self):
        """Temperature exactly 1.0 is valid for MiniMax."""
        self.assertEqual(self.clamp(1.0, "minimax"), 1.0)

    def test_openai_zero_temp_unchanged(self):
        """OpenAI accepts temperature=0, should not be modified."""
        self.assertEqual(self.clamp(0, "openai"), 0)

    def test_openai_high_temp_unchanged(self):
        """OpenAI accepts higher temperatures, should not be modified."""
        self.assertEqual(self.clamp(2.0, "openai"), 2.0)


class TestGetLlmProviderSelection(unittest.TestCase):
    """Tests for provider selection logic in get_llm."""

    def test_invalid_provider_raises(self):
        """Unsupported provider should raise ValueError."""
        from llm_provider import get_llm

        with self.assertRaises(ValueError) as ctx:
            get_llm(provider="unsupported_provider")
        self.assertIn("Unsupported LLM provider", str(ctx.exception))

    def test_env_var_selects_minimax(self):
        """LLM_PROVIDER env var should select the provider."""
        from llm_provider import get_llm

        with patch.dict(os.environ, {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "test-key"}):
            with patch("llm_provider._create_minimax_llm") as mock_create:
                mock_create.return_value = MagicMock()
                get_llm()
                mock_create.assert_called_once_with(0)

    def test_provider_arg_overrides_env(self):
        """Explicit provider argument should override LLM_PROVIDER env var."""
        from llm_provider import get_llm

        with patch.dict(os.environ, {"LLM_PROVIDER": "minimax"}):
            with patch("llm_provider._create_openai_llm") as mock_openai:
                mock_openai.return_value = MagicMock()
                get_llm(provider="openai")
                mock_openai.assert_called_once()

    def test_default_provider_is_openai(self):
        """Default provider should be OpenAI when no env var is set."""
        from llm_provider import get_llm

        env = {k: v for k, v in os.environ.items() if k != "LLM_PROVIDER"}
        with patch.dict(os.environ, env, clear=True):
            with patch("llm_provider._create_openai_llm") as mock_openai:
                mock_openai.return_value = MagicMock()
                get_llm()
                mock_openai.assert_called_once()

    def test_provider_case_insensitive(self):
        """Provider name should be case-insensitive."""
        from llm_provider import get_llm

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            with patch("llm_provider._create_minimax_llm") as mock_create:
                mock_create.return_value = MagicMock()
                get_llm(provider="MiniMax")
                mock_create.assert_called_once()

    def test_provider_whitespace_stripped(self):
        """Provider name whitespace should be stripped."""
        from llm_provider import get_llm

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            with patch("llm_provider._create_minimax_llm") as mock_create:
                mock_create.return_value = MagicMock()
                get_llm(provider="  minimax  ")
                mock_create.assert_called_once()

    def test_temperature_passed_to_openai(self):
        """Temperature should be forwarded to the OpenAI factory."""
        from llm_provider import get_llm

        with patch("llm_provider._create_openai_llm") as mock_openai:
            mock_openai.return_value = MagicMock()
            get_llm(provider="openai", temperature=0.7)
            mock_openai.assert_called_once_with(0.7)

    def test_temperature_passed_to_minimax(self):
        """Temperature should be forwarded to the MiniMax factory."""
        from llm_provider import get_llm

        with patch("llm_provider._create_minimax_llm") as mock_create:
            mock_create.return_value = MagicMock()
            get_llm(provider="minimax", temperature=0.5)
            mock_create.assert_called_once_with(0.5)


class TestMiniMaxProvider(unittest.TestCase):
    """Tests for MiniMax provider creation logic."""

    def _setup_langchain_mocks(self):
        """Set up mock langchain modules for import."""
        mock_chat_module = MagicMock()
        self._mock_ChatOpenAI = MagicMock()
        mock_chat_module.ChatOpenAI = self._mock_ChatOpenAI
        return {"langchain.chat_models": mock_chat_module}

    def test_missing_api_key_raises(self):
        """Missing MINIMAX_API_KEY should raise ValueError."""
        from llm_provider import _create_minimax_llm

        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                _create_minimax_llm(0.5)
            self.assertIn("MINIMAX_API_KEY", str(ctx.exception))

    def test_invalid_model_raises(self):
        """Unsupported MiniMax model should raise ValueError."""
        from llm_provider import _create_minimax_llm

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "k", "MINIMAX_MODEL": "bad-model"}):
            with self.assertRaises(ValueError) as ctx:
                _create_minimax_llm(0.5)
            self.assertIn("Unsupported MiniMax model", str(ctx.exception))

    def test_default_model_is_m27(self):
        """Default MiniMax model should be MiniMax-M2.7."""
        from llm_provider import _create_minimax_llm

        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_MODEL"}
        env["MINIMAX_API_KEY"] = "test-key"
        mocks = self._setup_langchain_mocks()
        with patch.dict(os.environ, env, clear=True):
            with patch.dict(sys.modules, mocks):
                _create_minimax_llm(0.5)
                call_kwargs = self._mock_ChatOpenAI.call_args[1]
                self.assertEqual(call_kwargs["model_name"], "MiniMax-M2.7")

    def test_custom_model_from_env(self):
        """MINIMAX_MODEL env var should set the model."""
        from llm_provider import _create_minimax_llm

        mocks = self._setup_langchain_mocks()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "k", "MINIMAX_MODEL": "MiniMax-M2.7-highspeed"}):
            with patch.dict(sys.modules, mocks):
                _create_minimax_llm(0.5)
                call_kwargs = self._mock_ChatOpenAI.call_args[1]
                self.assertEqual(call_kwargs["model_name"], "MiniMax-M2.7-highspeed")

    def test_api_base_url(self):
        """MiniMax should use the correct API base URL."""
        from llm_provider import _create_minimax_llm

        mocks = self._setup_langchain_mocks()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            with patch.dict(sys.modules, mocks):
                _create_minimax_llm(0.5)
                call_kwargs = self._mock_ChatOpenAI.call_args[1]
                self.assertEqual(call_kwargs["openai_api_base"], "https://api.minimax.io/v1")

    def test_temperature_clamped(self):
        """Temperature=0 should be clamped to 0.01 for MiniMax."""
        from llm_provider import _create_minimax_llm

        mocks = self._setup_langchain_mocks()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            with patch.dict(sys.modules, mocks):
                _create_minimax_llm(0)
                call_kwargs = self._mock_ChatOpenAI.call_args[1]
                self.assertEqual(call_kwargs["temperature"], 0.01)

    def test_valid_temperature_passed(self):
        """Valid temperature should be passed through."""
        from llm_provider import _create_minimax_llm

        mocks = self._setup_langchain_mocks()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            with patch.dict(sys.modules, mocks):
                _create_minimax_llm(0.7)
                call_kwargs = self._mock_ChatOpenAI.call_args[1]
                self.assertEqual(call_kwargs["temperature"], 0.7)

    def test_api_key_passed(self):
        """MINIMAX_API_KEY should be passed as openai_api_key."""
        from llm_provider import _create_minimax_llm

        mocks = self._setup_langchain_mocks()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "my-secret-key"}):
            with patch.dict(sys.modules, mocks):
                _create_minimax_llm(0.5)
                call_kwargs = self._mock_ChatOpenAI.call_args[1]
                self.assertEqual(call_kwargs["openai_api_key"], "my-secret-key")

    def test_all_m25_models_accepted(self):
        """All MiniMax M2.5 model variants should be accepted."""
        from llm_provider import _create_minimax_llm

        for model in ("MiniMax-M2.5", "MiniMax-M2.5-highspeed"):
            mocks = self._setup_langchain_mocks()
            with patch.dict(os.environ, {"MINIMAX_API_KEY": "k", "MINIMAX_MODEL": model}):
                with patch.dict(sys.modules, mocks):
                    _create_minimax_llm(0.5)
                    call_kwargs = self._mock_ChatOpenAI.call_args[1]
                    self.assertEqual(call_kwargs["model_name"], model)


class TestOpenAIProvider(unittest.TestCase):
    """Tests for OpenAI provider creation."""

    def _setup_langchain_mocks(self):
        """Set up mock langchain modules for import."""
        mock_llm_module = MagicMock()
        self._mock_OpenAI = MagicMock()
        mock_llm_module.OpenAI = self._mock_OpenAI
        return {"langchain.llms.openai": mock_llm_module}

    def test_openai_temperature_passed(self):
        """Temperature should be passed through for OpenAI."""
        from llm_provider import _create_openai_llm

        mocks = self._setup_langchain_mocks()
        with patch.dict(sys.modules, mocks):
            _create_openai_llm(0.7)
            self._mock_OpenAI.assert_called_once_with(temperature=0.7)

    def test_openai_zero_temp(self):
        """OpenAI should accept temperature=0."""
        from llm_provider import _create_openai_llm

        mocks = self._setup_langchain_mocks()
        with patch.dict(sys.modules, mocks):
            _create_openai_llm(0)
            self._mock_OpenAI.assert_called_once_with(temperature=0)


class TestSupportedModels(unittest.TestCase):
    """Tests for model and provider constants."""

    def test_minimax_models_defined(self):
        """All expected MiniMax models should be listed."""
        from llm_provider import MINIMAX_MODELS

        expected = {"MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M2.5", "MiniMax-M2.5-highspeed"}
        self.assertEqual(set(MINIMAX_MODELS.keys()), expected)

    def test_minimax_context_windows(self):
        """All MiniMax models should have 204K context window."""
        from llm_provider import MINIMAX_MODELS

        for model, ctx in MINIMAX_MODELS.items():
            self.assertEqual(ctx, 204_000, f"{model} should have 204K context")

    def test_supported_providers_tuple(self):
        """Supported providers should include openai and minimax."""
        from llm_provider import SUPPORTED_PROVIDERS

        self.assertIn("openai", SUPPORTED_PROVIDERS)
        self.assertIn("minimax", SUPPORTED_PROVIDERS)


if __name__ == "__main__":
    unittest.main()
