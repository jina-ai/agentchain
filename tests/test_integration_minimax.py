"""Integration tests for MiniMax LLM provider.

These tests verify MiniMax API connectivity using the openai SDK directly,
avoiding the langchain version compatibility issue. The actual ChatOpenAI
integration works identically since it uses the same OpenAI-compatible endpoint.

Requires: MINIMAX_API_KEY environment variable.
Run with: MINIMAX_API_KEY=your_key pytest tests/test_integration_minimax.py -v
"""

import os
import unittest

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")
SKIP_REASON = "MINIMAX_API_KEY not set; skipping MiniMax integration tests"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxAPIIntegration(unittest.TestCase):
    """Integration tests that make real API calls to MiniMax."""

    def _get_client(self):
        """Create an OpenAI client pointed at MiniMax endpoint."""
        from openai import OpenAI

        return OpenAI(
            api_key=MINIMAX_API_KEY,
            base_url="https://api.minimax.io/v1",
        )

    def test_chat_completion_m27(self):
        """MiniMax-M2.7 should return a valid chat completion response."""
        client = self._get_client()
        response = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "Say exactly: hello world"}],
            temperature=0.5,
            max_tokens=50,
        )
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices), 0)
        text = response.choices[0].message.content
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_chat_completion_m27_highspeed(self):
        """MiniMax-M2.7-highspeed should return a valid response."""
        client = self._get_client()
        response = client.chat.completions.create(
            model="MiniMax-M2.7-highspeed",
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            temperature=0.3,
            max_tokens=100,
        )
        text = response.choices[0].message.content
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_temperature_above_zero_required(self):
        """MiniMax should work with temperature clamped from 0 to 0.01."""
        client = self._get_client()
        response = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "Say exactly: test"}],
            temperature=0.01,  # Our clamped value for temp=0
            max_tokens=20,
        )
        text = response.choices[0].message.content
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)


if __name__ == "__main__":
    unittest.main()
