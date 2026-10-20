"""Exercise the real pinned LangChain/SDK stack at the prepared HTTP boundary."""

import ast
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import openai
import requests
from langchain.agents import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory

from llm_provider import get_llm


class TestMiniMaxTransport(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.environment = patch.dict(os.environ, {
            "LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "fixture-minimax",
        }, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.transport = patch.object(requests.Session, "send", self.respond)
        self.transport.start()
        self.addCleanup(self.transport.stop)
        self.globals = patch.multiple(openai, api_key="fixture-existing", api_base="https://api.openai.com/v1")
        self.globals.start()
        self.addCleanup(self.globals.stop)

    def respond(self, request, **kwargs):
        self.requests.append(request)
        response = requests.Response()
        response.status_code = 200
        response.headers["Content-Type"] = "application/json"
        response._content = json.dumps({
            "id": "fixture", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "Final Answer: fixture answer"},
                         "text": "fixture completion"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        return response

    def test_default_saved_legacy_and_custom_models_in_both_regions(self):
        models = [None, "MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "custom-saved-model"]
        for region, host in [("global_en", "api.minimax.io"), ("cn_zh", "api.minimaxi.com")]:
            for model in models:
                with self.subTest(region=region, model=model):
                    env = {"MINIMAX_REGION": region}
                    if model is not None:
                        env["MINIMAX_MODEL"] = model
                    with patch.dict(os.environ, env):
                        llm = get_llm(temperature=0)
                        self.assertEqual(llm("saved conversation", stop=["Observation:"]), "Final Answer: fixture answer")
                    request = self.requests[-1]
                    payload = json.loads(request.body)
                    self.assertEqual(request.url, f"https://{host}/v1/chat/completions")
                    self.assertEqual(request.method, "POST")
                    self.assertEqual(request.headers["Authorization"], "Bearer fixture-minimax")
                    self.assertEqual(payload["model"], model or "MiniMax-M3.1-Flash")
                    self.assertEqual(payload["messages"], [{"role": "user", "content": "saved conversation"}])
                    self.assertEqual(payload["temperature"], 0.01)
                    self.assertEqual(payload["stop"], ["Observation:"])
                    for field in ("thinking", "reasoning_effort", "reasoning_split", "service_tier", "openai_api_base", "api_key", "api_base"):
                        self.assertNotIn(field, payload)
                    self.assertEqual(openai.api_key, "fixture-existing")
                    self.assertEqual(openai.api_base, "https://api.openai.com/v1")

    def test_existing_default_client_is_not_redirected(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-default"}):
            default = get_llm(provider="openai", temperature=0)
        get_llm()
        self.assertEqual(default("existing provider"), "fixture completion")
        request = self.requests[-1]
        self.assertEqual(request.url, "https://api.openai.com/v1/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer fixture-default")
        self.assertEqual(json.loads(request.body)["temperature"], 0)

    def test_prefix_multimodal_and_full_tool_history_are_preserved(self):
        history = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                {"type": "video_url", "video_url": {"url": "mm_file://fixture-video"}},
            ]},
            {"role": "assistant", "content": "<think>prior reasoning</think>",
             "reasoning_content": "preserve reasoning",
             "tool_calls": [{"id": "fixture-call", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "fixture-call", "content": "saved result"},
        ]
        llm = get_llm()
        llm.prefix_messages = history
        llm("continue")
        self.assertEqual(json.loads(self.requests[-1].body)["messages"], history + [{"role": "user", "content": "continue"}])

    def test_conversation_bot_factory_and_agent_memory_path(self):
        # Exercise the actual initialization expression without loading GPU tools.
        tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text())
        bot = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ConversationBot")
        init = next(node for node in bot.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        assignment = next(node for node in init.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute) and target.attr == "llm" for target in node.targets))
        llm = eval(compile(ast.Expression(assignment.value), "main.py", "eval"), {"get_llm": get_llm})
        agent = initialize_agent(
            [Tool(name="Fixture", func=lambda text: text, description="An offline fixture")],
            llm, agent="conversational-react-description",
            memory=ConversationBufferMemory(memory_key="chat_history", output_key="output"),
        )
        self.assertEqual(agent({"input": "remember fixture"})["output"], "fixture answer")
        self.assertEqual(agent({"input": "continue fixture"})["output"], "fixture answer")
        payload = json.loads(self.requests[-1].body)
        self.assertEqual(payload["model"], "MiniMax-M3.1-Flash")
        self.assertIn("remember fixture", payload["messages"][-1]["content"])
        self.assertIn("fixture answer", payload["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
