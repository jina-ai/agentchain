"""MiniMax transport for the repository's pinned LangChain version."""

import openai
from langchain.llms.openai import OpenAIChat
from pydantic import root_validator


class MiniMaxChat(OpenAIChat):
    """Use per-request authentication without changing SDK global settings."""

    @root_validator()
    def validate_environment(cls, values):
        # The inherited validator mutates openai.api_key. Keep existing clients
        # isolated: api_key and api_base are supplied through model_kwargs.
        values["client"] = openai.ChatCompletion
        return values
