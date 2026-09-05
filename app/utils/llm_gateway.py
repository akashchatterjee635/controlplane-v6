"""LLM Gateway — Centralized LLM invocation and cost tracking."""

import os
import time
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.utils.cost import update_cost_record


class LLMGateway:
    """Wrapper for all LLM calls to ensure uniform cost tracking and observability."""

    def __init__(self, cost_tracker: dict[str, Any], trace_id: str = ""):
        self.cost_tracker = cost_tracker
        self.trace_id = trace_id
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        self._client = ChatOpenAI(
            model=self.model,
            temperature=0,
        )

    def invoke(
        self,
        messages: list[BaseMessage],
        purpose: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Invoke the LLM and record telemetry.

        Returns:
            The string content of the response.
        """
        start_time = time.time()

        self._client.temperature = temperature
        if max_tokens is not None:
            self._client.max_tokens = max_tokens
        else:
            self._client.max_tokens = 500

        response = self._client.invoke(messages)
        content = response.content

        usage_metadata = response.response_metadata.get("token_usage", {})
        prompt_tokens = usage_metadata.get("prompt_tokens", 0)
        completion_tokens = usage_metadata.get("completion_tokens", 0)

        (time.time() - start_time) * 1000

        self.cost_tracker = update_cost_record(
            self.cost_tracker,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.model,
        )

        return content
