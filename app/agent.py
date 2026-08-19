"""
The agent loop: Claude with tool use, streamed.

WHERE AI IS AND IS NOT USED
---------------------------
The model does four things: understands the question, picks a tool, reads the
structured result, and writes the explanation. It does not compute anything. Every
number it can quote was produced by kpis.py and arrived in a tool result, which is
why the analysis is reproducible and unit-testable while the conversation stays
natural.

This module is the loop and nothing else. What the model is TOLD lives in
app/prompt.py; what it can DO lives in app/tools.py; how a tool result is
rendered for the browser lives in app/main.py. The loop yields raw tool
results and lets the transport decide what to show.

This uses a manual tool-use loop over `messages.stream()` rather than the SDK's
tool runner, because the UI needs both per-token text deltas AND interception of
each tool call to render the transparency panel. The runner surfaces one or the
other, not both.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic

from app import config, prompt, tools


class Agent:
    """One conversation. Holds message history; the DB connection is per-request."""

    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        self.client = client or AsyncAnthropic()
        self.messages: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.messages = []

    # -- request construction ------------------------------------------------

    def _request_params(self) -> dict[str, Any]:
        return {
            "model": config.MODEL,
            "max_tokens": config.MAX_TOKENS,
            # Adaptive thinking is on by default on Opus 5; set explicitly so the
            # intent is visible. `summarized` surfaces reasoning to the UI panel --
            # the default omits it and the panel would sit empty.
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": config.EFFORT},
            # Cache the tools + system prefix. Both come from app/prompt.py and
            # are byte-stable across turns, so every turn after the first reads
            # the prefix instead of paying for it.
            "system": [
                {
                    "type": "text",
                    "text": prompt.SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "tools": prompt.TOOL_SCHEMAS,
            "messages": self.messages,
        }

    # -- the loop ------------------------------------------------------------

    async def run(self, user_message: str, con: sqlite3.Connection) -> AsyncIterator[dict[str, Any]]:
        """Stream one turn, running tools as requested.

        Yields event dicts: {type: text|thinking|tool_start|tool_result|error|done}.
        """
        self.messages.append({"role": "user", "content": user_message})

        for iteration in range(config.MAX_TOOL_ITERATIONS):
            try:
                final = None
                async with self.client.messages.stream(**self._request_params()) as stream:
                    async for event in stream:
                        if event.type == "content_block_start":
                            if event.content_block.type == "thinking":
                                yield {"type": "thinking_start"}
                            elif event.content_block.type == "tool_use":
                                yield {
                                    "type": "tool_start",
                                    "name": event.content_block.name,
                                }
                        elif event.type == "content_block_delta":
                            if event.delta.type == "text_delta":
                                yield {"type": "text", "text": event.delta.text}
                            elif event.delta.type == "thinking_delta":
                                yield {"type": "thinking", "text": event.delta.thinking}
                    final = await stream.get_final_message()
            except Exception as exc:  # noqa: BLE001
                yield {
                    "type": "error",
                    "message": f"Model request failed: {type(exc).__name__}: {exc}",
                }
                return

            # Check the stop reason BEFORE reading content: on a refusal the
            # content array may be empty or partial.
            if final.stop_reason == "refusal":
                detail = getattr(final, "stop_details", None)
                category = getattr(detail, "category", None) if detail else None
                yield {
                    "type": "error",
                    "message": (
                        "The request was declined by safety classifiers"
                        + (f" (category: {category})" if category else "")
                        + ". Try rephrasing the question."
                    ),
                }
                return

            self.messages.append({"role": "assistant", "content": final.content})

            if final.stop_reason == "max_tokens":
                yield {
                    "type": "error",
                    "message": (
                        "Response hit the token limit and was cut off. Ask a narrower "
                        "question, or request fewer airports."
                    ),
                }
                return

            # A server-side tool paused the turn; re-send to let it continue.
            if final.stop_reason == "pause_turn":
                continue

            if final.stop_reason != "tool_use":
                yield {"type": "done", "usage": _usage(final)}
                return

            # Execute every requested tool and return all results in ONE user
            # message -- splitting them across messages trains the model out of
            # parallel tool calls.
            tool_results = []
            for block in final.content:
                if block.type != "tool_use":
                    continue
                result = tools.dispatch(con, block.name, dict(block.input or {}))
                yield {
                    "type": "tool_result",
                    "name": block.name,
                    "input": dict(block.input or {}),
                    "result": result,
                    "is_error": "error" in result,
                }
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                        **({"is_error": True} if "error" in result else {}),
                    }
                )

            if not tool_results:
                yield {"type": "done", "usage": _usage(final)}
                return

            self.messages.append({"role": "user", "content": tool_results})

        yield {
            "type": "error",
            "message": (
                f"Stopped after {config.MAX_TOOL_ITERATIONS} tool rounds without a "
                f"final answer. Try a narrower question."
            ),
        }


def _usage(message: Any) -> dict[str, Any]:
    u = getattr(message, "usage", None)
    if u is None:
        return {}
    return {
        "input_tokens": getattr(u, "input_tokens", None),
        "output_tokens": getattr(u, "output_tokens", None),
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None),
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None),
    }
