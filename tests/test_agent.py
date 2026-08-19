"""
Agent loop mechanics, exercised against a fake Anthropic client.

No API key and no network. What is under test is the loop itself: tool dispatch,
message accumulation, stop-reason handling and the cache-stability of the prompt
prefix. Those are the parts that break silently and are tedious to debug live.
"""

from __future__ import annotations

import json
import types
from typing import Any

import pytest

from app import config
from app.agent import Agent
from app.data import bts

pytestmark = pytest.mark.skipif(
    not bts.is_populated(),
    reason="local cache not populated; run python -m app.data.bts --refresh",
)


# ---------------------------------------------------------------------------
# Fake SDK surface
# ---------------------------------------------------------------------------

def text_block(text: str) -> Any:
    return types.SimpleNamespace(type="text", text=text)


def tool_block(name: str, arguments: dict[str, Any], block_id: str = "toolu_1") -> Any:
    return types.SimpleNamespace(type="tool_use", name=name, input=arguments, id=block_id)


class FakeMessage:
    def __init__(self, content: list[Any], stop_reason: str, stop_details: Any = None) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.stop_details = stop_details
        self.usage = types.SimpleNamespace(
            input_tokens=100, output_tokens=50,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )


class FakeStream:
    """Replays a scripted turn as the SDK's stream events, then the final message."""

    def __init__(self, message: FakeMessage) -> None:
        self.message = message

    async def __aenter__(self) -> "FakeStream":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def __aiter__(self):
        for block in self.message.content:
            if block.type == "text":
                yield types.SimpleNamespace(
                    type="content_block_start",
                    content_block=types.SimpleNamespace(type="text"),
                )
                yield types.SimpleNamespace(
                    type="content_block_delta",
                    delta=types.SimpleNamespace(type="text_delta", text=block.text),
                )
            elif block.type == "tool_use":
                yield types.SimpleNamespace(
                    type="content_block_start",
                    content_block=types.SimpleNamespace(type="tool_use", name=block.name),
                )

    async def get_final_message(self) -> FakeMessage:
        return self.message


class FakeClient:
    """Returns scripted turns in order and records the request params it was given."""

    def __init__(self, turns: list[FakeMessage]) -> None:
        self.turns = list(turns)
        self.calls: list[dict[str, Any]] = []
        self.messages = types.SimpleNamespace(stream=self._stream)

    def _stream(self, **params: Any) -> FakeStream:
        # Deep-copy the message list: the agent mutates it between turns, and we
        # want a snapshot of what each request actually carried.
        self.calls.append({**params, "messages": json.loads(json.dumps(
            params["messages"], default=str))})
        if not self.turns:
            raise AssertionError("agent requested more turns than the script provides")
        return FakeStream(self.turns.pop(0))


async def collect(agent: Agent, message: str, con) -> list[dict[str, Any]]:
    return [event async for event in agent.run(message, con)]


@pytest.fixture
def con():
    connection = bts.connect()
    yield connection
    connection.close()


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_plain_answer_streams_text_then_done(con):
    agent = Agent(FakeClient([FakeMessage([text_block("Hello there.")], "end_turn")]))
    events = await collect(agent, "hi", con)

    assert [e["type"] for e in events] == ["text", "done"]
    assert events[0]["text"] == "Hello there."
    assert events[-1]["usage"]["input_tokens"] == 100


@pytest.mark.asyncio
async def test_tool_call_is_dispatched_and_fed_back(con):
    client = FakeClient([
        FakeMessage([tool_block("airport_profile", {"airport": "BOS"})], "tool_use"),
        FakeMessage([text_block("Boston scores moderately.")], "end_turn"),
    ])
    agent = Agent(client)
    events = await collect(agent, "tell me about Boston", con)

    kinds = [e["type"] for e in events]
    assert "tool_start" in kinds and "tool_result" in kinds and kinds[-1] == "done"

    result_event = next(e for e in events if e["type"] == "tool_result")
    assert result_event["name"] == "airport_profile"
    assert result_event["is_error"] is False
    assert result_event["result"]["airport"] == "BOS"

    # Second request must carry: user turn, assistant tool_use, user tool_result.
    second = client.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[2]["content"][0]["type"] == "tool_result"


@pytest.mark.asyncio
async def test_parallel_tool_calls_return_in_a_single_user_message(con):
    """Splitting results across messages trains the model out of parallel calls."""
    client = FakeClient([
        FakeMessage(
            [
                tool_block("airport_profile", {"airport": "BOS"}, "toolu_a"),
                tool_block("airport_profile", {"airport": "SFO"}, "toolu_b"),
            ],
            "tool_use",
        ),
        FakeMessage([text_block("Compared.")], "end_turn"),
    ])
    agent = Agent(client)
    events = await collect(agent, "compare", con)

    assert sum(1 for e in events if e["type"] == "tool_result") == 2

    tool_turn = client.calls[1]["messages"][-1]
    assert tool_turn["role"] == "user"
    assert len(tool_turn["content"]) == 2
    assert {b["tool_use_id"] for b in tool_turn["content"]} == {"toolu_a", "toolu_b"}


@pytest.mark.asyncio
async def test_failing_tool_is_reported_without_breaking_the_loop(con):
    client = FakeClient([
        FakeMessage([tool_block("airport_profile", {"airport": "Nowheresville"})], "tool_use"),
        FakeMessage([text_block("I could not resolve that airport.")], "end_turn"),
    ])
    agent = Agent(client)
    events = await collect(agent, "tell me about Nowheresville", con)

    result_event = next(e for e in events if e["type"] == "tool_result")
    assert result_event["is_error"] is True
    assert events[-1]["type"] == "done"

    # The error must reach the model flagged as such, so it can explain it.
    tool_turn = client.calls[1]["messages"][-1]
    assert tool_turn["content"][0].get("is_error") is True


# ---------------------------------------------------------------------------
# Stop reasons
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refusal_is_reported_and_content_is_not_consumed(con):
    """On a refusal the content array may be empty, so stop_reason is checked first."""
    client = FakeClient([
        FakeMessage([], "refusal", types.SimpleNamespace(category="cyber", explanation="no"))
    ])
    agent = Agent(client)
    events = await collect(agent, "something disallowed", con)

    assert events[-1]["type"] == "error"
    assert "declined" in events[-1]["message"]
    assert "cyber" in events[-1]["message"]
    # A refused turn must not be appended as if it were a real assistant message.
    assert all(m["role"] != "assistant" for m in agent.messages)


@pytest.mark.asyncio
async def test_max_tokens_is_surfaced_as_a_truncation_error(con):
    agent = Agent(FakeClient([FakeMessage([text_block("Partial...")], "max_tokens")]))
    events = await collect(agent, "long question", con)
    assert events[-1]["type"] == "error"
    assert "cut off" in events[-1]["message"]


@pytest.mark.asyncio
async def test_pause_turn_resumes_rather_than_ending(con):
    client = FakeClient([
        FakeMessage([text_block("working")], "pause_turn"),
        FakeMessage([text_block("done")], "end_turn"),
    ])
    agent = Agent(client)
    events = await collect(agent, "go", con)
    assert len(client.calls) == 2
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_runaway_tool_loop_is_capped(con):
    client = FakeClient([
        FakeMessage([tool_block("airport_profile", {"airport": "BOS"})], "tool_use")
        for _ in range(config.MAX_TOOL_ITERATIONS + 2)
    ])
    agent = Agent(client)
    events = await collect(agent, "loop forever", con)

    assert events[-1]["type"] == "error"
    assert "tool rounds" in events[-1]["message"]
    assert len(client.calls) == config.MAX_TOOL_ITERATIONS


@pytest.mark.asyncio
async def test_api_exception_becomes_an_error_event(con):
    class Boom(FakeClient):
        def _stream(self, **params: Any):
            raise RuntimeError("connection reset")

    agent = Agent(Boom([]))
    events = await collect(agent, "hi", con)
    assert events[-1]["type"] == "error"
    assert "connection reset" in events[-1]["message"]


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_request_uses_the_configured_model_and_no_removed_parameters(con):
    agent = Agent(FakeClient([FakeMessage([text_block("ok")], "end_turn")]))
    await collect(agent, "hi", con)
    params = agent.client.calls[0]

    assert params["model"] == config.MODEL == "claude-sonnet-5"
    assert params["thinking"]["type"] == "adaptive"
    assert params["output_config"]["effort"] == config.EFFORT
    # These are rejected with a 400 on Sonnet 5.
    for removed in ("temperature", "top_p", "top_k"):
        assert removed not in params


@pytest.mark.asyncio
async def test_system_prompt_is_cached_and_byte_stable_across_turns(con):
    """A single changing byte in the prefix would re-bill it on every turn."""
    client = FakeClient([
        FakeMessage([tool_block("airport_profile", {"airport": "BOS"})], "tool_use"),
        FakeMessage([text_block("done")], "end_turn"),
    ])
    agent = Agent(client)
    await collect(agent, "tell me about Boston", con)

    first, second = client.calls[0], client.calls[1]
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["system"][0]["text"] == second["system"][0]["text"]
    assert first["tools"] == second["tools"]


@pytest.mark.asyncio
async def test_every_tool_schema_is_offered_to_the_model(con):
    agent = Agent(FakeClient([FakeMessage([text_block("ok")], "end_turn")]))
    await collect(agent, "hi", con)
    offered = {t["name"] for t in agent.client.calls[0]["tools"]}
    assert offered == {
        "rank_expansion_candidates", "compare_airports",
        "airport_profile", "haul_mix", "unmet_demand",
    }


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_history_accumulates_across_turns(con):
    client = FakeClient([
        FakeMessage([text_block("First.")], "end_turn"),
        FakeMessage([text_block("Second.")], "end_turn"),
    ])
    agent = Agent(client)
    await collect(agent, "one", con)
    await collect(agent, "two", con)

    roles = [m["role"] for m in agent.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    # The follow-up request must carry the earlier turns.
    assert len(client.calls[1]["messages"]) == 3


@pytest.mark.asyncio
async def test_reset_clears_history(con):
    agent = Agent(FakeClient([FakeMessage([text_block("hi")], "end_turn")]))
    await collect(agent, "one", con)
    assert agent.messages
    agent.reset()
    assert agent.messages == []
