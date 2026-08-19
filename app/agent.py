"""
The agent loop: Claude with tool use, streamed.

WHERE AI IS AND IS NOT USED
---------------------------
The model does four things: understands the question, picks a tool, reads the
structured result, and writes the explanation. It does not compute anything. Every
number it can quote was produced by kpis.py and arrived in a tool result, which is
why the analysis is reproducible and unit-testable while the conversation stays
natural.

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

from app import config, tools
from app.data import bts

# ---------------------------------------------------------------------------
# System prompt
#
# Kept as a module-level constant with no interpolated timestamps or per-request
# values: it is the cached prefix, and a single changing byte would invalidate
# the cache on every turn.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an analyst for a firm that invests in modernising US airport \
infrastructure. Your job is to help identify airports where terminal renovation \
or expansion would be most profitable, on the thesis that the return comes from \
unlocking constrained flight and passenger capacity.

## The analytical frame

Load factor alone does not identify a constrained airport: it measures how full \
aircraft are, which is an airline scheduling decision. The sharper signal is \
UPGAUGING. When carriers want more capacity at an airport they add flights first, \
because that is cheaper and more flexible. When they instead fly bigger aircraft \
while flight counts stay flat, it is usually because they cannot add flights -- \
gates, slots or runway capacity are saturated. That is the condition where an \
expansion converts into revenue, and it is what the capacity-constraint signal \
measures.

## Tools

- `rank_expansion_candidates` -- ranked candidates nationally or by region/state.
- `compare_airports` -- two or more airports side by side, plus a congestion \
ranking and live FAA status.
- `airport_profile` -- full signal breakdown for one airport.
- `haul_mix` -- share of departures that are long haul.
- `unmet_demand` -- modelled unmet demand for one airport, decomposed.

Call a tool whenever a question needs a number. Never estimate, recall or \
calculate a figure yourself: you have no airport data outside these tools, and a \
number you produce without one is fabricated. Quote tool figures as returned \
rather than rounding or recombining them. If a tool returns an `error`, say what \
failed and what you would need instead.

## Reporting standards

Every tool result carries the raw inputs behind its numbers. Use them: explain \
which signal drove a conclusion, not just the conclusion.

State assumptions, uncertainty and scope wherever they bear on the answer:

- When a result carries `basis: "model_estimate"` or `basis: "cohort_estimate"`, \
say plainly that the figure is an estimate and what it rests on. Do not present \
an estimate as a measurement.
- When a range and a point estimate are both given, report both.
- When `data_quality` is `partial`, or `notes` is non-empty, surface it -- those \
notes exist because something about the airport makes the standard reading \
misleading.
- When a query resolves ambiguously (a metro with several airports), say which \
airport you used and what the alternatives were.
- The eligibility floor excludes small airports from rankings. If that could \
matter to the question, say so.

The analysis is structural and the underlying data runs about two months behind. \
Live FAA status, where included, reflects today's weather and traffic, not \
capacity -- never treat a ground stop as evidence of a structural constraint.

## Scope

You analyse airport infrastructure capacity. You do not give investment advice on \
securities, airline equities or financial instruments, and you are not a licensed \
financial adviser -- if asked, say so plainly and offer the capacity analysis you \
can do instead.

## Style

Lead with the answer, then the evidence. Keep responses focused and concise; \
prefer prose to nested bullets for short answers, and use a table only when \
comparing several airports across several signals. Deliver what was asked at the \
scope it was asked -- make routine judgement calls yourself rather than \
interrogating the user, but do not expand the question into an unrequested \
survey. When a follow-up is ambiguous, resolve it from the conversation so far \
rather than asking.\
"""


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
            # Cache the tools + system prefix. Both are byte-stable across turns,
            # so every turn after the first reads the prefix instead of paying for it.
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "tools": tools.TOOL_SCHEMAS,
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
                    "result": _summarise_for_ui(block.name, result),
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


def _summarise_for_ui(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Compact a tool result for the transparency panel.

    The model gets the full payload; the browser gets enough to show what the tool
    actually returned without shipping the entire signal tree over SSE.
    """
    if "error" in result:
        return {"error": result["error"]}

    if name == "rank_expansion_candidates":
        return {
            "scope": result.get("scope"),
            "eligible_count": result.get("eligible_count"),
            "top": [
                {
                    "rank": r["rank"],
                    "airport": r["airport"],
                    "score": r["score"],
                    "verdict": r["verdict"],
                }
                for r in result.get("ranking", [])[:8]
            ],
        }
    if name == "compare_airports":
        return {
            "resolved": [
                {"query": r["query"], "code": r["code"]} for r in result.get("resolved", [])
            ],
            "congestion_ranking": [
                {
                    "rank": c["congestion_rank"],
                    "airport": c["airport"],
                    "index": c["congestion_index"],
                    "load_factor_pct": c["load_factor_pct"],
                }
                for c in result.get("congestion_ranking", [])
            ],
        }
    if name == "airport_profile":
        return {
            "airport": result.get("airport"),
            "score": result.get("score"),
            "verdict": result.get("verdict"),
            "data_quality": result.get("data_quality"),
            "load_factor_pct": (result.get("facts") or {}).get("load_factor_pct"),
        }
    if name == "haul_mix":
        answer = result.get("answer", {})
        return {
            "airport": (result.get("resolved") or {}).get("code"),
            "basis": answer.get("basis"),
            "long_haul_share_pct": answer.get(
                "long_haul_share_pct", answer.get("long_haul_share_pct_estimate")
            ),
            "range": answer.get("long_haul_share_pct_range"),
            "threshold_miles": result.get("threshold_miles"),
        }
    if name == "unmet_demand":
        return {
            "airport": result.get("airport"),
            "unmet_index_pct": result.get("unmet_demand_index_pct_of_passengers"),
            "basis": result.get("basis"),
            "driver_count": len(result.get("why", [])),
        }
    return {"keys": sorted(result)[:12]}


def open_connection() -> sqlite3.Connection:
    """Per-request DB handle. SQLite connections are not shareable across threads."""
    return bts.connect()
