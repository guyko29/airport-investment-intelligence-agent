"""
FastAPI server: static chat UI plus a streaming chat endpoint.

Sessions are kept in memory, keyed by an id the browser generates. That is the
right trade for a single-analyst tool: no database, no auth, and conversation
history survives as long as the process. It does mean history is lost on restart
and does not span replicas -- a production deployment would move `_SESSIONS` to
Redis and put the API key behind a real auth layer.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app import config
from app.agent import Agent
from app import ingest, store

_SESSIONS: dict[str, Agent] = {}
_MAX_SESSIONS = 200

# Set when the startup ingest fails, so /health can report why and /chat can
# refuse with a readable message instead of querying an empty database.
_DATA_ERROR: str | None = None


async def _ensure_data() -> None:
    """Populate the cache if it is empty, recording failure rather than raising.

    On a host without a persistent disk the container is rebuilt on every
    spin-up, so an empty cache is the normal cold-start state, not an operator
    error -- building it here costs ~30s once per cold start and has the side
    benefit that the data is always the freshest BTS publishes. Refusing to boot
    would instead crash-loop the service.
    """
    global _DATA_ERROR
    try:
        if not store.bts_populated():
            await asyncio.to_thread(ingest.refresh_bts)
        if not store.airports_populated():
            await asyncio.to_thread(ingest.refresh_airports)
        _DATA_ERROR = None
    except Exception as exc:  # noqa: BLE001
        _DATA_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"Startup ingest failed: {_DATA_ERROR}", flush=True)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await _ensure_data()
    yield


app = FastAPI(title="Airport Investment Intelligence Agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


def _get_agent(session_id: str) -> Agent:
    agent = _SESSIONS.get(session_id)
    if agent is None:
        if len(_SESSIONS) >= _MAX_SESSIONS:
            # Drop the oldest session rather than growing without bound.
            _SESSIONS.pop(next(iter(_SESSIONS)), None)
        agent = Agent()
        _SESSIONS[session_id] = agent
    return agent


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


# ---------------------------------------------------------------------------
# Presentation
#
# The agent yields raw tool results. What the browser gets is decided here,
# because it is a property of this UI -- the transparency panel wants a handful
# of headline fields, not the entire signal tree down the SSE pipe. The model
# still receives the full payload.
# ---------------------------------------------------------------------------

def _for_browser(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("type") != "tool_result":
        return event
    return {
        **event,
        "result": _summarise_tool_result(event["name"], event["result"]),
    }


def _summarise_tool_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
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


@app.get("/health")
async def health() -> dict[str, Any]:
    if _DATA_ERROR is not None:
        return {
            "status": "degraded",
            "error": _DATA_ERROR,
            "detail": "Startup ingest failed. The upstream source was unreachable.",
            "model": config.MODEL,
            "scoring_version": config.SCORING_VERSION,
        }
    con = store.connect()
    try:
        latest = store.latest_month(con)
        rows = con.execute("SELECT COUNT(*) FROM bts_monthly").fetchone()[0]
        airports = con.execute("SELECT COUNT(*) FROM airports").fetchone()[0]
        tier2 = store.has_segment_data(con)
    finally:
        con.close()
    return {
        "status": "ok",
        "model": config.MODEL,
        "scoring_version": config.SCORING_VERSION,
        "api_key_configured": bool(
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        ),
        "data": {
            "bts_rows": rows,
            "latest_month": latest,
            "baseline_lookback_years": config.BASELINE_LOOKBACK_YEARS,
            "prepandemic_reference": config.REFERENCE_BASELINE_END,
            "airports_reference": airports,
            "haul_mix_tier_2_enabled": tier2,
        },
        "active_sessions": len(_SESSIONS),
    }


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    if _DATA_ERROR is not None:
        # A failed startup ingest is usually a transient upstream outage, so try
        # once more here rather than staying broken until the next deploy.
        await _ensure_data()
    if _DATA_ERROR is not None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Aviation data is unavailable: the upstream source could not be "
                f"reached at startup ({_DATA_ERROR}). Please retry shortly."
            ),
        )

    agent = _get_agent(req.session_id)

    async def event_stream() -> AsyncIterator[str]:
        # A fresh handle per request: SQLite connections are not shareable across
        # the threads the event loop may run this on.
        con = store.connect()
        try:
            async for event in agent.run(req.message, con):
                yield _sse(_for_browser(event))
                # Give the event loop a chance to flush each chunk to the client.
                await asyncio.sleep(0)
        except Exception as exc:  # noqa: BLE001
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            con.close()
            yield _sse({"type": "end"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # stop nginx buffering the stream
        },
    )


@app.post("/reset")
async def reset(req: ChatRequest) -> dict[str, str]:
    agent = _SESSIONS.get(req.session_id)
    if agent:
        agent.reset()
    return {"status": "reset", "session_id": req.session_id}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(config.WEB_DIR / "index.html")
