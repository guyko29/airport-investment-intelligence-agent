"""
FAA National Airspace System status: live ground stops and delay programs.

The BTS analysis is structural and runs about two months behind. This adds a live
read on whether an airport is congested *right now*, which is useful corroboration
for congestion questions -- a high modelled constraint index reads differently when
the airport also happens to be under a ground delay program this afternoon.

Strictly supplementary. It is never scored, and every caller degrades gracefully
when the FAA endpoint is unreachable: an investment thesis should not depend on
this afternoon's weather.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from app import config

# FAA delay-type names that actually indicate congestion (traffic flow management),
# as opposed to administrative closures.
_CONGESTION_PROGRAMS = {
    "Ground Stop Programs",
    "Ground Delay Programs",
    "Arrival/Departure Delay",
    "Airspace Flow Programs",
}

# The endpoint is polled per question at most; a short TTL keeps a burst of
# follow-up questions from hammering it.
_CACHE_TTL_SECONDS = 120.0
_cache: dict[str, Any] = {"fetched_at": 0.0, "payload": None}


def _parse(xml_text: str) -> dict[str, Any]:
    """Parse the FAA status XML into per-airport disruption records."""
    root = ET.fromstring(xml_text)
    updated = (root.findtext("Update_Time") or "").strip()
    by_airport: dict[str, list[dict[str, str]]] = {}

    for delay_type in root.findall("Delay_type"):
        kind = (delay_type.findtext("Name") or "").strip()
        # Each delay type wraps its entries in a differently-named list element
        # (Ground_Stop_List, Ground_Delay_List, Arrival_Departure_Delay_List...),
        # so walk every descendant rather than hard-coding the container names.
        for program in delay_type.iter():
            arpt = (program.findtext("ARPT") or "").strip().upper() if len(program) else ""
            if not arpt:
                continue
            entry = {
                "type": kind,
                "reason": (program.findtext("Reason") or "").strip(),
                # "Airport Closures" entries are administrative NOTAMs -- e.g. LAX
                # closed to non-scheduled transient GA without prior permission --
                # which say nothing about congestion. Only the flow-control
                # programs do, so tag them and let callers filter.
                "congestion_related": kind in _CONGESTION_PROGRAMS,
            }
            for field, key in (
                ("End_Time", "end_time"),
                ("Avg", "average_delay"),
                ("Max", "max_delay"),
                ("Trend", "trend"),
            ):
                value = (program.findtext(field) or "").strip()
                if value:
                    entry[key] = value
            by_airport.setdefault(arpt, []).append(entry)

    return {"updated": updated, "by_airport": by_airport}


def snapshot(force: bool = False) -> dict[str, Any]:
    """Current NAS status, cached briefly. Never raises."""
    now = time.monotonic()
    if not force and _cache["payload"] is not None and now - _cache["fetched_at"] < _CACHE_TTL_SECONDS:
        return _cache["payload"]

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(config.FAA_NAS_STATUS_URL)
            resp.raise_for_status()
        payload = _parse(resp.text)
        payload["available"] = True
    except Exception as exc:  # noqa: BLE001 -- supplementary data, never fatal
        payload = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "by_airport": {},
        }

    _cache["payload"] = payload
    _cache["fetched_at"] = now
    return payload


def disruptions_for(codes: list[str]) -> dict[str, Any]:
    """Live disruption records for the given airports.

    Absence of a record means no active FAA program -- which is the normal state
    for most airports most of the time, not missing data.
    """
    snap = snapshot()
    if not snap.get("available"):
        return {
            "available": False,
            "note": "FAA live status unavailable; structural analysis is unaffected.",
        }

    congestion: dict[str, list[dict[str, str]]] = {}
    advisories: dict[str, list[dict[str, str]]] = {}
    for code in codes:
        entries = snap["by_airport"].get(code.upper(), [])
        hits = [e for e in entries if e.get("congestion_related")]
        other = [e for e in entries if not e.get("congestion_related")]
        if hits:
            congestion[code.upper()] = hits
        if other:
            advisories[code.upper()] = other

    return {
        "available": True,
        "as_of": snap.get("updated"),
        "airports_under_flow_control": congestion,
        "airports_clear": [
            c.upper() for c in codes if c.upper() not in congestion
        ],
        "other_advisories": advisories,
        "note": (
            "Live FAA traffic-flow programs (ground stops, ground delays). "
            "Supplementary context only -- not part of any score, and a snapshot of "
            "today's weather rather than evidence of structural capacity. "
            "'other_advisories' are administrative NOTAMs such as general-aviation "
            "access restrictions, which do not indicate congestion."
        ),
    }
