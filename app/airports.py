"""
Turning what a person typed into an airport.

Users ask about "LA and Santa Ana", not "LAX and SNA". This module is the whole
translation layer between the two, and it is deliberately separate from the SQL it
runs (app/store.py) and from the ingest that filled those tables (app/ingest.py):
resolution is a judgement problem, not a data-access one. It decides what "Boston
airport" means, which of several fields serving one city to pick, and -- the part
that matters most -- when to admit the question was ambiguous.

The rule throughout: never resolve ambiguity silently. Every result carries the
alternatives that were considered, so the agent can say "I used LAX; BUR, LGB, ONT
and SNA also serve the metro" instead of quietly answering about one airport as
though it were the only one.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from app import config, store


@dataclass
class Resolved:
    """The outcome of resolving one free-text airport reference."""

    query: str
    code: str | None = None
    name: str | None = None
    city: str | None = None
    state: str | None = None
    how: str = "unresolved"          # iata | metro_alias | city | name | unresolved
    alternatives: list[dict[str, str]] = field(default_factory=list)
    note: str | None = None

    @property
    def ok(self) -> bool:
        return self.code is not None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "query": self.query,
            "code": self.code,
            "name": self.name,
            "city": self.city,
            "state": self.state,
            "matched_by": self.how,
        }
        if self.alternatives:
            out["alternatives"] = self.alternatives
        if self.note:
            out["note"] = self.note
        return out

def _describe(con: sqlite3.Connection, code: str) -> dict[str, str]:
    row = store.lookup_airport(con, code)
    if row is None:
        return {"code": code, "name": code}
    return {
        "code": row["iata"],
        "name": row["name"] or row["iata"],
        "city": row["municipality"] or "",
    }

def resolve(con: sqlite3.Connection, query: str) -> Resolved:
    """Resolve a free-text airport reference to a single IATA code.

    Resolution order: exact IATA code, then metro alias, then city name, then
    airport name substring. Ambiguity is reported via `alternatives` rather than
    resolved silently.
    """
    raw = (query or "").strip()
    if not raw:
        return Resolved(query=query, note="empty query")

    cleaned = raw.lower().strip(" .,?!")
    for suffix in (" international airport", " airport", " intl"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()

    # 1. Exact IATA code.
    if len(raw) == 3 and raw.isalpha():
        row = store.lookup_airport(con, raw)
        if row is not None:
            return Resolved(
                query=query,
                code=row["iata"],
                name=row["name"],
                city=row["municipality"],
                state=row["state"],
                how="iata",
            )

    # 2. Metro alias -- the colloquial names the exam questions actually use.
    if cleaned in config.METRO_ALIASES:
        primary, others = config.METRO_ALIASES[cleaned]
        row = store.lookup_airport(con, primary)
        alternatives = [_describe(con, c) for c in others]
        note = None
        if alternatives:
            alt_codes = ", ".join(a["code"] for a in alternatives)
            note = (
                f"'{raw}' is a metro area with multiple commercial airports; used "
                f"{primary} as the primary. Others in the metro: {alt_codes}."
            )
        return Resolved(
            query=query,
            code=primary,
            name=row["name"] if row else primary,
            city=row["municipality"] if row else None,
            state=row["state"] if row else None,
            how="metro_alias",
            alternatives=alternatives,
            note=note,
        )

    # 3. City name.
    rows = con.execute(
        """SELECT iata, name, municipality, state FROM airports
           WHERE LOWER(municipality) = ? ORDER BY iata""",
        (cleaned,),
    ).fetchall()
    if rows:
        chosen = _pick_busiest(con, rows)
        others = [r for r in rows if r["iata"] != chosen["iata"]]
        return Resolved(
            query=query,
            code=chosen["iata"],
            name=chosen["name"],
            city=chosen["municipality"],
            state=chosen["state"],
            how="city",
            alternatives=[_describe(con, r["iata"]) for r in others],
            note=(
                f"Multiple airports serve {chosen['municipality']}; used the busiest "
                f"({chosen['iata']})."
            ) if others else None,
        )

    # 4. Airport name substring.
    rows = con.execute(
        """SELECT iata, name, municipality, state FROM airports
           WHERE LOWER(name) LIKE ? ORDER BY iata LIMIT 8""",
        (f"%{cleaned}%",),
    ).fetchall()
    if rows:
        chosen = _pick_busiest(con, rows)
        others = [r for r in rows if r["iata"] != chosen["iata"]]
        return Resolved(
            query=query,
            code=chosen["iata"],
            name=chosen["name"],
            city=chosen["municipality"],
            state=chosen["state"],
            how="name",
            alternatives=[_describe(con, r["iata"]) for r in others],
        )

    return Resolved(
        query=query,
        note=(
            f"Could not resolve '{raw}' to a US airport. Try a 3-letter IATA code "
            f"(e.g. BOS) or a city name."
        ),
    )

def _pick_busiest(con: sqlite3.Connection, rows: list[sqlite3.Row]) -> sqlite3.Row:
    """Among candidate airports, prefer the one with the most recent BTS traffic."""
    if len(rows) == 1:
        return rows[0]
    winner = store.busiest_code(con, [r["iata"] for r in rows])
    for r in rows:
        if r["iata"] == winner:
            return r
    return rows[0]


# ---------------------------------------------------------------------------
# Region names
#
# Pure lookups against the table in config.py -- no database involved. The SQL
# counterpart, "which airports are in these states", lives in store.py.
# ---------------------------------------------------------------------------

def resolve_region(name: str) -> tuple[list[str], str] | None:
    """Named region -> (state codes, canonical label). None if unknown."""
    key = (name or "").strip().lower()
    for suffix in (" region", " us", " usa", " states"):
        if key.endswith(suffix):
            key = key[: -len(suffix)].strip()
    key = key.replace("-", " ").replace("_", " ")
    if key in config.REGIONS:
        return config.REGIONS[key], key.title()
    return None

def known_region_names() -> list[str]:
    return sorted(config.REGIONS)


# ---------------------------------------------------------------------------
# Debugging aid: python -m app.airports "Santa Ana"
#
# Resolution is the layer most likely to surprise, and reading its verdict on a
# phrasing is much faster than inferring it from an agent transcript.
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import json
    import sys

    if len(sys.argv) < 2:
        print('usage: python -m app.airports "<query>"')
        raise SystemExit(2)
    con = store.connect()
    try:
        print(json.dumps(resolve(con, " ".join(sys.argv[1:])).to_dict(), indent=2))
    finally:
        con.close()
