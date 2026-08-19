"""
OurAirports reference data: ingest plus airport-name resolution.

Two jobs:

  1. Region filtering. OurAirports `iso_region` ("US-MA") is what lets us answer
     "airports in New England" without hand-maintaining an airport list.
  2. Free-text resolution. The exam asks about "LA and Santa Ana", not "LAX and
     SNA". Resolution returns the match *plus* any alternatives, so the agent can
     say which airport it used rather than silently guessing.

Run `python -m app.data.airports --refresh` to (re)populate.

Note on `type`: OurAirports labels several GA fields (BED, BVY, OWD) as
"medium_airport", so we never use it as a commercial-service filter. Traffic
thresholds in config.py do that job instead.
"""

from __future__ import annotations

import argparse
import csv
import io
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable

import httpx

from app import config
from app.data.bts import connect

_SCHEMA = """
CREATE TABLE IF NOT EXISTS airports (
    iata        TEXT PRIMARY KEY,
    icao        TEXT,
    name        TEXT,
    municipality TEXT,
    iso_region  TEXT,          -- 'US-MA'
    state       TEXT,          -- 'MA'
    country     TEXT,
    latitude    REAL,
    longitude   REAL,
    type        TEXT
);
CREATE INDEX IF NOT EXISTS idx_airports_state ON airports(state);
CREATE INDEX IF NOT EXISTS idx_airports_muni  ON airports(municipality);
"""


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def refresh(verbose: bool = True) -> int:
    con = connect()
    con.executescript(_SCHEMA)

    with httpx.Client(timeout=config.HTTP_TIMEOUT_SECONDS, follow_redirects=True) as client:
        resp = client.get(config.OURAIRPORTS_URL)
        resp.raise_for_status()
        text = resp.text

    records = []
    for row in csv.DictReader(io.StringIO(text)):
        iata = (row.get("iata_code") or "").strip().upper()
        if not iata or len(iata) != 3:
            continue
        if (row.get("iso_country") or "").strip().upper() != "US":
            continue
        iso_region = (row.get("iso_region") or "").strip().upper()
        state = iso_region.split("-")[-1] if "-" in iso_region else None
        records.append(
            (
                iata,
                (row.get("icao_code") or "").strip().upper() or None,
                (row.get("name") or "").strip() or None,
                (row.get("municipality") or "").strip() or None,
                iso_region or None,
                state,
                "US",
                _to_float(row.get("latitude_deg")),
                _to_float(row.get("longitude_deg")),
                (row.get("type") or "").strip() or None,
            )
        )

    con.executemany(
        """INSERT OR REPLACE INTO airports
           (iata, icao, name, municipality, iso_region, state, country,
            latitude, longitude, type)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        records,
    )
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM airports").fetchone()[0]
    con.close()
    if verbose:
        print(f"Airport reference ready: {total:,} US airports with IATA codes")
    return total


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_populated() -> bool:
    """Mirror of `bts.is_populated`: the reference table exists and has rows."""
    if not config.DB_PATH.exists():
        return False
    try:
        con = connect()
        con.executescript(_SCHEMA)
        n = con.execute("SELECT COUNT(*) FROM airports").fetchone()[0]
        con.close()
        return n > 0
    except sqlite3.Error:
        return False


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

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


def _lookup_code(con: sqlite3.Connection, code: str) -> sqlite3.Row | None:
    return con.execute(
        "SELECT iata, name, municipality, state FROM airports WHERE iata = ?",
        (code.upper(),),
    ).fetchone()


def _describe(con: sqlite3.Connection, code: str) -> dict[str, str]:
    row = _lookup_code(con, code)
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
        row = _lookup_code(con, raw)
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
        row = _lookup_code(con, primary)
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
    """Among candidate airports, prefer the one with the most recent BTS traffic.

    Traffic beats OurAirports' `type` field here: it is current, and it correctly
    ranks a busy commercial field above a co-located GA strip.
    """
    if len(rows) == 1:
        return rows[0]
    codes = [r["iata"] for r in rows]
    placeholders = ",".join("?" * len(codes))
    ranked = con.execute(
        f"""SELECT airport, SUM(total_passengers) AS pax
            FROM bts_monthly
            WHERE airport IN ({placeholders})
            GROUP BY airport ORDER BY pax DESC LIMIT 1""",
        codes,
    ).fetchone()
    if ranked is not None:
        for r in rows:
            if r["iata"] == ranked["airport"]:
                return r
    return rows[0]


# ---------------------------------------------------------------------------
# Region filtering
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


def airports_in_states(con: sqlite3.Connection, states: Iterable[str]) -> list[str]:
    codes = [s.strip().upper() for s in states if s and s.strip()]
    if not codes:
        return []
    placeholders = ",".join("?" * len(codes))
    rows = con.execute(
        f"SELECT iata FROM airports WHERE state IN ({placeholders}) ORDER BY iata",
        codes,
    ).fetchall()
    return [r["iata"] for r in rows]


def state_of(con: sqlite3.Connection, code: str) -> str | None:
    row = con.execute("SELECT state FROM airports WHERE iata = ?", (code.upper(),)).fetchone()
    return row["state"] if row else None


def known_region_names() -> list[str]:
    return sorted(config.REGIONS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage the airport reference cache.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--resolve", metavar="QUERY", help="test resolution")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.refresh:
        refresh()
        return 0

    con = connect()
    if args.resolve:
        import json
        print(json.dumps(resolve(con, args.resolve).to_dict(), indent=2))
        con.close()
        return 0

    total = con.execute("SELECT COUNT(*) FROM airports").fetchone()[0]
    print(f"{total:,} US airports cached" if total else
          "Airport cache empty. Run: python -m app.data.airports --refresh")
    con.close()
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
