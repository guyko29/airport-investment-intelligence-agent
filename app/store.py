"""
Everything that reads the SQLite cache: schema, connections, queries.

One module owns the database. Before, three ingest modules each carried their own
CREATE TABLE and each ran it on the way past, so which tables existed depended on
which refresh had happened to run -- `airports.is_populated` had to re-run its own
DDL just to avoid "no such table". Here the schema is declared once and `connect`
applies all of it, so every connection sees every table.

The split from app/ingest.py is by direction, not by data source: this is the read
side (plus the schema the write side fills), and ingest.py is network-to-disk. They
are separated because they run at different moments and fail for different reasons
-- a query failing means a bug, an ingest failing means the upstream is down.

IMPORTANT field semantics: `total_distance_flight_sm` from BTS is the MEAN STAGE
LENGTH per departure in statute miles, not a total distance flown. Verified against
SFO 2026-04: 1,881 with 15,753 departures. Aggregating it across months therefore
requires a departure-weighted mean, not a sum.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Sequence

from app import config

# ---------------------------------------------------------------------------
# Schema
#
# Every table in one place. `connect` applies the whole thing, so a fresh cache
# and a half-ingested one present the same shape to callers.
# ---------------------------------------------------------------------------

_SCHEMA = """
-- BTS T-100 Segment Summary by origin airport, one row per airport-month.
CREATE TABLE IF NOT EXISTS bts_monthly (
    airport                  TEXT NOT NULL,
    month                    TEXT NOT NULL,          -- 'YYYY-MM'
    total_departures         INTEGER NOT NULL DEFAULT 0,
    total_passengers         INTEGER NOT NULL DEFAULT 0,
    total_seats              INTEGER NOT NULL DEFAULT 0,
    total_stage_miles        REAL    NOT NULL DEFAULT 0,   -- mean per departure
    domestic_departures      INTEGER NOT NULL DEFAULT 0,
    domestic_passengers      INTEGER NOT NULL DEFAULT 0,
    domestic_seats           INTEGER NOT NULL DEFAULT 0,
    domestic_stage_miles     REAL    NOT NULL DEFAULT 0,   -- mean per departure
    city_name                TEXT,
    airport_name             TEXT,
    PRIMARY KEY (airport, month)
);
CREATE INDEX IF NOT EXISTS idx_bts_month ON bts_monthly(month);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

-- OurAirports reference data: codes, location, iso_region.
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

-- Optional Tier-2 per-route distances. Empty unless segment CSVs were ingested.
CREATE TABLE IF NOT EXISTS segments (
    origin      TEXT NOT NULL,
    dest        TEXT NOT NULL,
    year        INTEGER NOT NULL,
    month       INTEGER NOT NULL,
    distance    REAL NOT NULL,
    departures  REAL NOT NULL,
    seats       REAL NOT NULL DEFAULT 0,
    passengers  REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_seg_origin ON segments(origin);
"""


# ---------------------------------------------------------------------------
# Month helpers ('YYYY-MM' strings sort lexicographically, which is why we use them)
#
# These live here rather than with the ingest they were written for: every window
# in the analysis is expressed in them, so they are vocabulary, not plumbing.
# ---------------------------------------------------------------------------

def month_index(month: str) -> int:
    """'2026-04' -> absolute month number, for arithmetic."""
    year, mon = month.split("-")
    return int(year) * 12 + (int(mon) - 1)


def index_to_month(idx: int) -> str:
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def shift_month(month: str, delta_months: int) -> str:
    return index_to_month(month_index(month) + delta_months)


def window(end_month: str, length: int = config.TTM_MONTHS) -> tuple[str, str]:
    """Inclusive [start, end] month window of `length` months ending at end_month."""
    return shift_month(end_month, -(length - 1)), end_month


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect() -> sqlite3.Connection:
    """Open the cache, creating any missing tables.

    A fresh handle per request: SQLite connections are not shareable across the
    threads the event loop may run a request on.
    """
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    return con


def _count(table: str) -> int:
    """Row count, or 0 if there is no database file yet."""
    if not config.DB_PATH.exists():
        return 0
    try:
        con = connect()
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        con.close()
        return n
    except sqlite3.Error:
        return 0


def bts_populated() -> bool:
    return _count("bts_monthly") > 0


def airports_populated() -> bool:
    return _count("airports") > 0


def has_segment_data(con: sqlite3.Connection) -> bool:
    """Whether Tier-2 exact haul-mix data was ingested. Usually False."""
    row = con.execute("SELECT COUNT(*) FROM segments").fetchone()
    return bool(row and row[0])


# ---------------------------------------------------------------------------
# Traffic queries
# ---------------------------------------------------------------------------

def latest_month(con: sqlite3.Connection) -> str:
    row = con.execute("SELECT MAX(month) AS m FROM bts_monthly").fetchone()
    if not row or not row["m"]:
        raise RuntimeError(
            "BTS cache is empty. Run: python -m app.ingest --source bts"
        )
    return row["m"]



# Departure-weighted mean stage length. Summing the monthly means would weight a
# 100-departure month the same as a 15,000-departure one.
_AGG_SELECT = """
    airport,
    MAX(airport_name)                            AS airport_name,
    MAX(city_name)                               AS city_name,
    COUNT(*)                                     AS months,
    SUM(total_departures)                        AS departures,
    SUM(total_passengers)                        AS passengers,
    SUM(total_seats)                             AS seats,
    SUM(total_departures * total_stage_miles)     AS dep_miles,
    SUM(domestic_departures)                     AS dom_departures,
    SUM(domestic_passengers)                     AS dom_passengers,
    SUM(domestic_seats)                          AS dom_seats,
    SUM(domestic_departures * domestic_stage_miles) AS dom_dep_miles
"""


def _shape(row: sqlite3.Row) -> dict[str, Any]:
    departures = row["departures"] or 0
    dom_departures = row["dom_departures"] or 0
    intl_departures = departures - dom_departures

    dep_miles = row["dep_miles"] or 0.0
    dom_dep_miles = row["dom_dep_miles"] or 0.0
    intl_dep_miles = dep_miles - dom_dep_miles

    return {
        "airport": row["airport"],
        "airport_name": row["airport_name"],
        "city_name": row["city_name"],
        "months": row["months"],
        "departures": departures,
        "passengers": row["passengers"] or 0,
        "seats": row["seats"] or 0,
        "domestic_departures": dom_departures,
        "domestic_passengers": row["dom_passengers"] or 0,
        "domestic_seats": row["dom_seats"] or 0,
        "international_departures": intl_departures,
        # Departure-weighted mean stage lengths. The international figure is derived
        # algebraically -- BTS publishes total and domestic, not international.
        "mean_stage_miles": (dep_miles / departures) if departures else 0.0,
        "mean_stage_miles_domestic": (dom_dep_miles / dom_departures) if dom_departures else 0.0,
        "mean_stage_miles_international": (
            (intl_dep_miles / intl_departures) if intl_departures > 0 else 0.0
        ),
    }


def aggregate(
    con: sqlite3.Connection,
    airport: str,
    end_month: str,
    months: int = config.TTM_MONTHS,
) -> dict[str, Any] | None:
    """Windowed aggregate for one airport, or None if it has no rows in the window."""
    start, end = window(end_month, months)
    row = con.execute(
        f"SELECT {_AGG_SELECT} FROM bts_monthly "
        "WHERE airport = ? AND month BETWEEN ? AND ? GROUP BY airport",
        (airport.upper(), start, end),
    ).fetchone()
    if row is None or not row["departures"]:
        return None
    return _shape(row)


def aggregate_many(
    con: sqlite3.Connection,
    end_month: str,
    months: int = config.TTM_MONTHS,
    airports: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Windowed aggregates keyed by airport code, for every airport (or a subset)."""
    start, end = window(end_month, months)
    sql = (
        f"SELECT {_AGG_SELECT} FROM bts_monthly WHERE month BETWEEN ? AND ?"
    )
    params: list[Any] = [start, end]
    if airports:
        codes = [a.upper() for a in airports]
        sql += f" AND airport IN ({','.join('?' * len(codes))})"
        params.extend(codes)
    sql += " GROUP BY airport"

    out: dict[str, dict[str, Any]] = {}
    for row in con.execute(sql, params):
        if not row["departures"]:
            continue
        out[row["airport"]] = _shape(row)
    return out


def monthly_series(
    con: sqlite3.Connection,
    airport: str,
    end_month: str,
    months: int = config.TTM_MONTHS,
) -> list[dict[str, Any]]:
    """Per-month rows for trend display."""
    start, end = window(end_month, months)
    rows = con.execute(
        """SELECT month, total_departures, total_passengers, total_seats
           FROM bts_monthly
           WHERE airport = ? AND month BETWEEN ? AND ?
           ORDER BY month""",
        (airport.upper(), start, end),
    ).fetchall()
    return [
        {
            "month": r["month"],
            "departures": r["total_departures"],
            "passengers": r["total_passengers"],
            "seats": r["total_seats"],
            "load_factor": round(
                100.0 * r["total_passengers"] / r["total_seats"], 1
            ) if r["total_seats"] else None,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Airport reference queries
# ---------------------------------------------------------------------------

def lookup_airport(con: sqlite3.Connection, code: str) -> sqlite3.Row | None:
    return con.execute(
        "SELECT iata, name, municipality, state FROM airports WHERE iata = ?",
        (code.upper(),),
    ).fetchone()


def busiest_code(con: sqlite3.Connection, codes: Sequence[str]) -> str | None:
    """Of these airports, the one with the most BTS passengers on record.

    Traffic beats OurAirports' `type` field for choosing between co-located fields:
    it is current, and it correctly ranks a busy commercial airport above a GA strip.
    """
    if not codes:
        return None
    placeholders = ",".join("?" * len(codes))
    row = con.execute(
        f"""SELECT airport, SUM(total_passengers) AS pax
            FROM bts_monthly
            WHERE airport IN ({placeholders})
            GROUP BY airport ORDER BY pax DESC LIMIT 1""",
        list(codes),
    ).fetchone()
    return row["airport"] if row else None



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

# ---------------------------------------------------------------------------
# Segment queries (Tier 2, present only if segment CSVs were ingested)
# ---------------------------------------------------------------------------


def haul_mix_exact(
    con: sqlite3.Connection, airport: str, threshold_miles: float
) -> dict[str, Any] | None:
    """True departure-weighted distance distribution for one origin airport.

    Returns None when no segment rows exist for the airport, so callers fall back
    to Tier 1 rather than reporting a spuriously precise zero.
    """
    if not has_segment_data(con):
        return None

    rows = con.execute(
        """SELECT dest, distance, SUM(departures) AS departures,
                  SUM(seats) AS seats, SUM(passengers) AS passengers
           FROM segments WHERE origin = ?
           GROUP BY dest, distance""",
        (airport.upper(),),
    ).fetchall()
    if not rows:
        return None

    total_dep = sum(r["departures"] for r in rows)
    if total_dep <= 0:
        return None

    long_dep = sum(r["departures"] for r in rows if r["distance"] >= threshold_miles)
    long_routes = sorted(
        ({"dest": r["dest"], "distance_miles": round(r["distance"]),
          "departures": round(r["departures"])}
         for r in rows if r["distance"] >= threshold_miles),
        key=lambda r: -r["departures"],
    )
    years = con.execute(
        "SELECT MIN(year), MAX(year) FROM segments WHERE origin = ?", (airport.upper(),)
    ).fetchone()

    return {
        "basis": "exact",
        "threshold_miles": threshold_miles,
        "long_haul_share_pct": round(100.0 * long_dep / total_dep, 1),
        "long_haul_departures": round(long_dep),
        "total_departures": round(total_dep),
        "route_count": len(rows),
        "long_haul_route_count": len(long_routes),
        "top_long_haul_routes": long_routes[:10],
        "mean_stage_miles": round(
            sum(r["distance"] * r["departures"] for r in rows) / total_dep, 1
        ),
        "data_years": [y for y in years if y] if years else [],
    }
