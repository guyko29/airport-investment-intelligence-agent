"""
BTS T-100 Segment Summary By Origin Airport: ingest and query.

The full table is ~132k rows (one row per origin airport per month, Jan 2014 to
present), which is small enough to cache in its entirety. We pull it once into
SQLite and query locally thereafter. Two reasons that matters:

  1. The demo is instant and does not depend on Socrata being reachable.
  2. Ranking queries touch every airport at once; doing that over HTTP would be
     both slow and rude.

Run `python -m app.data.bts --refresh` to (re)populate the cache.

IMPORTANT field semantics: `total_distance_flight_sm` is the MEAN STAGE LENGTH per
departure in statute miles, not a total distance flown. Verified against SFO
2026-04: 1,881 with 15,753 departures. Aggregating it across months therefore
requires a departure-weighted mean, not a sum.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from typing import Any, Iterable, Iterator, Sequence

import httpx

from app import config

# Columns we pull from Socrata. Everything else in the table (freight, mail,
# payload, per-flight ratios we recompute ourselves) is not used by any signal.
_FIELDS = [
    "origin_airport_code",
    "reporting_month",
    "total_departures",
    "total_passengers",
    "total_seats",
    "total_distance_flight_sm",
    "domestic_departures",
    "domestic_passengers",
    "domestic_seats",
    "domestic_distance_flight",
    "origin_city_name",
    "origin_airport_name",
]

_SCHEMA = """
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
"""


# ---------------------------------------------------------------------------
# Month helpers ('YYYY-MM' strings sort lexicographically, which is why we use them)
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

def connect(read_only: bool = False) -> sqlite3.Connection:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    if not read_only:
        con.executescript(_SCHEMA)
    return con


def is_populated() -> bool:
    if not config.DB_PATH.exists():
        return False
    try:
        con = connect()
        n = con.execute("SELECT COUNT(*) FROM bts_monthly").fetchone()[0]
        con.close()
        return n > 0
    except sqlite3.Error:
        return False


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def _to_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fetch_pages(client: httpx.Client) -> Iterator[list[dict[str, Any]]]:
    """Page through the Socrata resource with a deterministic sort order.

    Socrata paging without an explicit $order is not guaranteed stable across
    requests, which would silently drop or duplicate rows.
    """
    offset = 0
    while True:
        params = {
            "$select": ",".join(_FIELDS),
            "$order": "origin_airport_code,reporting_month",
            "$limit": str(config.BTS_PAGE_SIZE),
            "$offset": str(offset),
        }
        resp = client.get(config.BTS_BASE_URL, params=params)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return
        yield rows
        if len(rows) < config.BTS_PAGE_SIZE:
            return
        offset += config.BTS_PAGE_SIZE


def _normalise(row: dict[str, Any]) -> tuple | None:
    airport = (row.get("origin_airport_code") or "").strip().upper()
    raw_month = row.get("reporting_month") or ""
    if not airport or len(raw_month) < 7:
        return None
    month = raw_month[:7]                      # '2026-04-01T00:00:00.000' -> '2026-04'
    return (
        airport,
        month,
        _to_int(row.get("total_departures")),
        _to_int(row.get("total_passengers")),
        _to_int(row.get("total_seats")),
        _to_float(row.get("total_distance_flight_sm")),
        _to_int(row.get("domestic_departures")),
        _to_int(row.get("domestic_passengers")),
        _to_int(row.get("domestic_seats")),
        _to_float(row.get("domestic_distance_flight")),
        (row.get("origin_city_name") or "").strip() or None,
        (row.get("origin_airport_name") or "").strip() or None,
    )


def refresh(verbose: bool = True) -> int:
    """Pull the whole table into SQLite. Idempotent -- safe to re-run."""
    con = connect()
    inserted = 0
    with httpx.Client(timeout=config.HTTP_TIMEOUT_SECONDS, follow_redirects=True) as client:
        for page in _fetch_pages(client):
            records = [r for r in (_normalise(row) for row in page) if r is not None]
            con.executemany(
                """INSERT OR REPLACE INTO bts_monthly
                   (airport, month, total_departures, total_passengers, total_seats,
                    total_stage_miles, domestic_departures, domestic_passengers,
                    domestic_seats, domestic_stage_miles, city_name, airport_name)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                records,
            )
            con.commit()
            inserted += len(records)
            if verbose:
                print(f"  ingested {inserted:,} rows...", flush=True)

    latest = con.execute("SELECT MAX(month) FROM bts_monthly").fetchone()[0]
    con.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('latest_month', ?)", (latest,)
    )
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM bts_monthly").fetchone()[0]
    con.close()
    if verbose:
        print(f"BTS cache ready: {total:,} rows, latest month {latest}")
    return total


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def latest_month(con: sqlite3.Connection) -> str:
    row = con.execute("SELECT MAX(month) AS m FROM bts_monthly").fetchone()
    if not row or not row["m"]:
        raise RuntimeError(
            "BTS cache is empty. Run: python -m app.data.bts --refresh"
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
# CLI
# ---------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage the BTS T-100 local cache.")
    parser.add_argument("--refresh", action="store_true", help="(re)populate the cache")
    parser.add_argument("--status", action="store_true", help="show cache status")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.refresh:
        refresh()
        return 0

    con = connect()
    total = con.execute("SELECT COUNT(*) FROM bts_monthly").fetchone()[0]
    if total == 0:
        print("BTS cache is empty. Run: python -m app.data.bts --refresh")
        con.close()
        return 1
    lo, hi = con.execute("SELECT MIN(month), MAX(month) FROM bts_monthly").fetchone()
    airports = con.execute("SELECT COUNT(DISTINCT airport) FROM bts_monthly").fetchone()[0]
    print(f"rows      {total:,}")
    print(f"months    {lo} .. {hi}")
    print(f"airports  {airports:,}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
