"""
Tier-2 (optional): exact per-flight distance distribution from T-100 Segment data.

WHY THIS EXISTS
---------------
The exam asks for "the percentage of long haul flights out of Anchorage". No free
API publishes per-flight distances. The BTS origin-summary table we use everywhere
else gives only a MEAN stage length per airport-month, and a mean cannot be turned
back into a distribution.

So the haul-mix tool answers in two tiers:

  Tier 1 (always available, live API): decompose the two cohorts BTS does publish
         -- domestic and international -- and report a range with the assumption
         stated. See kpis.haul_mix_from_cohorts.

  Tier 2 (this module, if segment data is present): the true per-route distance
         distribution, so the answer becomes an exact percentage.

To enable Tier 2, drop one or more T-100 Segment CSV exports into data/segments/
and run `python -m app.data.segments --refresh`. Get them from BTS TranStats:
  https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIM&QO_fu146_anzr=Nv4%20Pn44vr45
selecting at minimum: ORIGIN, DEST, DISTANCE, DEPARTURES_PERFORMED, SEATS,
PASSENGERS, YEAR, MONTH.

Everything degrades cleanly: with no CSV present the tool still answers via Tier 1
and says so.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from typing import Any, Iterable

from app import config
from app.data.bts import connect

_SCHEMA = """
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

# T-100 exports vary in header case and naming between the TranStats UI and the
# prezipped files, so match on a set of accepted aliases per logical column.
_ALIASES = {
    "origin": ("origin", "origin_airport", "origin_code"),
    "dest": ("dest", "destination", "dest_airport", "dest_code"),
    "distance": ("distance", "distance_miles", "stage_length"),
    "departures": ("departures_performed", "departures", "dep_performed"),
    "seats": ("seats",),
    "passengers": ("passengers", "pax"),
    "year": ("year",),
    "month": ("month",),
}


def _build_column_map(fieldnames: Iterable[str]) -> dict[str, str] | None:
    """Map logical column -> actual header, or None if required columns are absent."""
    lowered = {(f or "").strip().lower(): f for f in fieldnames}
    mapping: dict[str, str] = {}
    for logical, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                mapping[logical] = lowered[alias]
                break
    required = {"origin", "dest", "distance", "departures"}
    if not required.issubset(mapping):
        return None
    return mapping


def _num(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def refresh(verbose: bool = True) -> int:
    """Ingest every CSV in data/segments/ into the `segments` table."""
    config.SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(config.SEGMENTS_DIR.glob("*.csv"))

    con = connect()
    con.executescript(_SCHEMA)
    con.execute("DELETE FROM segments")

    if not csv_paths:
        con.commit()
        con.close()
        if verbose:
            print(
                f"No CSVs in {config.SEGMENTS_DIR}. Tier 2 stays disabled; the "
                f"haul-mix tool will answer from Tier 1 (cohort estimate)."
            )
        return 0

    total = 0
    for path in csv_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            colmap = _build_column_map(reader.fieldnames or [])
            if colmap is None:
                if verbose:
                    print(f"  skipped {path.name}: missing required T-100 Segment columns")
                continue

            batch = []
            for row in reader:
                origin = (row.get(colmap["origin"]) or "").strip().upper()
                dest = (row.get(colmap["dest"]) or "").strip().upper()
                departures = _num(row.get(colmap["departures"]))
                distance = _num(row.get(colmap["distance"]))
                # Scheduled-but-not-flown rows carry 0 departures; they would drag
                # the distribution toward routes that never operated.
                if not origin or not dest or departures <= 0 or distance <= 0:
                    continue
                batch.append(
                    (
                        origin,
                        dest,
                        int(_num(row.get(colmap.get("year", ""), 0))),
                        int(_num(row.get(colmap.get("month", ""), 0))),
                        distance,
                        departures,
                        _num(row.get(colmap.get("seats", ""), 0)),
                        _num(row.get(colmap.get("passengers", ""), 0)),
                    )
                )
                if len(batch) >= 20_000:
                    con.executemany(
                        "INSERT INTO segments VALUES (?,?,?,?,?,?,?,?)", batch
                    )
                    total += len(batch)
                    batch.clear()

            if batch:
                con.executemany("INSERT INTO segments VALUES (?,?,?,?,?,?,?,?)", batch)
                total += len(batch)
        con.commit()
        if verbose:
            print(f"  loaded {path.name}")

    con.close()
    if verbose:
        print(f"Segment cache ready: {total:,} route-months. Tier 2 enabled.")
    return total


def has_segment_data(con: sqlite3.Connection) -> bool:
    try:
        row = con.execute("SELECT COUNT(*) FROM segments").fetchone()
        return bool(row and row[0])
    except sqlite3.Error:
        return False


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


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest optional T-100 Segment CSVs to enable exact haul-mix answers."
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.refresh:
        refresh()
        return 0

    con = connect()
    enabled = has_segment_data(con)
    if enabled:
        n = con.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
        origins = con.execute("SELECT COUNT(DISTINCT origin) FROM segments").fetchone()[0]
        print(f"Tier 2 ENABLED: {n:,} route-months across {origins:,} origin airports")
    else:
        print(
            "Tier 2 disabled (no segment data). Haul-mix answers use the Tier 1 "
            "cohort estimate.\n"
            f"To enable: put T-100 Segment CSVs in {config.SEGMENTS_DIR} and run "
            "python -m app.data.segments --refresh"
        )
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
