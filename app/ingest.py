"""
Filling the cache: network and disk in, SQLite out.

Three sources, one module, because they are one job -- "make the cache current" --
and the app runs them together at startup. Keeping them apart bought nothing but
three argparse blocks and three module docstrings saying the same thing.

    python -m app.ingest                      # status
    python -m app.ingest --source bts         # BTS T-100 (~132k rows, ~30s)
    python -m app.ingest --source airports    # OurAirports reference
    python -m app.ingest --source segments    # optional Tier-2 CSVs
    python -m app.ingest --source all

All three are idempotent, so re-running is always safe. The schema they write into
belongs to app/store.py; nothing here creates a table.

BTS
  T-100 Segment Summary By Origin Airport, one row per origin airport per month
  from Jan 2014. ~132k rows, small enough to cache whole. We pull it once and query
  locally thereafter: the demo is then instant and does not depend on Socrata being
  reachable, and ranking queries -- which touch every airport at once -- do not turn
  into thousands of HTTP calls.

AIRPORTS
  OurAirports reference data, for `iso_region` (which is what lets us answer
  "airports in New England" without hand-maintaining a list) and for the names and
  cities free-text resolution matches against. Note on its `type` column: it labels
  several GA fields (BED, BVY, OWD) "medium_airport", so we never use it as a
  commercial-service filter -- the traffic thresholds in config.py do that job.

SEGMENTS
  Optional. Enables exact haul-mix answers; see app/store.haul_mix_exact and the
  Tier-1/Tier-2 note in kpis.haul_mix_from_cohorts. Drop T-100 Segment CSV exports
  into cache/segments/ from BTS TranStats:
    https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIM
  selecting at minimum ORIGIN, DEST, DISTANCE, DEPARTURES_PERFORMED, SEATS,
  PASSENGERS, YEAR, MONTH. With no CSV present everything still works; the haul-mix
  tool answers from Tier 1 and says so.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from typing import Any, Iterable, Iterator

import httpx

from app import config, store


# Three coercions, deliberately not one. BTS wants a hard zero for a missing count
# so sums stay arithmetic; a missing coordinate must stay None rather than become a
# real point at 0N 0E; and TranStats CSV exports carry thousands separators that the
# Socrata JSON never does.

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


def _coord(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _num(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0

# ---------------------------------------------------------------------------
# BTS T-100 origin summary
# ---------------------------------------------------------------------------

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


def refresh_bts(verbose: bool = True) -> int:
    """Pull the whole table into SQLite. Idempotent -- safe to re-run."""
    con = store.connect()
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
# OurAirports reference data
# ---------------------------------------------------------------------------

def refresh_airports(verbose: bool = True) -> int:
    con = store.connect()

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
                _coord(row.get("latitude_deg")),
                _coord(row.get("longitude_deg")),
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


# ---------------------------------------------------------------------------
# Optional Tier-2 segment CSVs
# ---------------------------------------------------------------------------

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


def refresh_segments(verbose: bool = True) -> int:
    """Ingest every CSV in cache/segments/ into the `segments` table."""
    config.SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(config.SEGMENTS_DIR.glob("*.csv"))

    con = store.connect()
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_SOURCES = {
    "bts": refresh_bts,
    "airports": refresh_airports,
    "segments": refresh_segments,
}


def _status() -> int:
    con = store.connect()
    try:
        rows = con.execute("SELECT COUNT(*) FROM bts_monthly").fetchone()[0]
        if rows:
            lo, hi = con.execute("SELECT MIN(month), MAX(month) FROM bts_monthly").fetchone()
            n = con.execute("SELECT COUNT(DISTINCT airport) FROM bts_monthly").fetchone()[0]
            print(f"bts        {rows:,} rows, {lo} .. {hi}, {n:,} airports")
        else:
            print("bts        EMPTY   -- run: python -m app.ingest --source bts")

        airports = con.execute("SELECT COUNT(*) FROM airports").fetchone()[0]
        print(
            f"airports   {airports:,} US airports with IATA codes" if airports
            else "airports   EMPTY   -- run: python -m app.ingest --source airports"
        )

        seg = con.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
        if seg:
            origins = con.execute("SELECT COUNT(DISTINCT origin) FROM segments").fetchone()[0]
            print(f"segments   Tier 2 ENABLED: {seg:,} route-months, {origins:,} origins")
        else:
            print(
                f"segments   Tier 2 disabled (haul mix answers from the cohort "
                f"estimate). To enable, put T-100 Segment CSVs in "
                f"{config.SEGMENTS_DIR} and run: python -m app.ingest --source segments"
            )
        return 0 if rows and airports else 1
    finally:
        con.close()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingest",
        description="Populate the local cache. Idempotent; safe to re-run.",
    )
    parser.add_argument(
        "--source",
        choices=[*_SOURCES, "all"],
        help="which source to refresh. Omit to print cache status.",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.source is None:
        return _status()

    names = list(_SOURCES) if args.source == "all" else [args.source]
    for name in names:
        _SOURCES[name](verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
