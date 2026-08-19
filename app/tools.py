"""
Tool layer: the five capabilities the agent can invoke.

Design rules, applied uniformly:

  * The model never does arithmetic. Every number in an answer is computed here
    (via kpis.py) and handed over with the raw inputs it came from, so any claim
    can be recomputed offline.
  * Every tool accepts free-text airport references and echoes what it resolved
    them to, including alternatives. "Compare LA and Santa Ana" is only honestly
    answerable if the agent can say which LA airport it used.
  * Every result carries `scoring_version` and, where the number is modelled
    rather than measured, an explicit `basis` field.
  * Tools never raise into the agent loop. A failure returns a structured `error`
    the model can read and explain.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

from app import config, kpis
from app.data import airports as airports_data
from app.data import bts, faa, segments


# ---------------------------------------------------------------------------
# Analysis context
# ---------------------------------------------------------------------------

class Analysis:
    """Holds the DB connection and the three comparison windows for one request."""

    def __init__(self, con: sqlite3.Connection) -> None:
        self.con = con
        self.current_month = bts.latest_month(con)
        # Scored baseline: rolling N years back from the current window.
        self.baseline_years = float(config.BASELINE_LOOKBACK_YEARS)
        self.baseline_month = bts.shift_month(
            self.current_month, -12 * config.BASELINE_LOOKBACK_YEARS
        )
        # Unscored pre-pandemic reference, fixed in time.
        self.reference_month = config.REFERENCE_BASELINE_END
        self.reference_years = (
            bts.month_index(self.current_month) - bts.month_index(self.reference_month)
        ) / 12.0

    def data_window(self) -> dict[str, Any]:
        start, end = bts.window(self.current_month)
        base_start, base_end = bts.window(self.baseline_month)
        return {
            "current_window": f"{start} .. {end}",
            "baseline_window": f"{base_start} .. {base_end}",
            "baseline_window_ends": self.baseline_month,
            "baseline_is": f"rolling {config.BASELINE_LOOKBACK_YEARS}-year lookback",
            "years_between": round(self.baseline_years, 2),
            "prepandemic_reference_window_ends": self.reference_month,
            "source": "BTS T-100 Segment Summary by Origin Airport (data.bts.gov, r495-tyji)",
        }

    def assess_one(self, code: str) -> kpis.Assessment | None:
        cur = bts.aggregate(self.con, code, self.current_month)
        if cur is None:
            return None
        base = bts.aggregate(self.con, code, self.baseline_month)
        ref = bts.aggregate(self.con, code, self.reference_month)
        return kpis.assess(
            cur, base, self.baseline_years, self.baseline_month,
            ref, self.reference_years, self.reference_month,
        )

    def assess_many(self, codes: list[str]) -> dict[str, kpis.Assessment]:
        cur = bts.aggregate_many(self.con, self.current_month, airports=codes)
        base = bts.aggregate_many(self.con, self.baseline_month, airports=codes)
        ref = bts.aggregate_many(self.con, self.reference_month, airports=codes)
        return {
            code: kpis.assess(
                agg, base.get(code), self.baseline_years, self.baseline_month,
                ref.get(code), self.reference_years, self.reference_month,
            )
            for code, agg in cur.items()
        }


def _resolve_all(con: sqlite3.Connection, queries: list[str]) -> tuple[list[str], list[dict], list[dict]]:
    """Resolve free-text airport references. Returns (codes, resolutions, failures)."""
    codes: list[str] = []
    resolutions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for q in queries:
        r = airports_data.resolve(con, q)
        resolutions.append(r.to_dict())
        if r.ok and r.code not in codes:
            codes.append(r.code)
        elif not r.ok:
            failures.append(r.to_dict())
    return codes, resolutions, failures


# ---------------------------------------------------------------------------
# Tool 1: rank expansion candidates
# ---------------------------------------------------------------------------

def rank_expansion_candidates(
    con: sqlite3.Connection,
    region: str | None = None,
    states: list[str] | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Rank airports as terminal-expansion candidates, nationally or in a region."""
    an = Analysis(con)
    scope_label = "United States"
    candidate_codes: list[str] | None = None

    if region:
        resolved = airports_data.resolve_region(region)
        if resolved is None:
            return {
                "error": f"Unknown region '{region}'.",
                "known_regions": airports_data.known_region_names(),
                "hint": "You can also pass explicit state codes via the `states` argument.",
            }
        region_states, label = resolved
        scope_label = label
        candidate_codes = airports_data.airports_in_states(con, region_states)
    elif states:
        scope_label = ", ".join(s.upper() for s in states)
        candidate_codes = airports_data.airports_in_states(con, states)

    assessments = an.assess_many(candidate_codes) if candidate_codes is not None else \
        an.assess_many(list(bts.aggregate_many(con, an.current_month).keys()))

    # Apply the eligibility gate, and report what it excluded so the ranking is
    # not silently narrower than the question asked.
    eligible: list[kpis.Assessment] = []
    excluded: list[dict[str, Any]] = []
    for code, a in assessments.items():
        agg = {"departures": a.facts["ttm_departures"], "passengers": a.facts["ttm_passengers"]}
        if kpis.is_eligible(agg):
            eligible.append(a)
        else:
            excluded.append({"airport": code, "reason": kpis.eligibility_reason(agg)})

    eligible.sort(key=lambda a: (-a.score, -a.facts["ttm_passengers"]))
    top = eligible[: max(1, min(limit, 50))]

    return {
        "scope": scope_label,
        "ranking": [
            {
                "rank": i,
                **{
                    k: v
                    for k, v in a.to_dict().items()
                    if k in ("airport", "airport_name", "city", "score", "verdict",
                             "data_quality", "notes")
                },
                "why": _why_ranked(a, an.baseline_month),
                "key_facts": a.facts,
                "signals": [s.to_dict() for s in a.signals],
            }
            for i, a in enumerate(top, start=1)
        ],
        "eligible_count": len(eligible),
        "excluded_count": len(excluded),
        "excluded_sample": excluded[:5],
        "eligibility_rule": (
            f"At least {config.MIN_TTM_DEPARTURES:,} departures and "
            f"{config.MIN_TTM_PASSENGERS:,} passengers over the trailing 12 months."
        ),
        "methodology": _methodology(an.baseline_month),
        "data_window": an.data_window(),
        "scoring_version": config.SCORING_VERSION,
    }


def _why_ranked(a: kpis.Assessment, baseline_label: str) -> str:
    """One-line plain-English rationale built from the signals themselves."""
    parts: list[str] = []
    util = a.signal("utilization")
    growth = a.signal("growth")
    cons = a.signal("constraint")
    unmet = a.signal("unmet")

    if util and util.value is not None:
        parts.append(f"load factor {util.value:.1f}%")
    if growth and growth.value is not None:
        parts.append(f"passengers {growth.value * 100:+.1f}%/yr vs {baseline_label}")
    if cons and cons.value is not None:
        direction = "upgauging faster than it adds flights" if cons.value > 0 else \
                    "adding flights faster than it upgauges"
        parts.append(f"{direction} ({cons.value:+.1f} pp/yr)")
    if unmet and unmet.value:
        parts.append(f"modelled unmet demand {unmet.value:.1f}% of passengers")
    return "; ".join(parts)


def _methodology(baseline_label: str) -> dict[str, Any]:
    return {
        "composite": (
            "score = 0.25*utilisation + 0.30*growth + 0.30*capacity-constraint "
            "+ 0.15*unmet-demand, each mapped 0-100 through fixed anchor curves."
        ),
        "verdicts": (
            f"STRONG >= {config.VERDICT_STRONG_MIN} and load factor >= "
            f"{config.STRONG_GATE_MIN_LOAD_FACTOR}% and growth > "
            f"{config.STRONG_GATE_MIN_GROWTH}; MODERATE >= {config.VERDICT_MODERATE_MIN}; "
            f"otherwise WEAK."
        ),
        "key_signal": (
            "Capacity constraint compares growth in seats per departure against "
            "growth in departures. Positive means carriers are adding capacity with "
            "bigger aircraft rather than more flights, which is the fingerprint of "
            "gate/slot saturation -- the condition a terminal expansion relieves. "
            "Load factor alone measures airline scheduling, not terminal capacity."
        ),
        "baseline_note": (
            f"Growth and constraint are measured against the trailing 12 months "
            f"ending {baseline_label}, a rolling {config.BASELINE_LOOKBACK_YEARS}-year "
            f"lookback. That window still contains some post-COVID flight restoration, "
            f"which inflates departure growth and therefore biases the constraint "
            f"signal downward; growth against the fixed "
            f"{config.REFERENCE_BASELINE_END} reference is reported alongside as "
            f"long-run context but is not scored."
        ),
    }


# ---------------------------------------------------------------------------
# Tool 2: compare airports
# ---------------------------------------------------------------------------

def compare_airports(
    con: sqlite3.Connection,
    airports: list[str],
    include_live_status: bool = True,
) -> dict[str, Any]:
    """Compare two or more airports on congestion and expansion signals."""
    if not airports or len(airports) < 2:
        return {"error": "Provide at least two airports to compare."}

    an = Analysis(con)
    codes, resolutions, failures = _resolve_all(con, airports)
    if len(codes) < 2:
        return {
            "error": "Could not resolve at least two distinct airports.",
            "resolved": resolutions,
            "failures": failures,
        }

    assessments = an.assess_many(codes)
    missing = [c for c in codes if c not in assessments]
    ordered = [assessments[c] for c in codes if c in assessments]
    if len(ordered) < 2:
        return {
            "error": "Not enough airports have traffic data in the analysis window.",
            "resolved": resolutions,
            "no_data_for": missing,
        }

    congestion = kpis.congestion_ranking(ordered)
    leader = congestion[0]
    trailer = congestion[-1]

    result: dict[str, Any] = {
        "resolved": resolutions,
        "congestion_ranking": congestion,
        "congestion_verdict": (
            f"{leader['airport']} is the more congested of the set "
            f"(congestion index {leader['congestion_index']} vs "
            f"{trailer['congestion_index']} for {trailer['airport']})."
        ),
        "congestion_index_definition": (
            "0.5*utilisation + 0.3*capacity-constraint + 0.2*unmet-demand sub-scores. "
            "Deliberately different from the investment score: an airport can be "
            "congested without being a good expansion candidate, and vice versa."
        ),
        "detail": {a.airport: a.to_dict() for a in ordered},
        "methodology": _methodology(an.baseline_month),
        "data_window": an.data_window(),
        "scoring_version": config.SCORING_VERSION,
    }
    if missing:
        result["no_data_for"] = missing
    if include_live_status:
        result["live_faa_status"] = faa.disruptions_for([a.airport for a in ordered])
    return result


# ---------------------------------------------------------------------------
# Tool 3: airport profile
# ---------------------------------------------------------------------------

def airport_profile(
    con: sqlite3.Connection,
    airport: str,
    include_trend: bool = True,
) -> dict[str, Any]:
    """Full signal breakdown and verdict for a single airport."""
    an = Analysis(con)
    resolution = airports_data.resolve(con, airport)
    if not resolution.ok:
        return {"error": "Could not resolve airport.", "resolved": resolution.to_dict()}

    assessment = an.assess_one(resolution.code)
    if assessment is None:
        return {
            "error": f"No BTS traffic data for {resolution.code} in the analysis window.",
            "resolved": resolution.to_dict(),
            "data_window": an.data_window(),
        }

    agg = {
        "departures": assessment.facts["ttm_departures"],
        "passengers": assessment.facts["ttm_passengers"],
    }
    out: dict[str, Any] = {
        "resolved": resolution.to_dict(),
        **assessment.to_dict(),
        "meets_eligibility_floor": kpis.is_eligible(agg),
        "methodology": _methodology(an.baseline_month),
        "data_window": an.data_window(),
        "live_faa_status": faa.disruptions_for([resolution.code]),
    }
    if not out["meets_eligibility_floor"]:
        out["eligibility_note"] = kpis.eligibility_reason(agg)
    if include_trend:
        out["monthly_trend"] = bts.monthly_series(con, resolution.code, an.current_month)
    return out


# ---------------------------------------------------------------------------
# Tool 4: haul mix
# ---------------------------------------------------------------------------

def haul_mix(
    con: sqlite3.Connection,
    airport: str,
    threshold_miles: float = config.LONG_HAUL_THRESHOLD_MILES,
) -> dict[str, Any]:
    """Share of departures that are long haul.

    Answers in two tiers and always says which one it used. See
    app/data/segments.py for why an exact figure is not available from the live
    API alone.
    """
    an = Analysis(con)
    resolution = airports_data.resolve(con, airport)
    if not resolution.ok:
        return {"error": "Could not resolve airport.", "resolved": resolution.to_dict()}

    try:
        threshold = float(threshold_miles)
    except (TypeError, ValueError):
        threshold = config.LONG_HAUL_THRESHOLD_MILES
    if threshold <= 0:
        threshold = config.LONG_HAUL_THRESHOLD_MILES

    agg = bts.aggregate(con, resolution.code, an.current_month)
    if agg is None:
        return {
            "error": f"No BTS traffic data for {resolution.code}.",
            "resolved": resolution.to_dict(),
        }

    exact = segments.haul_mix_exact(con, resolution.code, threshold)
    tier1 = kpis.haul_mix_from_cohorts(agg, threshold)

    result: dict[str, Any] = {
        "resolved": resolution.to_dict(),
        "threshold_miles": threshold,
        "threshold_rationale": (
            "3,500 km / 2,175 statute miles is the conventional industry boundary "
            "between medium and long haul. Adjustable per question."
        ),
        "tier_1_cohort_estimate": tier1,
        "tier_2_available": exact is not None,
        "data_window": an.data_window(),
        "scoring_version": config.SCORING_VERSION,
    }

    if exact is not None:
        result["tier_2_exact"] = exact
        result["answer"] = {
            "basis": "exact",
            "long_haul_share_pct": exact["long_haul_share_pct"],
            "statement": (
                f"{exact['long_haul_share_pct']}% of departures from "
                f"{resolution.code} exceed {threshold:,.0f} statute miles "
                f"({exact['long_haul_departures']:,} of {exact['total_departures']:,} "
                f"departures across {exact['route_count']} routes). Computed from "
                f"per-route segment data."
            ),
        }
    else:
        lo, hi = tier1["long_haul_share_pct_range"]
        result["answer"] = {
            "basis": "cohort_estimate",
            "long_haul_share_pct_estimate": tier1["long_haul_share_pct_estimate"],
            "long_haul_share_pct_range": tier1["long_haul_share_pct_range"],
            "robust_floor_pct": tier1["robust_floor_pct"],
            "statement": (
                f"An estimated {tier1['long_haul_share_pct_estimate']}% of departures "
                f"from {resolution.code} exceed {threshold:,.0f} statute miles "
                f"(range {lo}-{hi}%). At least {tier1['robust_floor_pct']}% is certain "
                f"without any distributional assumption. This is an ESTIMATE: BTS "
                f"publishes a mean stage length per cohort, not per-flight distances."
            ),
        }
    return result


# ---------------------------------------------------------------------------
# Tool 5: unmet demand
# ---------------------------------------------------------------------------

def unmet_demand(con: sqlite3.Connection, airport: str) -> dict[str, Any]:
    """Modelled unmet demand for one airport, decomposed into its drivers."""
    an = Analysis(con)
    resolution = airports_data.resolve(con, airport)
    if not resolution.ok:
        return {"error": "Could not resolve airport.", "resolved": resolution.to_dict()}

    assessment = an.assess_one(resolution.code)
    if assessment is None:
        return {
            "error": f"No BTS traffic data for {resolution.code}.",
            "resolved": resolution.to_dict(),
        }

    unmet = assessment.signal("unmet")
    util = assessment.signal("utilization")
    cons = assessment.signal("constraint")
    growth = assessment.signal("growth")
    components = unmet.detail["components"] if unmet else {}

    drivers: list[str] = []
    if components.get("spill_from_high_load_factor_pct", 0) > 0:
        drivers.append(
            f"Load factor {util.value:.1f}% is above the "
            f"{config.LOAD_FACTOR_COMFORT}% comfort threshold, implying roughly "
            f"{components['estimated_spilled_passengers']:,} passengers of spill "
            f"({components['spill_from_high_load_factor_pct']}% of throughput)."
        )
    if components.get("demand_outrunning_capacity_pct_per_year", 0) > 0:
        pax_rate = components.get("passenger_growth_pct_per_year") or 0.0
        seat_rate = components.get("seat_growth_pct_per_year") or 0.0
        gap = components["demand_outrunning_capacity_pct_per_year"]
        if pax_rate >= 0:
            drivers.append(
                f"Passengers are growing {pax_rate}%/yr against seat growth of "
                f"{seat_rate}%/yr, so demand is outrunning capacity by {gap} points "
                f"a year."
            )
        else:
            # Both shrinking: capacity is being cut faster than demand is falling,
            # which still tightens the airport even though nothing is growing.
            drivers.append(
                f"Both traffic and capacity are below the baseline, but seats are "
                f"being cut faster ({seat_rate}%/yr) than passengers are falling "
                f"({pax_rate}%/yr), so the airport is tightening by {gap} points a "
                f"year despite the decline."
            )
    if cons and cons.value is not None and cons.value > 0:
        drivers.append(
            f"Carriers are upgauging {cons.detail.get('gauge_growth_pct_per_year')}%/yr "
            f"while departures move "
            f"{cons.detail.get('departure_growth_pct_per_year')}%/yr, which is what "
            f"an airport does when it cannot add flights."
        )
    if not drivers:
        drivers.append(
            f"No unmet demand detected: load factor "
            f"{util.value:.1f}% is below the {config.LOAD_FACTOR_COMFORT}% comfort "
            f"threshold and seat capacity is keeping pace with passenger growth."
            if util and util.value is not None else "No unmet demand detected."
        )

    return {
        "resolved": resolution.to_dict(),
        "airport": assessment.airport,
        "airport_name": assessment.airport_name,
        "unmet_demand_index_pct_of_passengers": (
            round(unmet.value, 2) if unmet and unmet.value is not None else 0.0
        ),
        "basis": "model_estimate",
        "components": components,
        "why": drivers,
        "supporting_signals": {
            "utilization": util.to_dict() if util else None,
            "growth": growth.to_dict() if growth else None,
            "constraint": cons.to_dict() if cons else None,
        },
        "key_facts": assessment.facts,
        "caveat": (
            "This is a MODEL ESTIMATE, not a measurement. BTS reports passengers "
            "carried; no public dataset reports passengers turned away. The figure "
            "infers spill from load factor above a comfort threshold and from "
            "passenger growth outpacing seat growth. Fare data and schedule-level "
            "booking curves would be needed to measure it directly."
        ),
        "data_quality": assessment.data_quality,
        "notes": assessment.notes,
        "data_window": an.data_window(),
        "scoring_version": config.SCORING_VERSION,
    }


# ---------------------------------------------------------------------------
# Schemas exposed to the model
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "rank_expansion_candidates",
        "description": (
            "Rank US airports as terminal-expansion investment candidates, either "
            "nationally or within a named region or set of states. Returns a scored, "
            "ranked list with each airport's signal breakdown and a plain-English "
            "rationale. Use this for questions like 'which airports in New England "
            "are strong candidates for terminal expansion' or 'best expansion "
            "opportunities in Texas'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": (
                        "Named region, e.g. 'New England', 'Pacific Northwest', "
                        "'Southeast', 'California'. Omit for a national ranking."
                    ),
                },
                "states": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Two-letter state codes, e.g. ['MA','CT']. Use when the user "
                        "names states rather than a region."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "How many airports to return. Default 10, max 50.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "compare_airports",
        "description": (
            "Compare two or more airports side by side on congestion and expansion "
            "signals, and rank them by a congestion index. Accepts free-text names "
            "including metro shorthand such as 'LA' or 'Santa Ana' and reports what "
            "each resolved to. Also returns live FAA ground-stop and delay-program "
            "status. Use for 'compare X and Y congestion levels'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "airports": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Two or more airport references: IATA codes, city names, or "
                        "metro shorthand."
                    ),
                },
                "include_live_status": {
                    "type": "boolean",
                    "description": "Include live FAA status. Default true.",
                },
            },
            "required": ["airports"],
        },
    },
    {
        "name": "airport_profile",
        "description": (
            "Full investment assessment for one airport: all four scored signals with "
            "raw inputs, the composite score, verdict with gate detail, key facts, a "
            "12-month trend series, and live FAA status. Use for 'tell me about X', "
            "'is X a good candidate', or any follow-up needing one airport's detail."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "airport": {
                    "type": "string",
                    "description": "Airport reference: IATA code, city, or metro name.",
                },
                "include_trend": {
                    "type": "boolean",
                    "description": "Include the monthly trend series. Default true.",
                },
            },
            "required": ["airport"],
        },
    },
    {
        "name": "haul_mix",
        "description": (
            "Share of an airport's departures that are long haul, with the distance "
            "threshold configurable. Returns an exact figure when per-route segment "
            "data is loaded, otherwise a clearly-labelled estimate with an "
            "uncertainty range and a robust floor. Always report which basis was "
            "used. Use for 'what percentage of flights out of X are long haul' or "
            "questions about route mix and stage length."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "airport": {"type": "string", "description": "Airport reference."},
                "threshold_miles": {
                    "type": "number",
                    "description": (
                        "Long-haul boundary in statute miles. Default 2175 "
                        "(3,500 km), the conventional industry definition."
                    ),
                },
            },
            "required": ["airport"],
        },
    },
    {
        "name": "unmet_demand",
        "description": (
            "Estimated unmet passenger demand at one airport, decomposed into its "
            "drivers with an explanation of each. Returns a model estimate, never a "
            "measurement, and says so. Use for 'what is the unmet demand at X and "
            "why' or 'is X turning away passengers'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "airport": {"type": "string", "description": "Airport reference."},
            },
            "required": ["airport"],
        },
    },
]


_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "rank_expansion_candidates": rank_expansion_candidates,
    "compare_airports": compare_airports,
    "airport_profile": airport_profile,
    "haul_mix": haul_mix,
    "unmet_demand": unmet_demand,
}


def dispatch(con: sqlite3.Connection, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run a tool by name. Never raises -- errors come back as data the model reads."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'.", "available": sorted(_DISPATCH)}
    try:
        return fn(con, **(arguments or {}))
    except TypeError as exc:
        return {"error": f"Invalid arguments for '{name}': {exc}"}
    except Exception as exc:  # noqa: BLE001 -- surface to the model, don't crash the loop
        return {"error": f"{name} failed: {type(exc).__name__}: {exc}"}
