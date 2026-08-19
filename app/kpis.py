"""
Deterministic scoring engine.

Pure functions only -- no I/O, no network, no database. Everything here takes
plain dicts of aggregates and returns dataclasses, which is what makes the whole
scoring surface unit-testable and reproducible.

The language model never performs arithmetic. It selects a tool, the tool calls
into this module, and this module returns both the numbers and the raw inputs
they came from. Any figure the agent quotes can therefore be recomputed offline.

THE INVESTMENT THESIS
---------------------
We are looking for airports where a renovation unlocks revenue, which means
airports where demand is already pressing against physical capacity. Load factor
alone does not capture that: it measures how full the aircraft are, which is an
airline scheduling decision, not a terminal constraint.

The sharper signal is UPGAUGING (signal 3). When carriers want to add capacity at
an airport they can either add flights or fly bigger aircraft. Adding flights is
cheaper and more flexible, so they do that first -- unless they cannot, because
gates, slots or runway capacity are saturated. Seats-per-departure rising while
departures stay flat is the fingerprint of an airport that has run out of room,
and that is precisely where a terminal expansion converts into revenue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from app import config


# ---------------------------------------------------------------------------
# Anchor interpolation
# ---------------------------------------------------------------------------

def interpolate(anchors: Sequence[tuple[float, float]], value: float) -> float:
    """Map a raw signal value to 0-100 through a piecewise-linear anchor curve.

    Clamps outside the anchor range, so an airport far beyond the top anchor
    scores 100 rather than running off the scale.
    """
    pts = sorted(anchors)
    if value <= pts[0][0]:
        return pts[0][1]
    if value >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= value <= x1:
            if x1 == x0:
                return y1
            frac = (value - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    return pts[-1][1]


def _cagr(current: float, prior: float, years: float) -> float | None:
    """Compound annual growth rate, or None when it is not defined."""
    if prior is None or current is None or prior <= 0 or current <= 0 or years <= 0:
        return None
    return (current / prior) ** (1.0 / years) - 1.0


def _safe_div(numerator: float, denominator: float) -> float | None:
    if not denominator:
        return None
    return numerator / denominator


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """One scored signal: the raw measurement plus its 0-100 sub-score."""

    key: str
    label: str
    value: float | None
    unit: str
    subscore: float
    weight: float
    available: bool = True
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "signal": self.key,
            "label": self.label,
            "value": None if self.value is None else round(self.value, 3),
            "unit": self.unit,
            "subscore": round(self.subscore, 1),
            "weight": self.weight,
            "contribution": round(self.subscore * self.weight, 1),
            "available": self.available,
        }
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass
class Assessment:
    """A fully scored airport."""

    airport: str
    airport_name: str | None
    city: str | None
    score: float
    verdict: str
    signals: list[Signal]
    facts: dict[str, Any]
    gates: dict[str, Any]
    data_quality: str
    notes: list[str] = field(default_factory=list)

    def signal(self, key: str) -> Signal | None:
        return next((s for s in self.signals if s.key == key), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "airport": self.airport,
            "airport_name": self.airport_name,
            "city": self.city,
            "score": round(self.score, 1),
            "verdict": self.verdict,
            "signals": [s.to_dict() for s in self.signals],
            "facts": self.facts,
            "verdict_gates": self.gates,
            "data_quality": self.data_quality,
            "notes": self.notes,
            "scoring_version": config.SCORING_VERSION,
        }


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def is_eligible(agg: dict[str, Any]) -> bool:
    """Traffic floor for the expansion thesis to be meaningful."""
    return (
        agg.get("departures", 0) >= config.MIN_TTM_DEPARTURES
        and agg.get("passengers", 0) >= config.MIN_TTM_PASSENGERS
    )


def eligibility_reason(agg: dict[str, Any]) -> str | None:
    if agg.get("departures", 0) < config.MIN_TTM_DEPARTURES:
        return (
            f"{agg.get('departures', 0):,} TTM departures is below the "
            f"{config.MIN_TTM_DEPARTURES:,} floor for expansion analysis"
        )
    if agg.get("passengers", 0) < config.MIN_TTM_PASSENGERS:
        return (
            f"{agg.get('passengers', 0):,} TTM passengers is below the "
            f"{config.MIN_TTM_PASSENGERS:,} floor for expansion analysis"
        )
    return None


# ---------------------------------------------------------------------------
# The five signals
# ---------------------------------------------------------------------------

def load_factor(agg: dict[str, Any]) -> float | None:
    lf = _safe_div(agg.get("passengers", 0), agg.get("seats", 0))
    return None if lf is None else lf * 100.0


def gauge(agg: dict[str, Any]) -> float | None:
    """Seats per departure -- average aircraft size operated at the airport."""
    return _safe_div(agg.get("seats", 0), agg.get("departures", 0))


def assess(
    agg: dict[str, Any],
    baseline: dict[str, Any] | None,
    baseline_years: float,
    baseline_label: str,
    reference: dict[str, Any] | None = None,
    reference_years: float = 0.0,
    reference_label: str = config.REFERENCE_BASELINE_END,
) -> Assessment:
    """Score one airport against the rolling baseline window.

    `baseline` is the matched TTM window ending at `baseline_label`, one
    `BASELINE_LOOKBACK_YEARS` behind the current window. When it is missing -- a new
    airport, or one with no service then -- the growth-derived signals are marked
    unavailable, scored at their neutral anchor, and the STRONG verdict is gated
    off. An explicitly partial answer beats a confident-looking one built on a
    missing denominator.

    `reference` is the fixed pre-pandemic window, reported as long-run context but
    never scored: it says where the airport sits relative to before COVID, which is
    a different question from which way it is moving now.
    """
    notes: list[str] = []

    lf = load_factor(agg)
    cur_gauge = gauge(agg)

    pax_cagr = seat_cagr = dep_cagr = gauge_cagr = None
    if baseline:
        pax_cagr = _cagr(agg.get("passengers", 0), baseline.get("passengers", 0), baseline_years)
        seat_cagr = _cagr(agg.get("seats", 0), baseline.get("seats", 0), baseline_years)
        dep_cagr = _cagr(agg.get("departures", 0), baseline.get("departures", 0), baseline_years)
        base_gauge = gauge(baseline)
        if cur_gauge and base_gauge:
            gauge_cagr = _cagr(cur_gauge, base_gauge, baseline_years)

    reference_pax_cagr = None
    if reference:
        reference_pax_cagr = _cagr(
            agg.get("passengers", 0), reference.get("passengers", 0), reference_years
        )

    growth_available = pax_cagr is not None
    constraint_available = gauge_cagr is not None and dep_cagr is not None

    # Baseline adequacy: a CAGR against a tiny or structurally different baseline
    # is measuring a step-change in service, not an organic trend.
    baseline_thin = False
    expansion_ratio = None
    if baseline and baseline.get("departures"):
        expansion_ratio = agg.get("departures", 0) / baseline["departures"]
        if baseline["departures"] < config.MIN_BASELINE_DEPARTURES:
            baseline_thin = True
            notes.append(
                f"Baseline window has only {baseline['departures']:,} departures, "
                f"below the {config.MIN_BASELINE_DEPARTURES:,} floor for a stable "
                f"growth rate. Treat growth and constraint as indicative only."
            )
        if expansion_ratio > config.MAX_BASELINE_EXPANSION_RATIO:
            baseline_thin = True
            notes.append(
                f"Departures are {expansion_ratio:.1f}x the {baseline_label} baseline. "
                f"That is a step-change in service -- a new terminal, or a carrier "
                f"opening a base -- rather than organic growth, so the CAGR describes "
                f"a level shift and the constraint index compares two different "
                f"route networks."
            )

    if not growth_available:
        notes.append(
            f"No comparable traffic in the {baseline_label} baseline window, so "
            f"growth and capacity-constraint signals could not be computed. Scored "
            f"on current-state signals only; cannot qualify as STRONG."
        )

    # -- Signal 1: utilisation -------------------------------------------------
    util_signal = Signal(
        key="utilization",
        label="Seat utilisation (load factor)",
        value=lf,
        unit="percent",
        subscore=interpolate(config.ANCHORS["utilization"], lf) if lf is not None else 0.0,
        weight=config.WEIGHTS["utilization"],
        available=lf is not None,
        detail={
            "ttm_passengers": agg.get("passengers", 0),
            "ttm_seats": agg.get("seats", 0),
            "interpretation": (
                "How full departing aircraft are. Necessary but not sufficient: "
                "this is an airline scheduling outcome, not a terminal constraint."
            ),
        },
    )

    # -- Signal 2: demand growth ----------------------------------------------
    growth_signal = Signal(
        key="growth",
        label=f"Passenger demand growth (CAGR vs {baseline_label} baseline)",
        value=pax_cagr,
        unit="fraction_per_year",
        subscore=(
            interpolate(config.ANCHORS["growth"], pax_cagr)
            if growth_available
            else config.UNAVAILABLE_SUBSCORE
        ),
        weight=config.WEIGHTS["growth"],
        available=growth_available,
        detail={
            "ttm_passengers": agg.get("passengers", 0),
            "baseline_passengers": baseline.get("passengers") if baseline else None,
            "baseline_window_ends": baseline_label,
            "years_elapsed": round(baseline_years, 2),
            "percent_per_year": round(pax_cagr * 100, 2) if pax_cagr is not None else None,
            "vs_prepandemic_reference_pct_per_year": (
                round(reference_pax_cagr * 100, 2) if reference_pax_cagr is not None else None
            ),
            "reference_window_ends": reference_label,
            "interpretation": (
                "Trailing-12-month passengers against the matched 12-month window "
                f"ending {baseline_label}, so seasonality cancels. Growth against the "
                f"fixed {reference_label} reference is shown for long-run context but "
                "is not scored."
            ),
        },
    )

    # -- Signal 3: upgauging constraint (the differentiator) -------------------
    constraint_index = None
    if constraint_available:
        # Percentage points per year: how much faster aircraft size is growing
        # than flight count.
        constraint_index = (gauge_cagr - dep_cagr) * 100.0

    constraint_signal = Signal(
        key="constraint",
        label="Capacity constraint (upgauging vs frequency)",
        value=constraint_index,
        unit="percentage_points_per_year",
        subscore=(
            interpolate(config.ANCHORS["constraint"], constraint_index)
            if constraint_available
            else config.UNAVAILABLE_SUBSCORE
        ),
        weight=config.WEIGHTS["constraint"],
        available=constraint_available,
        detail={
            "seats_per_departure_now": round(cur_gauge, 1) if cur_gauge else None,
            "seats_per_departure_baseline": (
                round(gauge(baseline), 1) if baseline and gauge(baseline) else None
            ),
            "baseline_window_ends": baseline_label,
            "gauge_growth_pct_per_year": round(gauge_cagr * 100, 2) if gauge_cagr is not None else None,
            "departure_growth_pct_per_year": round(dep_cagr * 100, 2) if dep_cagr is not None else None,
            "interpretation": (
                "Positive means carriers are adding capacity with bigger aircraft "
                "rather than more flights -- the signature of gate, slot or runway "
                "saturation, and the case where a terminal expansion unlocks revenue. "
                f"Measured against the rolling {baseline_label} baseline, which still "
                "carries some post-COVID flight restoration: that restoration inflates "
                "departure growth and so biases this signal downward."
            ),
        },
    )

    # -- Signal 4: unmet demand (model estimate, never a measurement) ----------
    spill_pct_of_pax = 0.0
    spill_pax = 0.0
    if lf is not None and lf > config.LOAD_FACTOR_COMFORT:
        # Seats that would have been needed to hold the load factor at the comfort
        # threshold; expressed against passengers so it is comparable across sizes.
        spill_pax = agg.get("seats", 0) * (lf - config.LOAD_FACTOR_COMFORT) / 100.0
        spill_pct_of_pax = 100.0 * spill_pax / agg["passengers"] if agg.get("passengers") else 0.0

    growth_gap_pct = 0.0
    if pax_cagr is not None and seat_cagr is not None:
        # Demand outrunning capacity: passengers compounding faster than seats.
        growth_gap_pct = max((pax_cagr - seat_cagr) * 100.0, 0.0)

    unmet_index = spill_pct_of_pax + growth_gap_pct

    unmet_signal = Signal(
        key="unmet",
        label="Unmet demand (modelled)",
        value=unmet_index,
        unit="percent_of_ttm_passengers",
        subscore=interpolate(config.ANCHORS["unmet"], unmet_index),
        weight=config.WEIGHTS["unmet"],
        available=True,
        detail={
            "basis": "model_estimate",
            "components": {
                "spill_from_high_load_factor_pct": round(spill_pct_of_pax, 2),
                "estimated_spilled_passengers": round(spill_pax),
                "demand_outrunning_capacity_pct_per_year": round(growth_gap_pct, 2),
                "passenger_growth_pct_per_year": (
                    round(pax_cagr * 100, 2) if pax_cagr is not None else None
                ),
                "seat_growth_pct_per_year": (
                    round(seat_cagr * 100, 2) if seat_cagr is not None else None
                ),
            },
            "load_factor_comfort_threshold": config.LOAD_FACTOR_COMFORT,
            "interpretation": (
                "A MODEL ESTIMATE, not a measurement. BTS reports passengers "
                "carried, never passengers turned away. Component one infers spill "
                "from load factor above the comfort threshold; component two from "
                "passenger growth outpacing seat growth."
            ),
        },
    )

    signals = [util_signal, growth_signal, constraint_signal, unmet_signal]
    score = sum(s.subscore * s.weight for s in signals)

    # -- Verdict, with hard gates ---------------------------------------------
    gate_lf = lf is not None and lf >= config.STRONG_GATE_MIN_LOAD_FACTOR
    gate_growth = pax_cagr is not None and pax_cagr > config.STRONG_GATE_MIN_GROWTH
    gates_passed = gate_lf and gate_growth

    if score >= config.VERDICT_STRONG_MIN and gates_passed:
        verdict = "STRONG"
    elif score >= config.VERDICT_MODERATE_MIN:
        verdict = "MODERATE"
    else:
        verdict = "WEAK"

    if score >= config.VERDICT_STRONG_MIN and not gates_passed:
        failed = []
        if not gate_lf:
            failed.append(
                f"load factor {lf:.1f}% is below the {config.STRONG_GATE_MIN_LOAD_FACTOR}% gate"
                if lf is not None else "load factor unavailable"
            )
        if not gate_growth:
            failed.append(
                f"passenger growth {pax_cagr * 100:.1f}%/yr is not above "
                f"{config.STRONG_GATE_MIN_GROWTH}%"
                if pax_cagr is not None else "passenger growth unavailable"
            )
        notes.append(
            f"Scored {score:.1f} (STRONG range) but held at MODERATE: "
            + "; ".join(failed)
            + ". Upgauging without growth is fleet rationalisation, not capacity pressure."
        )

    return Assessment(
        airport=agg.get("airport", "?"),
        airport_name=agg.get("airport_name"),
        city=agg.get("city_name"),
        score=score,
        verdict=verdict,
        signals=signals,
        facts={
            "ttm_departures": agg.get("departures", 0),
            "ttm_passengers": agg.get("passengers", 0),
            "ttm_seats": agg.get("seats", 0),
            "load_factor_pct": round(lf, 1) if lf is not None else None,
            "seats_per_departure": round(cur_gauge, 1) if cur_gauge else None,
            "mean_stage_miles": round(agg.get("mean_stage_miles", 0.0)),
            "domestic_share_of_departures_pct": (
                round(100.0 * agg.get("domestic_departures", 0) / agg["departures"], 1)
                if agg.get("departures") else None
            ),
            "baseline_window_ends": baseline_label,
            "baseline_departures": baseline.get("departures") if baseline else None,
            "baseline_seats_per_departure": (
                round(gauge(baseline), 1) if baseline and gauge(baseline) else None
            ),
            "prepandemic_reference_window_ends": reference_label,
            "pax_growth_vs_prepandemic_pct_per_year": (
                round(reference_pax_cagr * 100, 2) if reference_pax_cagr is not None else None
            ),
        },
        gates={
            "strong_requires_load_factor_at_least": config.STRONG_GATE_MIN_LOAD_FACTOR,
            "strong_requires_growth_above": config.STRONG_GATE_MIN_GROWTH,
            "load_factor_gate_passed": gate_lf,
            "growth_gate_passed": gate_growth,
        },
        data_quality=(
            "complete"
            if growth_available and constraint_available and not baseline_thin
            else "partial"
        ),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Haul mix -- Tier 1 (cohort estimate)
# ---------------------------------------------------------------------------

def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def lognormal_tail_share(mean: float, threshold: float, cv: float) -> float:
    """P(X >= threshold) for a lognormal with the given mean and coefficient of variation.

    Used to estimate what fraction of a cohort's flights exceed the long-haul
    threshold when only the cohort MEAN is published.
    """
    if mean <= 0 or threshold <= 0:
        return 0.0
    if cv <= 0:
        return 1.0 if mean >= threshold else 0.0
    sigma_sq = math.log(1.0 + cv * cv)
    sigma = math.sqrt(sigma_sq)
    mu = math.log(mean) - sigma_sq / 2.0
    return _normal_cdf((mu - math.log(threshold)) / sigma)


def haul_mix_from_cohorts(
    agg: dict[str, Any],
    threshold_miles: float = config.LONG_HAUL_THRESHOLD_MILES,
) -> dict[str, Any]:
    """Tier-1 haul mix from the two cohorts BTS actually publishes.

    BTS gives departures and a MEAN stage length for the total and domestic
    cohorts; international is derived as the difference. That yields two cohorts
    with known size and known mean, but no within-cohort distribution -- so the
    honest answer is a range plus a robust floor, not a single number.
    """
    total_dep = agg.get("departures", 0)
    dom_dep = agg.get("domestic_departures", 0)
    intl_dep = agg.get("international_departures", 0)
    dom_mean = agg.get("mean_stage_miles_domestic", 0.0)
    intl_mean = agg.get("mean_stage_miles_international", 0.0)

    if not total_dep:
        return {
            "basis": "unavailable",
            "reason": "no departures in the analysis window",
        }

    cohorts = []
    if dom_dep > 0:
        cohorts.append(("domestic", dom_dep, dom_mean))
    if intl_dep > 0:
        cohorts.append(("international", intl_dep, intl_mean))

    def share_at(cv: float) -> float:
        weighted = sum(
            dep * lognormal_tail_share(mean, threshold_miles, cv)
            for _, dep, mean in cohorts
        )
        return 100.0 * weighted / total_dep

    point = share_at(config.HAUL_MIX_CV)
    lo_cv, hi_cv = config.HAUL_MIX_CV_BAND
    # The tail share is not monotonic in CV: raising spread moves mass ABOVE the
    # threshold for a cohort whose mean sits below it, and BELOW the threshold for
    # one whose mean sits above it. With cohorts on both sides those effects fight,
    # so take the envelope over all three CVs to guarantee the range contains the
    # point estimate rather than assuming the endpoints bracket it.
    candidates = [share_at(lo_cv), point, share_at(hi_cv)]
    band = [min(candidates), max(candidates)]

    # The robust floor needs no distributional assumption at all: a cohort whose
    # MEAN already exceeds the threshold is unambiguously long-haul in aggregate.
    floor_dep = sum(dep for _, dep, mean in cohorts if mean >= threshold_miles)
    floor_pct = 100.0 * floor_dep / total_dep

    cohort_detail = [
        {
            "cohort": name,
            "departures": dep,
            "share_of_departures_pct": round(100.0 * dep / total_dep, 1),
            "mean_stage_miles": round(mean),
            "mean_exceeds_threshold": mean >= threshold_miles,
            "estimated_pct_of_cohort_over_threshold": round(
                100.0 * lognormal_tail_share(mean, threshold_miles, config.HAUL_MIX_CV), 1
            ),
        }
        for name, dep, mean in cohorts
    ]

    return {
        "basis": "cohort_estimate",
        "threshold_miles": threshold_miles,
        "long_haul_share_pct_estimate": round(point, 1),
        "long_haul_share_pct_range": [round(band[0], 1), round(band[1], 1)],
        "robust_floor_pct": round(floor_pct, 1),
        "total_departures": total_dep,
        "cohorts": cohort_detail,
        "mean_stage_miles_overall": round(agg.get("mean_stage_miles", 0.0)),
        "assumptions": [
            "BTS publishes a mean stage length per cohort, never a per-flight "
            "distance distribution, so an exact percentage is not directly observable.",
            f"Within each cohort, stage lengths are modelled as lognormal with "
            f"coefficient of variation {config.HAUL_MIX_CV}; the range comes from "
            f"varying that between {config.HAUL_MIX_CV_BAND[0]} and "
            f"{config.HAUL_MIX_CV_BAND[1]}.",
            "The robust floor assumes nothing about the distribution: it counts only "
            "cohorts whose mean already exceeds the threshold.",
            "International figures are derived as total minus domestic.",
        ],
        "how_to_get_exact": (
            "Add a BTS T-100 Segment CSV to data/segments/ and run "
            "python -m app.data.segments --refresh to compute the true per-route "
            "distance distribution."
        ),
    }


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def congestion_ranking(assessments: Sequence[Assessment]) -> list[dict[str, Any]]:
    """Order airports by how congested they are, most first.

    Congestion is deliberately not the same thing as the investment score: an
    airport can be congested without being a good expansion candidate (no growth),
    and vice versa. This ranks on utilisation and constraint only.
    """
    rows = []
    for a in assessments:
        util = a.signal("utilization")
        cons = a.signal("constraint")
        unmet = a.signal("unmet")
        congestion = (
            0.5 * (util.subscore if util else 0.0)
            + 0.3 * (cons.subscore if cons else 0.0)
            + 0.2 * (unmet.subscore if unmet else 0.0)
        )
        rows.append(
            {
                "airport": a.airport,
                "airport_name": a.airport_name,
                "congestion_index": round(congestion, 1),
                "load_factor_pct": a.facts.get("load_factor_pct"),
                "seats_per_departure": a.facts.get("seats_per_departure"),
                "constraint_index": (
                    round(cons.value, 2) if cons and cons.value is not None else None
                ),
                "ttm_departures": a.facts.get("ttm_departures"),
                "ttm_passengers": a.facts.get("ttm_passengers"),
            }
        )
    rows.sort(key=lambda r: -r["congestion_index"])
    for i, row in enumerate(rows, start=1):
        row["congestion_rank"] = i
    return rows
