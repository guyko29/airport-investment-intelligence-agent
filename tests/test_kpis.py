"""
Scoring invariants.

These are the tests that matter most: the scoring engine is the part of the system
that must be defensible to an investment committee, so its behaviour is pinned
here rather than left to inspection of the live data.
"""

from __future__ import annotations

import pytest

from app import config, kpis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_agg(
    departures: int = 20_000,
    passengers: int = 2_000_000,
    seats: int = 2_500_000,
    domestic_departures: int | None = None,
    mean_stage: float = 900.0,
    domestic_stage: float = 800.0,
    airport: str = "TST",
) -> dict:
    """A synthetic airport-window aggregate shaped like bts.aggregate output."""
    dom_dep = departures if domestic_departures is None else domestic_departures
    intl_dep = departures - dom_dep
    dom_miles = dom_dep * domestic_stage
    total_miles = departures * mean_stage
    return {
        "airport": airport,
        "airport_name": f"{airport} Test Field",
        "city_name": "Testville",
        "months": 12,
        "departures": departures,
        "passengers": passengers,
        "seats": seats,
        "domestic_departures": dom_dep,
        "domestic_passengers": int(passengers * dom_dep / departures) if departures else 0,
        "domestic_seats": int(seats * dom_dep / departures) if departures else 0,
        "international_departures": intl_dep,
        "mean_stage_miles": mean_stage,
        "mean_stage_miles_domestic": domestic_stage,
        "mean_stage_miles_international": (
            (total_miles - dom_miles) / intl_dep if intl_dep > 0 else 0.0
        ),
    }


BASELINE_YEARS = 6.33


def assess(current: dict, baseline: dict | None) -> kpis.Assessment:
    return kpis.assess(current, baseline, BASELINE_YEARS, "2019-12")


# ---------------------------------------------------------------------------
# Anchor interpolation
# ---------------------------------------------------------------------------

def test_interpolate_hits_anchor_points_exactly():
    anchors = [(0.0, 0.0), (10.0, 50.0), (20.0, 100.0)]
    assert kpis.interpolate(anchors, 0.0) == 0.0
    assert kpis.interpolate(anchors, 10.0) == 50.0
    assert kpis.interpolate(anchors, 20.0) == 100.0


def test_interpolate_is_linear_between_anchors():
    anchors = [(0.0, 0.0), (10.0, 50.0)]
    assert kpis.interpolate(anchors, 5.0) == pytest.approx(25.0)


def test_interpolate_clamps_outside_the_anchor_range():
    anchors = [(0.0, 0.0), (10.0, 100.0)]
    assert kpis.interpolate(anchors, -999.0) == 0.0
    assert kpis.interpolate(anchors, 999.0) == 100.0


def test_interpolate_is_monotonic_across_every_configured_signal():
    for name, anchors in config.ANCHORS.items():
        lo = min(a[0] for a in anchors)
        hi = max(a[0] for a in anchors)
        step = (hi - lo) / 50.0
        previous = -1.0
        for i in range(51):
            score = kpis.interpolate(anchors, lo + i * step)
            assert score >= previous - 1e-9, f"{name} is not monotonic at {lo + i * step}"
            previous = score


# ---------------------------------------------------------------------------
# Composite behaviour
# ---------------------------------------------------------------------------

def test_score_rises_with_load_factor():
    baseline = make_agg(passengers=1_800_000, seats=2_400_000)
    low = assess(make_agg(passengers=1_800_000, seats=2_500_000), baseline)
    high = assess(make_agg(passengers=2_150_000, seats=2_500_000), baseline)
    assert high.score > low.score


def test_score_rises_with_passenger_growth():
    flat = assess(make_agg(passengers=2_000_000), make_agg(passengers=2_000_000))
    grown = assess(make_agg(passengers=2_000_000), make_agg(passengers=1_200_000))
    assert grown.score > flat.score


def test_upgauging_scores_higher_than_added_frequency():
    """The core thesis: same capacity growth, different mechanism, different score."""
    baseline = make_agg(departures=20_000, seats=2_000_000, passengers=1_650_000)

    # Capacity added by flying bigger aircraft; flight count unchanged.
    upgauged = assess(
        make_agg(departures=20_000, seats=2_600_000, passengers=2_150_000), baseline
    )
    # The same capacity added by flying more of the same aircraft.
    more_flights = assess(
        make_agg(departures=26_000, seats=2_600_000, passengers=2_150_000), baseline
    )

    assert upgauged.signal("constraint").value > more_flights.signal("constraint").value
    assert upgauged.score > more_flights.score


def test_composite_equals_the_weighted_sum_of_its_signals():
    a = assess(make_agg(), make_agg(passengers=1_500_000))
    expected = sum(s.subscore * s.weight for s in a.signals)
    assert a.score == pytest.approx(expected)
    assert sum(s.weight for s in a.signals) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Verdict gates
# ---------------------------------------------------------------------------

def test_high_score_is_held_below_strong_when_load_factor_gate_fails():
    """Upgauging on a low-load-factor airport must not read as capacity pressure."""
    baseline = make_agg(departures=30_000, seats=2_000_000, passengers=1_300_000)
    current = make_agg(departures=20_000, seats=2_600_000, passengers=1_750_000)
    a = assess(current, baseline)

    assert a.facts["load_factor_pct"] < config.STRONG_GATE_MIN_LOAD_FACTOR
    assert a.verdict != "STRONG"
    if a.score >= config.VERDICT_STRONG_MIN:
        assert any("held at MODERATE" in n for n in a.notes)


def test_strong_verdict_requires_positive_growth():
    baseline = make_agg(departures=20_000, seats=2_200_000, passengers=2_100_000)
    shrinking = make_agg(departures=15_000, seats=2_300_000, passengers=2_000_000)
    a = assess(shrinking, baseline)
    assert a.signal("growth").value < 0
    assert a.verdict != "STRONG"


def test_verdict_bands_follow_the_configured_thresholds():
    a = assess(make_agg(passengers=2_200_000, seats=2_500_000), make_agg(passengers=1_400_000))
    if a.verdict == "STRONG":
        assert a.score >= config.VERDICT_STRONG_MIN
    elif a.verdict == "MODERATE":
        assert a.score >= config.VERDICT_MODERATE_MIN
    else:
        assert a.score < config.VERDICT_MODERATE_MIN


# ---------------------------------------------------------------------------
# Unmet demand
# ---------------------------------------------------------------------------

def test_no_spill_below_the_comfort_threshold():
    current = make_agg(passengers=1_900_000, seats=2_500_000)   # 76% load factor
    a = assess(current, make_agg(passengers=1_900_000, seats=2_500_000))
    components = a.signal("unmet").detail["components"]
    assert components["spill_from_high_load_factor_pct"] == 0
    assert components["estimated_spilled_passengers"] == 0


def test_spill_appears_above_the_comfort_threshold():
    current = make_agg(passengers=2_200_000, seats=2_500_000)   # 88% load factor
    a = assess(current, make_agg(passengers=2_200_000, seats=2_500_000))
    components = a.signal("unmet").detail["components"]
    assert components["spill_from_high_load_factor_pct"] > 0
    assert components["estimated_spilled_passengers"] > 0


def test_unmet_demand_is_always_labelled_as_a_model_estimate():
    a = assess(make_agg(), make_agg())
    assert a.signal("unmet").detail["basis"] == "model_estimate"


# ---------------------------------------------------------------------------
# Degenerate input must not raise
# ---------------------------------------------------------------------------

def test_missing_baseline_marks_partial_and_blocks_strong():
    a = assess(make_agg(passengers=2_200_000, seats=2_500_000), None)
    assert a.data_quality == "partial"
    assert a.verdict != "STRONG"
    assert not a.signal("growth").available
    assert not a.signal("constraint").available
    assert a.notes


def test_zero_traffic_does_not_raise():
    a = assess(make_agg(departures=0, passengers=0, seats=0), None)
    assert a.facts["load_factor_pct"] is None
    assert a.score >= 0


def test_zero_baseline_traffic_does_not_raise():
    a = assess(make_agg(), make_agg(departures=0, passengers=0, seats=0))
    assert a.signal("growth").value is None
    assert a.data_quality == "partial"


def test_thin_baseline_is_flagged_partial():
    """A 6x expansion is a step-change in service, not a growth trend."""
    baseline = make_agg(departures=800, passengers=100_000, seats=130_000)
    current = make_agg(departures=5_000, passengers=600_000, seats=750_000)
    a = assess(current, baseline)
    assert a.data_quality == "partial"
    assert any("step-change" in n or "below the" in n for n in a.notes)


def test_eligibility_gate_uses_configured_floors():
    assert kpis.is_eligible({"departures": 50_000, "passengers": 5_000_000})
    assert not kpis.is_eligible({"departures": 10, "passengers": 5_000_000})
    assert not kpis.is_eligible({"departures": 50_000, "passengers": 100})
    assert kpis.eligibility_reason({"departures": 10, "passengers": 5_000_000})
    assert kpis.eligibility_reason({"departures": 50_000, "passengers": 5_000_000}) is None


# ---------------------------------------------------------------------------
# Haul mix
# ---------------------------------------------------------------------------

def test_lognormal_tail_share_is_bounded_and_ordered():
    for cv in (0.3, 0.6, 0.9):
        share = kpis.lognormal_tail_share(1500.0, 2175.0, cv)
        assert 0.0 <= share <= 1.0
    # A cohort whose mean sits well above the threshold has more of its mass above it.
    assert (
        kpis.lognormal_tail_share(4000.0, 2175.0, 0.6)
        > kpis.lognormal_tail_share(1000.0, 2175.0, 0.6)
    )


def test_haul_mix_range_always_contains_the_point_estimate():
    """The tail share is non-monotonic in CV, so the band must be an envelope."""
    for dom_stage, intl_stage in ((800, 3000), (1458, 4323), (2500, 5000), (400, 600)):
        agg = make_agg(
            departures=50_000, domestic_departures=42_000,
            mean_stage=(42_000 * dom_stage + 8_000 * intl_stage) / 50_000,
            domestic_stage=dom_stage,
        )
        mix = kpis.haul_mix_from_cohorts(agg, 2175.0)
        lo, hi = mix["long_haul_share_pct_range"]
        assert lo <= mix["long_haul_share_pct_estimate"] <= hi


def test_haul_mix_floor_never_exceeds_the_estimate():
    agg = make_agg(departures=50_000, domestic_departures=42_000,
                   mean_stage=1900.0, domestic_stage=1458.0)
    mix = kpis.haul_mix_from_cohorts(agg, 2175.0)
    assert mix["robust_floor_pct"] <= mix["long_haul_share_pct_estimate"]


def test_haul_mix_is_always_labelled_an_estimate_with_its_assumptions():
    agg = make_agg(departures=50_000, domestic_departures=42_000)
    mix = kpis.haul_mix_from_cohorts(agg, 2175.0)
    assert mix["basis"] == "cohort_estimate"
    assert len(mix["assumptions"]) >= 3


def test_haul_mix_handles_an_all_domestic_airport():
    agg = make_agg(departures=10_000, domestic_departures=10_000,
                   mean_stage=700.0, domestic_stage=700.0)
    mix = kpis.haul_mix_from_cohorts(agg, 2175.0)
    assert mix["robust_floor_pct"] == 0.0
    assert len(mix["cohorts"]) == 1


def test_haul_mix_reports_unavailable_with_no_departures():
    mix = kpis.haul_mix_from_cohorts(make_agg(departures=0), 2175.0)
    assert mix["basis"] == "unavailable"


def test_raising_the_threshold_cannot_increase_the_long_haul_share():
    agg = make_agg(departures=50_000, domestic_departures=42_000,
                   mean_stage=1900.0, domestic_stage=1458.0)
    shares = [
        kpis.haul_mix_from_cohorts(agg, t)["long_haul_share_pct_estimate"]
        for t in (1000.0, 2175.0, 4000.0, 8000.0)
    ]
    assert shares == sorted(shares, reverse=True)


# ---------------------------------------------------------------------------
# Serialisation contract -- the agent depends on these keys
# ---------------------------------------------------------------------------

def test_assessment_serialises_with_the_keys_the_tools_expect():
    payload = assess(make_agg(), make_agg(passengers=1_500_000)).to_dict()
    for key in ("airport", "score", "verdict", "signals", "facts",
                "verdict_gates", "data_quality", "scoring_version"):
        assert key in payload
    assert payload["scoring_version"] == config.SCORING_VERSION
    for signal in payload["signals"]:
        for key in ("signal", "label", "value", "subscore", "weight", "contribution"):
            assert key in signal


def test_congestion_ranking_orders_and_numbers_its_rows():
    quiet = assess(make_agg(passengers=1_700_000, seats=2_500_000, airport="AAA"),
                   make_agg(passengers=1_700_000, seats=2_500_000))
    busy = assess(make_agg(passengers=2_200_000, seats=2_500_000, airport="BBB"),
                  make_agg(passengers=1_600_000, seats=2_400_000))
    rows = kpis.congestion_ranking([quiet, busy])
    assert rows[0]["airport"] == "BBB"
    assert [r["congestion_rank"] for r in rows] == [1, 2]
