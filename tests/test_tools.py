"""
The four exam questions, end to end through the tool layer.

These assert the shape and internal consistency of each answer rather than exact
figures, so they keep working when BTS publishes a new month. Where a number is
pinned it is one that should be stable (Anchorage is overwhelmingly domestic;
LAX is busier than SNA).
"""

from __future__ import annotations

import pytest

from app import config, prompt, store, tools

pytestmark = pytest.mark.skipif(
    not store.bts_populated(),
    reason="local cache not populated; run python -m app.ingest --source all",
)


@pytest.fixture(scope="module")
def con():
    connection = store.connect()
    yield connection
    connection.close()


# ---------------------------------------------------------------------------
# Q1: "Which airports in New England are strong candidates for terminal expansion?"
# ---------------------------------------------------------------------------

def test_q1_new_england_ranking(con):
    result = tools.dispatch(con, "rank_expansion_candidates", {"region": "New England"})

    assert "error" not in result
    assert result["scope"] == "New England"
    assert result["ranking"], "expected at least one eligible New England airport"

    codes = {row["airport"] for row in result["ranking"]}
    assert codes <= {"BOS", "BDL", "PVD", "PWM", "MHT", "BTV", "BGR", "HVN", "ORH", "ACK", "HYA"}

    scores = [row["score"] for row in result["ranking"]]
    assert scores == sorted(scores, reverse=True), "ranking must be ordered by score"

    for row in result["ranking"]:
        assert row["verdict"] in {"STRONG", "MODERATE", "WEAK"}
        assert row["why"], "every ranked airport needs a stated rationale"
        assert row["signals"] and row["key_facts"]

    # The gate must be stated, and what it removed must be visible.
    assert result["eligibility_rule"]
    assert result["excluded_count"] > 0
    assert result["methodology"]["key_signal"]
    assert result["scoring_version"] == config.SCORING_VERSION


def test_q1_rejects_an_unknown_region_with_a_usable_list(con):
    result = tools.dispatch(con, "rank_expansion_candidates", {"region": "Atlantis"})
    assert "error" in result
    assert "new england" in result["known_regions"]


def test_q1_national_ranking_respects_the_limit(con):
    result = tools.dispatch(con, "rank_expansion_candidates", {"limit": 5})
    assert len(result["ranking"]) == 5
    assert result["scope"] == "United States"
    assert result["eligible_count"] > 100


def test_q1_accepts_explicit_states(con):
    result = tools.dispatch(con, "rank_expansion_candidates", {"states": ["MA", "CT"]})
    assert "error" not in result
    assert {r["airport"] for r in result["ranking"]} <= {"BOS", "BDL", "HVN", "ORH", "ACK", "HYA", "BDR"}


# ---------------------------------------------------------------------------
# Q2: "Compare LA and Santa Ana airport congestion levels."
# ---------------------------------------------------------------------------

def test_q2_la_versus_santa_ana(con):
    result = tools.dispatch(con, "compare_airports", {"airports": ["LA", "Santa Ana"]})

    assert "error" not in result
    resolved = {r["query"]: r["code"] for r in result["resolved"]}
    assert resolved == {"LA": "LAX", "Santa Ana": "SNA"}

    ranking = result["congestion_ranking"]
    assert len(ranking) == 2
    assert [r["congestion_rank"] for r in ranking] == [1, 2]
    # LAX is materially larger and fuller; this ordering should be stable.
    assert ranking[0]["airport"] == "LAX"
    assert ranking[0]["congestion_index"] > ranking[1]["congestion_index"]

    assert "LAX" in result["congestion_verdict"]
    assert result["congestion_index_definition"]
    assert set(result["detail"]) == {"LAX", "SNA"}

    # The metro ambiguity must be surfaced, not swallowed.
    la = next(r for r in result["resolved"] if r["query"] == "LA")
    assert la.get("note") and "alternatives" in la


def test_q2_requires_at_least_two_airports(con):
    assert "error" in tools.dispatch(con, "compare_airports", {"airports": ["LAX"]})
    assert "error" in tools.dispatch(con, "compare_airports", {"airports": []})


def test_q2_live_status_never_breaks_the_comparison(con):
    result = tools.dispatch(con, "compare_airports", {"airports": ["LAX", "SNA"]})
    live = result["live_faa_status"]
    assert "available" in live
    if live["available"]:
        assert "airports_under_flow_control" in live


# ---------------------------------------------------------------------------
# Q3: "What is the percentage of long haul flights out of Anchorage airport?"
# ---------------------------------------------------------------------------

def test_q3_anchorage_haul_mix(con):
    result = tools.dispatch(con, "haul_mix", {"airport": "Anchorage"})

    assert "error" not in result
    assert result["resolved"]["code"] == "ANC"
    assert result["threshold_miles"] == config.LONG_HAUL_THRESHOLD_MILES

    answer = result["answer"]
    assert answer["basis"] in {"exact", "cohort_estimate"}

    if answer["basis"] == "cohort_estimate":
        lo, hi = answer["long_haul_share_pct_range"]
        assert lo <= answer["long_haul_share_pct_estimate"] <= hi
        assert answer["robust_floor_pct"] <= answer["long_haul_share_pct_estimate"]
        # The uncertainty must be stated in the answer text itself, not just in a field.
        assert "ESTIMATE" in answer["statement"]

    tier1 = result["tier_1_cohort_estimate"]
    cohorts = {c["cohort"]: c for c in tier1["cohorts"]}
    assert set(cohorts) == {"domestic", "international"}
    # Anchorage is overwhelmingly domestic, and its international flying is long.
    assert cohorts["domestic"]["share_of_departures_pct"] > 70
    assert cohorts["international"]["mean_stage_miles"] > cohorts["domestic"]["mean_stage_miles"]
    assert cohorts["international"]["mean_exceeds_threshold"]
    assert tier1["assumptions"]


def test_q3_threshold_is_configurable_and_monotonic(con):
    shares = []
    for threshold in (1000, 2175, 5000):
        result = tools.dispatch(con, "haul_mix", {"airport": "ANC", "threshold_miles": threshold})
        assert result["threshold_miles"] == threshold
        shares.append(result["tier_1_cohort_estimate"]["long_haul_share_pct_estimate"])
    assert shares == sorted(shares, reverse=True)


def test_q3_invalid_threshold_falls_back_to_the_default(con):
    result = tools.dispatch(con, "haul_mix", {"airport": "ANC", "threshold_miles": -5})
    assert result["threshold_miles"] == config.LONG_HAUL_THRESHOLD_MILES


# ---------------------------------------------------------------------------
# Q4: "What is the unmet flight demand in SFO airport and why?"
# ---------------------------------------------------------------------------

def test_q4_sfo_unmet_demand_with_reasons(con):
    result = tools.dispatch(con, "unmet_demand", {"airport": "SFO"})

    assert "error" not in result
    assert result["airport"] == "SFO"
    assert result["basis"] == "model_estimate"
    assert isinstance(result["unmet_demand_index_pct_of_passengers"], (int, float))

    # The "why" is the point of the question.
    assert result["why"], "unmet demand must always come with an explanation"
    assert all(isinstance(reason, str) and reason for reason in result["why"])

    components = result["components"]
    for key in (
        "spill_from_high_load_factor_pct",
        "estimated_spilled_passengers",
        "demand_outrunning_capacity_pct_per_year",
    ):
        assert key in components

    assert "MODEL ESTIMATE" in result["caveat"]
    assert result["supporting_signals"]["utilization"]
    assert result["scoring_version"] == config.SCORING_VERSION


def test_q4_index_matches_its_components(con):
    result = tools.dispatch(con, "unmet_demand", {"airport": "BTV"})
    components = result["components"]
    total = (
        components["spill_from_high_load_factor_pct"]
        + components["demand_outrunning_capacity_pct_per_year"]
    )
    assert result["unmet_demand_index_pct_of_passengers"] == pytest.approx(total, abs=0.02)


# ---------------------------------------------------------------------------
# Profile tool and cross-cutting contracts
# ---------------------------------------------------------------------------

def test_airport_profile_returns_a_complete_assessment(con):
    result = tools.dispatch(con, "airport_profile", {"airport": "BOS"})
    assert "error" not in result
    assert result["airport"] == "BOS"
    assert len(result["signals"]) == 4
    assert result["verdict"] in {"STRONG", "MODERATE", "WEAK"}
    assert result["meets_eligibility_floor"] is True
    assert len(result["monthly_trend"]) == config.TTM_MONTHS
    assert result["verdict_gates"]


def test_profile_trend_can_be_suppressed(con):
    result = tools.dispatch(con, "airport_profile", {"airport": "BOS", "include_trend": False})
    assert "monthly_trend" not in result


def test_every_tool_reports_its_data_window_and_scoring_version(con):
    for name, args in [
        ("rank_expansion_candidates", {"limit": 3}),
        ("compare_airports", {"airports": ["LAX", "SNA"]}),
        ("airport_profile", {"airport": "SFO"}),
        ("haul_mix", {"airport": "ANC"}),
        ("unmet_demand", {"airport": "SFO"}),
    ]:
        result = tools.dispatch(con, name, args)
        assert "data_window" in result, f"{name} omits data_window"
        window = result["data_window"]
        assert window["years_between"] == float(config.BASELINE_LOOKBACK_YEARS)
        assert window["prepandemic_reference_window_ends"] == config.REFERENCE_BASELINE_END
        # The scored baseline rolls with the data, so assert the relationship
        # rather than a hardcoded month.
        assert store.month_index(window["baseline_window_ends"]) == (
            store.month_index(result["data_window"]["current_window"].split(" .. ")[1])
            - 12 * config.BASELINE_LOOKBACK_YEARS
        )


def test_tool_schemas_match_the_dispatch_table():
    """The schemas live in app/prompt.py, the callables in app/tools.py.

    Split across two modules they can drift, so the contract is asserted here:
    a schema with no implementation is a tool the model will call and get an
    "Unknown tool" back from.
    """
    from app.tools import _DISPATCH  # noqa: PLC0415
    assert {s["name"] for s in prompt.TOOL_SCHEMAS} == set(_DISPATCH)
    for schema in prompt.TOOL_SCHEMAS:
        assert schema["description"].strip()
        assert schema["input_schema"]["type"] == "object"
        for prop in schema["input_schema"]["properties"].values():
            assert prop.get("description"), "every tool parameter needs a description"


def test_dispatch_never_raises(con):
    for name, args in [
        ("does_not_exist", {}),
        ("haul_mix", {"bogus_argument": 1}),
        ("airport_profile", {"airport": "Nowheresville"}),
        ("compare_airports", {"airports": ["Nowhere", "Neither"]}),
        ("unmet_demand", {"airport": ""}),
    ]:
        result = tools.dispatch(con, name, args)
        assert isinstance(result, dict)
        assert "error" in result
