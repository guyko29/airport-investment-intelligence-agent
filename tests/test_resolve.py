"""
Airport and region resolution.

These run against the local cache. The exam questions use colloquial names ("LA",
"Santa Ana", "New England") rather than IATA codes, so resolution is on the
critical path for every one of them.
"""

from __future__ import annotations

import pytest

from app import config
from app.data import airports as airports_data
from app.data import bts

pytestmark = pytest.mark.skipif(
    not bts.is_populated(),
    reason="local cache not populated; run python -m app.data.bts --refresh",
)


@pytest.fixture(scope="module")
def con():
    connection = bts.connect()
    yield connection
    connection.close()


# ---------------------------------------------------------------------------
# Airport resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query,expected",
    [
        ("SFO", "SFO"),
        ("sfo", "SFO"),
        ("LA", "LAX"),
        ("Los Angeles", "LAX"),
        ("Santa Ana", "SNA"),
        ("Orange County", "SNA"),
        ("Anchorage", "ANC"),
        ("Boston", "BOS"),
        ("boston logan", "BOS"),
        ("Chicago", "ORD"),
    ],
)
def test_resolves_exam_style_references(con, query, expected):
    assert airports_data.resolve(con, query).code == expected


def test_metro_query_surfaces_its_alternatives(con):
    """'LA' must not silently become LAX -- the answer depends on which airport."""
    resolved = airports_data.resolve(con, "LA")
    assert resolved.code == "LAX"
    assert resolved.how == "metro_alias"
    alt_codes = {a["code"] for a in resolved.alternatives}
    assert {"BUR", "LGB", "ONT", "SNA"} <= alt_codes
    assert resolved.note and "LAX" in resolved.note


def test_single_airport_metro_has_no_spurious_alternatives(con):
    resolved = airports_data.resolve(con, "Santa Ana")
    assert resolved.code == "SNA"
    assert resolved.alternatives == []


def test_unresolvable_query_fails_with_guidance_rather_than_a_guess(con):
    resolved = airports_data.resolve(con, "Nowheresville")
    assert not resolved.ok
    assert resolved.code is None
    assert resolved.note and "IATA" in resolved.note


def test_empty_query_is_handled(con):
    assert not airports_data.resolve(con, "").ok
    assert not airports_data.resolve(con, "   ").ok


def test_resolution_serialises_for_tool_output(con):
    payload = airports_data.resolve(con, "LA").to_dict()
    for key in ("query", "code", "name", "matched_by"):
        assert key in payload


def test_airport_suffixes_are_stripped(con):
    assert airports_data.resolve(con, "Anchorage airport").code == "ANC"
    assert airports_data.resolve(con, "Boston.").code == "BOS"


# ---------------------------------------------------------------------------
# Region resolution
# ---------------------------------------------------------------------------

def test_new_england_maps_to_its_six_states():
    states, label = airports_data.resolve_region("New England")
    assert set(states) == {"ME", "NH", "VT", "MA", "RI", "CT"}
    assert label == "New England"


@pytest.mark.parametrize(
    "query", ["new england", "New England", "NEW ENGLAND", "new-england", "New England region"]
)
def test_region_lookup_is_forgiving_about_form(query):
    assert airports_data.resolve_region(query) is not None


def test_unknown_region_returns_none_rather_than_an_empty_result():
    assert airports_data.resolve_region("Atlantis") is None


def test_new_england_airports_include_the_expected_commercial_fields(con):
    states, _ = airports_data.resolve_region("New England")
    codes = set(airports_data.airports_in_states(con, states))
    assert {"BOS", "BDL", "PVD", "PWM", "MHT", "BTV", "BGR"} <= codes
    # And nothing from outside the region.
    assert "LAX" not in codes and "JFK" not in codes


def test_every_configured_region_resolves_to_real_airports(con):
    for name in airports_data.known_region_names():
        states, _ = airports_data.resolve_region(name)
        assert states, f"region {name} has no states"
        assert airports_data.airports_in_states(con, states), f"region {name} has no airports"


def test_state_lookup_round_trips(con):
    assert airports_data.state_of(con, "BOS") == "MA"
    assert airports_data.state_of(con, "LAX") == "CA"
    assert airports_data.state_of(con, "ZZZ") is None


def test_config_regions_use_valid_two_letter_codes():
    for name, states in config.REGIONS.items():
        for state in states:
            assert len(state) == 2 and state.isupper(), f"{name} has a bad code: {state}"
