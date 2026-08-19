"""
Central configuration for the Airport Investment Intelligence Agent.

Every tunable that affects a published number lives here, and SCORING_VERSION is
echoed in every tool result. That means any figure the agent quotes can be traced
back to an exact set of weights, anchors and thresholds -- the scoring is auditable
rather than a black box.
"""

from pathlib import Path

# Bump this whenever weights, anchors, thresholds or the composite formula change.
SCORING_VERSION = "aiia-2.0.0"   # 2.0.0: scored baseline moved 2019 -> rolling 3yr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
DB_PATH = CACHE_DIR / "bts.db"
SEGMENTS_DIR = PROJECT_ROOT / "data" / "segments"
WEB_DIR = PROJECT_ROOT / "web"

# ---------------------------------------------------------------------------
# Public data sources (no API keys required; all verified live)
# ---------------------------------------------------------------------------

# BTS T-100 Segment Summary By Origin Airport. One row per origin airport per month.
# Note: data.transportation.gov 302-redirects here, so we address data.bts.gov directly.
BTS_BASE_URL = "https://data.bts.gov/resource/r495-tyji.json"
BTS_PAGE_SIZE = 50_000

# OurAirports reference data: IATA/ICAO codes, coordinates, iso_region, municipality.
OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"

# FAA National Airspace System status: live ground stops and delay programs (XML).
FAA_NAS_STATUS_URL = "https://nasstatus.faa.gov/api/airport-status-information"

HTTP_TIMEOUT_SECONDS = 120.0

# ---------------------------------------------------------------------------
# Analysis window
# ---------------------------------------------------------------------------

# All headline figures are trailing-12-month (TTM) aggregates, and every
# comparison uses a MATCHED 12-month window so the month mix is identical on both
# sides and seasonality cancels out.
TTM_MONTHS = 12

# Scored baseline: a ROLLING lookback. Growth and capacity constraint are measured
# against the matched TTM window this many years before the current one.
#
# Rationale for rolling over a fixed pre-pandemic anchor: recovery from COVID hit
# every US airport, so it is largely a common shock rather than an airport-specific
# effect, and a recent window tracks what carriers are doing *now* rather than
# cumulative drift since a year that keeps receding.
#
# The cost is real and worth stating. Against a 2023-vintage base the constraint
# signal reads negative for 77% of eligible airports (flight restoration outrunning
# upgauging); against 2019 it reads positive for 75% of them. Switching the scored
# baseline therefore shifts the whole constraint distribution DOWN by roughly 3-4
# percentage points per year, so ANCHORS["constraint"] below is recalibrated to the
# rolling-window distribution rather than the 2019 one. Two further consequences to
# watch as time passes:
#   * When the lookback window starts in 2020-21, growth and constraint will read
#     off a collapsed base and overstate both. Revisit then.
#   * Because the window moves, a given airport's score is no longer comparable
#     across data vintages the way a fixed anchor made it.
BASELINE_LOOKBACK_YEARS = 3

# Fixed pre-pandemic window, reported alongside the scored figures as long-run
# context. NOT scored -- it answers "where is this airport relative to before the
# pandemic", which is a different question from "which way is it moving now".
REFERENCE_BASELINE_END = "2019-12"         # TTM ending Dec 2019 == calendar 2019

# ---------------------------------------------------------------------------
# Eligibility gate
# ---------------------------------------------------------------------------

# Screens out airports too small for a terminal-expansion thesis to be meaningful.
# Deliberately traffic-based rather than the OurAirports crowd-sourced `type` field,
# which labels GA fields such as BED, BVY and OWD as "medium_airport".
MIN_TTM_DEPARTURES = 1_200      # ~100 departures/month
MIN_TTM_PASSENGERS = 250_000

# ---------------------------------------------------------------------------
# Baseline adequacy
# ---------------------------------------------------------------------------

# A CAGR against a very small or very different baseline measures a step-change in
# service, not an organic trend. Two airports in the current data hit this: PVU
# (new terminal plus new carrier bases, 6x departures since 2019) and HVN (became
# an Avelo base, 5x). Their growth and constraint figures are arithmetically fine
# and analytically misleading, so we flag them rather than either hiding them or
# reporting them as if they were trends.
MIN_BASELINE_DEPARTURES = 1_000
MAX_BASELINE_EXPANSION_RATIO = 2.5

# ---------------------------------------------------------------------------
# Signal 4: unmet demand
# ---------------------------------------------------------------------------

# Load factor above which we treat the airport as spilling demand. Airlines target
# high-80s on their best routes, but a *system-wide* annual average above ~83%
# means peak-period flights are effectively sold out and some demand went unserved.
LOAD_FACTOR_COMFORT = 83.0

# ---------------------------------------------------------------------------
# Signal scoring anchors
# ---------------------------------------------------------------------------
# Each signal maps to a 0-100 sub-score through a fixed piecewise-linear curve
# defined by (raw_value, sub_score) anchor points, interpolated between and
# clamped outside.
#
# Why fixed anchors instead of z-scores against the peer set:
#   1. A single airport score stays meaningful in isolation -- "SFO scores 74"
#      means something without a comparison group.
#   2. Scores do not silently shift when the eligibility filter or region changes
#      the peer set underneath them.
#   3. Scores are comparable across queries and across time.
# The cost is that the anchors encode a judgement call, which is exactly why they
# are here in one readable table rather than buried in the scoring code.

ANCHORS = {
    # Seat utilisation, TTM passengers / TTM seats as a percentage.
    # Observed across eligible airports: p25 76.0, median 78.6, p75 81.1, max 86.5.
    "utilization": [
        (70.0, 0.0),
        (76.0, 25.0),
        (79.0, 50.0),
        (82.0, 72.0),
        (86.0, 95.0),
        (88.0, 100.0),
    ],
    # Passenger CAGR vs the rolling baseline, as a fraction (0.03 == 3%/yr).
    # Observed: p05 -1.5%, p25 +2.2%, median +4.4%, p75 +7.9%, p95 +13.0%.
    # Higher across the board than against 2019, because the rolling window still
    # captures post-COVID recovery on top of organic growth.
    "growth": [
        (-0.015, 0.0),
        (0.022, 25.0),
        (0.044, 50.0),
        (0.079, 72.0),
        (0.130, 92.0),
        (0.200, 100.0),
    ],
    # Upgauging constraint index: gauge growth minus departure growth, in
    # percentage points per year. Positive means carriers are adding capacity with
    # bigger aircraft rather than more flights.
    # Observed: p05 -14.4, p25 -7.2, median -3.3, p75 -0.3, p95 +6.3, max +16.9.
    #
    # NOTE the sign. Against the rolling baseline only 23% of eligible airports
    # score positive (against 2019 it was 75%), because restoring flight frequency
    # after COVID outruns upgauging almost everywhere. Calibrating to percentiles
    # therefore makes this signal explicitly RELATIVE: 50 points now means "more
    # capacity-constrained than the median US airport", NOT "capacity-constrained
    # in absolute terms". Read a high constraint sub-score as a ranking position,
    # and the raw pp/yr value -- which is reported alongside it -- for the absolute
    # claim.
    "constraint": [
        (-14.0, 0.0),
        (-7.2, 25.0),
        (-3.3, 50.0),
        (-0.3, 72.0),
        (6.3, 92.0),
        (11.0, 100.0),
    ],
    # Estimated unmet demand as a percentage of TTM passengers.
    # Observed: zero for 73% of airports, p95 +1.9, max +4.9. Deliberately sparse --
    # spill is exactly the condition that justifies expansion, so most airports
    # should score zero here.
    "unmet": [
        (0.0, 0.0),
        (0.3, 25.0),
        (1.0, 50.0),
        (2.0, 75.0),
        (3.5, 92.0),
        (5.0, 100.0),
    ],
}

# Anchors are calibrated once against the observed national distribution of
# eligible airports (2026-04 data vintage) and then frozen, so scores stay stable
# and comparable across queries instead of drifting with whatever peer set a
# particular question happens to select.
#
# The mapping is a fixed percentile recipe -- p05 -> 0, p25 -> 25, median -> 50,
# p75 -> 72, p95 -> 92, plus a headroom point at 100 -- applied to the observed
# distribution of each signal. Because the scored baseline now ROLLS, that
# distribution moves as the window advances, so these need re-measuring whenever
# the data vintage moves materially. Re-derive them by scoring every eligible
# airport and reading off the percentiles above.

# Sub-score assigned to a signal whose inputs are missing (no baseline traffic).
# Set explicitly rather than by interpolating the anchor curve at zero: with the
# rolling baseline, zero is no longer the neutral point of the constraint
# distribution, so interpolating there would silently REWARD missing data.
# A signal scored this way is always flagged `available: false`, and any airport
# with a missing signal is gated out of the STRONG verdict regardless.
UNAVAILABLE_SUBSCORE = 25.0

# Composite weights. Must sum to 1.0 (asserted at import).
WEIGHTS = {
    "utilization": 0.25,
    "growth": 0.30,
    "constraint": 0.30,
    "unmet": 0.15,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "WEIGHTS must sum to 1.0"
assert set(WEIGHTS) == set(ANCHORS), "WEIGHTS and ANCHORS must cover the same signals"

# ---------------------------------------------------------------------------
# Verdict bands
# ---------------------------------------------------------------------------

VERDICT_STRONG_MIN = 70.0
VERDICT_MODERATE_MIN = 50.0

# Hard gates on top of the score. Without these, a high constraint index alone
# could promote a shrinking airport to STRONG -- upgauging on a declining airport
# is fleet rationalisation, not capacity pressure.
STRONG_GATE_MIN_LOAD_FACTOR = 80.0
STRONG_GATE_MIN_GROWTH = 0.0

# ---------------------------------------------------------------------------
# Haul mix
# ---------------------------------------------------------------------------

# Tier-1 within-cohort distribution model.
#
# BTS publishes a MEAN stage length per cohort (domestic / international), not a
# distribution, so converting "mean 1,458 mi" into "% of flights over 2,175 mi"
# requires a distributional assumption. We model within-cohort stage lengths as
# lognormal -- route-length distributions are right-skewed and strictly positive,
# which is exactly the lognormal shape -- parameterised by coefficient of
# variation. CV_BAND drives the reported uncertainty range via sensitivity
# analysis, so the assumption is visible in the output rather than hidden.
HAUL_MIX_CV = 0.60
HAUL_MIX_CV_BAND = (0.40, 0.85)

# Default long-haul boundary: 3,500 km == 2,175 statute miles, the conventional
# ICAO/industry short/medium/long-haul split. Configurable per query.
LONG_HAUL_THRESHOLD_MILES = 2175.0

# ---------------------------------------------------------------------------
# Named regions -> USPS state codes (OurAirports iso_region is "US-<STATE>")
# ---------------------------------------------------------------------------

REGIONS = {
    "new england": ["ME", "NH", "VT", "MA", "RI", "CT"],
    "northeast": ["ME", "NH", "VT", "MA", "RI", "CT", "NY", "NJ", "PA"],
    "mid atlantic": ["NY", "NJ", "PA", "DE", "MD", "DC", "VA", "WV"],
    "southeast": ["NC", "SC", "GA", "FL", "AL", "MS", "TN", "KY"],
    "midwest": ["OH", "MI", "IN", "IL", "WI", "MN", "IA", "MO", "ND", "SD", "NE", "KS"],
    "great lakes": ["OH", "MI", "IN", "IL", "WI", "MN"],
    "south": ["TX", "OK", "AR", "LA", "MS", "AL", "TN", "KY"],
    "southwest": ["TX", "NM", "AZ", "NV", "OK"],
    "mountain west": ["MT", "ID", "WY", "CO", "UT", "NV", "AZ", "NM"],
    "rockies": ["MT", "ID", "WY", "CO", "UT"],
    "pacific northwest": ["WA", "OR", "ID"],
    "west coast": ["WA", "OR", "CA"],
    "california": ["CA"],
    "texas": ["TX"],
    "florida": ["FL"],
    "alaska": ["AK"],
    "hawaii": ["HI"],
}

# ---------------------------------------------------------------------------
# Metro aliases
# ---------------------------------------------------------------------------
# Colloquial names the exam questions actually use ("LA", "Santa Ana") mapped to
# a primary airport plus the other commercial fields in the same metro. Tools echo
# the alternatives so the agent can surface the ambiguity instead of silently
# picking one -- "compare LA and Santa Ana" is only answerable if the agent can
# say which LA airport it used.

METRO_ALIASES = {
    "la": ("LAX", ["BUR", "LGB", "ONT", "SNA"]),
    "l.a.": ("LAX", ["BUR", "LGB", "ONT", "SNA"]),
    "los angeles": ("LAX", ["BUR", "LGB", "ONT", "SNA"]),
    "socal": ("LAX", ["BUR", "LGB", "ONT", "SNA"]),
    "orange county": ("SNA", []),
    "santa ana": ("SNA", []),
    "john wayne": ("SNA", []),
    "nyc": ("JFK", ["LGA", "EWR"]),
    "new york": ("JFK", ["LGA", "EWR"]),
    "new york city": ("JFK", ["LGA", "EWR"]),
    "bay area": ("SFO", ["OAK", "SJC"]),
    "san francisco": ("SFO", ["OAK", "SJC"]),
    "sf": ("SFO", ["OAK", "SJC"]),
    "chicago": ("ORD", ["MDW"]),
    "washington": ("DCA", ["IAD", "BWI"]),
    "washington dc": ("DCA", ["IAD", "BWI"]),
    "dc": ("DCA", ["IAD", "BWI"]),
    "dallas": ("DFW", ["DAL"]),
    "houston": ("IAH", ["HOU"]),
    "boston": ("BOS", []),
    "anchorage": ("ANC", []),
    "miami": ("MIA", ["FLL", "PBI"]),
    "denver": ("DEN", []),
    "seattle": ("SEA", []),
    "atlanta": ("ATL", []),
    "phoenix": ("PHX", ["AZA"]),
}

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL = "claude-opus-5"
MAX_TOKENS = 8_000
EFFORT = "medium"          # low | medium | high | xhigh | max
MAX_TOOL_ITERATIONS = 8
