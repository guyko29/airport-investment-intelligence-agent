# Design & Architecture

An agent that helps identify US airports where terminal renovation would be most
profitable, on the thesis that the return comes from unlocking constrained flight
and passenger capacity.

---

## 1. The core analytical idea

The obvious metric for "is this airport full" is **load factor** — passengers
divided by seats. It is also the wrong one to build a renovation thesis on, because
load factor measures how full the *aircraft* are, which is an airline scheduling
decision. An airline can raise load factor by cutting frequency. Nothing about a
high load factor says the *terminal* is out of room.

The signal that does is **upgauging**.

When carriers want more capacity at an airport, they add flights first — it is
cheaper, more flexible, and better for passengers. They switch to bigger aircraft
instead when they *cannot* add flights, because gates, slots or runway capacity are
saturated. So:

> **seats per departure rising while departures stay flat = the airport has run out
> of room.**

That is precisely the condition where a terminal expansion converts into revenue,
and it is measurable from public data:

```
constraint_index = growth_in_seats_per_departure − growth_in_departures   (pp/yr)
```

Positive means capacity is being added through aircraft size rather than frequency.

This is the differentiator of the model. Everything else — utilisation, growth,
unmet demand — is context that stops the constraint signal from being read naively.

### Why the baseline choice is load-bearing

The scored baseline is `BASELINE_LOOKBACK_YEARS = 3` — a **rolling** window, currently
the TTM ending 2023-04. The fixed pre-pandemic window (`REFERENCE_BASELINE_END =
"2019-12"`) is still computed and reported alongside every growth figure, but it is
**not scored**.

The reasoning for rolling: COVID recovery was a common shock across essentially every
US airport rather than an airport-specific effect, and a recent window tracks what
carriers are doing *now* rather than cumulative drift since a year that keeps
receding.

The cost is real and is not hypothetical. Nationally:

| TTM ending | Departures | Passengers | Seats/departure |
|---|---:|---:|---:|
| 2019-04 | 10,080,597 | 918.2M | 111.4 |
| 2023-04 | 9,114,947 | 895.3M | 120.2 |
| 2026-04 | 9,939,791 | 978.6M | 124.1 |

Between 2023 and 2026 departures rose 9% while seats/departure rose only 3% — so
against the rolling baseline the constraint index reads **negative for 77% of
eligible airports**. Against 2019 it reads positive for 75% of them, because
departures are still below 2019 while gauge rose 11%. Same airports, opposite sign.
That difference is post-COVID frequency restoration, and it biases the scored
constraint signal downward.

Two consequences the model has to live with, both documented at the point of use:

1. **The constraint sub-score is now explicitly relative.** Anchors are percentile
   calibrated, so 50 points means "more constrained than the median US airport", not
   "constrained in absolute terms" — the median is −3.3 pp/yr. An airport can score
   above 50 on constraint while its raw index is negative. The raw pp/yr value is
   reported next to the sub-score precisely so the absolute claim stays checkable.
2. **The window will eventually reach the 2020–21 collapse.** When it does, growth and
   constraint will both read off a depressed base and overstate. That needs revisiting
   before the lookback window starts in 2020.

Because the window rolls, anchors are no longer frozen for all time: they are
calibrated to an observed distribution that moves with the data vintage, and must be
re-measured when it moves materially.

---

## 2. Scoring methodology

### Analysis unit

Trailing-12-month (TTM) aggregates, compared against the **matched** 12-month window
three years earlier. Matched windows mean the month mix is identical on both sides,
so seasonality cancels rather than contaminating the growth rate.

Stage-length figures are **departure-weighted** when aggregated across months —
`total_distance_flight_sm` is a mean per departure, not a total, so summing it would
weight a 100-departure month equally with a 15,000-departure one.

### Eligibility gate

`TTM departures ≥ 1,200` **and** `TTM passengers ≥ 250,000` — 185 of 1,319 airports
with traffic qualify. Deliberately traffic-based rather than OurAirports' `type`
field, which labels GA fields such as BED, BVY and OWD as `medium_airport`. Rankings
report how many airports the gate excluded, so the scope of the answer is visible.

### The four scored signals

| # | Signal | Measures | Weight |
|---|---|---|---:|
| 1 | Utilisation | TTM passengers / TTM seats | 0.25 |
| 2 | Demand growth | Passenger CAGR vs the rolling 3-year baseline | 0.30 |
| 3 | **Capacity constraint** | Gauge growth − departure growth | 0.30 |
| 4 | Unmet demand | Modelled spill, as % of passengers | 0.15 |

A fifth measure, **scale** (TTM passengers), is used only as the eligibility gate and
a tiebreaker. It is deliberately *not* a score component — otherwise the ranking is
just "biggest airports first" wearing a KPI costume.

**Unmet demand** is the sum of two components, both reported separately so the "why"
is answerable:
- *Spill*: `seats × max(load_factor − 83%, 0)`, on the reasoning that a system-wide
  annual average above ~83% means peak-period flights are effectively sold out.
- *Growth gap*: `max(passenger_CAGR − seat_CAGR, 0)` — demand outrunning capacity.

### Anchors, not z-scores

Each signal maps to 0–100 through a **fixed piecewise-linear anchor curve**, calibrated
once against the observed national distribution and then frozen:

| Signal | p05 | p25 | median | p75 | p95 |
|---|---:|---:|---:|---:|---:|
| Utilisation (%) | — | 76.0 | 78.6 | 81.1 | — |
| Growth (%/yr) | −2.1 | +0.2 | +1.6 | +4.1 | +8.0 |
| Constraint (pp/yr) | −4.3 | +0.1 | +2.0 | +4.0 | +7.0 |
| Unmet (% of pax) | 0 | 0 | 0 | +0.1 | +1.9 |

Peer-relative (z-score) scoring was rejected because it makes a single airport's score
meaningless in isolation — "SFO scores 74" would depend on who else was in the query —
and makes scores shift when the eligibility filter or region changes the peer set. The
cost of fixed anchors is that they encode a judgement call, which is why they sit in
one readable table in `config.py` rather than buried in the scoring code.

Unmet demand is zero for 73% of airports. That is intended: spill is the specific
condition that justifies expansion, so it should be a discriminator, not a
participation trophy.

### Composite and verdict

```
score = 0.25·utilisation + 0.30·growth + 0.30·constraint + 0.15·unmet
```

`STRONG ≥ 70`, `MODERATE ≥ 50`, else `WEAK` — plus **hard gates**: STRONG additionally
requires load factor ≥ 80% *and* positive growth. Without those, a high constraint
index alone could promote a shrinking airport, but upgauging on a declining airport is
fleet rationalisation, not capacity pressure. When a score clears 70 and the gates fail,
the airport is held at MODERATE and the reason is stated in `notes`.

Result across 185 eligible airports: 7 STRONG, 39 MODERATE, 139 WEAK (median 41.6).

### Baseline adequacy

A CAGR against a very small or structurally different baseline measures a *step-change
in service*, not a trend. Two airports currently trip this — PVU (new terminal plus new
carrier bases, 6× departures since 2019) and HVN (became an Avelo base, 5×). Their
figures are arithmetically fine and analytically misleading, so they are flagged
`data_quality: "partial"` with an explanatory note rather than hidden or reported
as if they were trends.

---

## 3. The Anchorage problem, and what honesty costs

The exam asks for "the percentage of long haul flights out of Anchorage". **No free
public API publishes per-flight distances.** BTS publishes a *mean stage length* per
airport-month, and a mean cannot be inverted into a distribution.

Four candidate sources were probed and rejected, each verified dead:

| Source | Result |
|---|---|
| OpenSky `/flights/departure` | 403 — now requires OAuth2 registration |
| BTS TranStats `/PREZIP` bulk | Only stale 2015 files remain |
| BTS ArcGIS `T100_Domestic_Market_and_Segment_Data` | Origin-aggregated; no destination, no distance |
| Socrata catalog search on `data.bts.gov` | Returns federated noise, not BTS tables |

Rather than quietly reporting a mean as though it were a distribution, `haul_mix`
answers in **two tiers** and always states which one it used.

**Tier 1 — always available.** Decompose the two cohorts BTS *does* publish. For
Anchorage (TTM through 2026-04):

```
total departures       86,373
  domestic             73,046  (84.6%)   mean stage 1,458 mi
  international        13,327  (15.4%)   mean stage 4,323 mi   ← derived as total − domestic
```

Two cohorts of known size and known mean, on opposite sides of the 2,175-mile
(3,500 km) long-haul boundary. Converting that into a percentage needs a within-cohort
distribution, so stage lengths are modelled as **lognormal** — route-length
distributions are right-skewed and strictly positive, which is the lognormal shape —
and the coefficient of variation is swept from 0.40 to 0.85 to produce an uncertainty
range. Output: **26.3% (range 23.8–26.3%)**, with a **robust floor of 15.4%** that
assumes nothing at all about the distribution, since it counts only the cohort whose
mean already exceeds the threshold.

One subtlety worth noting: the tail share is *not monotonic* in the coefficient of
variation. Raising spread moves mass above the threshold for the domestic cohort
(mean below it) and below the threshold for the international cohort (mean above it).
So the range is computed as the envelope over all three CVs rather than assuming the
endpoints bracket the point estimate — there is a regression test pinning this.

**Tier 2 — exact, when available.** Drop a BTS T-100 Segment CSV into `cache/segments/`
and run one command; the same tool then computes the true per-route distance
distribution and reports `basis: "exact"`. The reader tolerates the header-naming
variants between the TranStats UI export and the prezipped files.

The agent is instructed to state the basis in its answer either way. **This is the
part of the build I would defend hardest in review**: the alternative was a number
that looks authoritative and isn't.

---

## 4. Where AI is used — and where it deliberately is not

| Layer | Who does it | Why |
|---|---|---|
| Understanding the question | **Model** | "LA and Santa Ana" → two airports to compare |
| Choosing a tool | **Model** | Routing on tool descriptions |
| Resolving free text → IATA | Code | Deterministic, testable, and must surface ambiguity |
| **Every number** | **Code** (`kpis.py`) | Reproducible, unit-tested, auditable |
| Explaining the result | **Model** | Reads the raw inputs the tool returned alongside each figure |

**The model performs no arithmetic.** It has no airport data outside the tools, and
the system prompt says explicitly that a number produced without a tool is fabricated.
Every figure it can quote was computed by `kpis.py` and arrived in a tool result
carrying the raw inputs behind it. That is what makes the analysis reproducible while
the conversation stays natural — and it is why `kpis.py` is pure functions with no I/O,
so all 28 scoring invariants run in 0.07s without a network or a database.

Each tool result also carries `scoring_version`, so any figure in a transcript can be
traced to an exact set of weights, anchors and thresholds.

### Model configuration

`claude-sonnet-5`, adaptive thinking with `display: "summarized"` (so reasoning renders
in the transparency panel), `effort: "medium"`.

Sonnet over Opus because of how little is actually asked of the model here. It performs
no arithmetic and holds no domain data — it routes a question to one of five tools and
explains what came back. That is a classification-and-phrasing job, not a reasoning one,
and the hard part of the analysis lives in `kpis.py` where it is unit-tested. Against
that, this is an interactive tool where latency is felt on every turn. Opus 5 would buy
more headroom on genuinely ambiguous multi-step questions; the tool-selection guidance
in the system prompt is explicit enough that there was no observed routing difference to
pay for.

A **manual tool-use loop over `messages.stream()`** rather than the SDK's tool runner,
because the UI needs both per-token text deltas *and* interception of each tool call
to render the activity panel; the runner surfaces one or the other. The loop handles
`refusal` (checked **before** reading content, which may be empty on a refusal),
`pause_turn`, `max_tokens`, and caps at 8 tool rounds.

Prompt caching is applied to the system + tools prefix, which is byte-stable across
turns — no interpolated timestamps — so every turn after the first reads the prefix
rather than paying for it. There is a test pinning that stability.

---

## 5. Key tradeoffs

**Full local cache over live API calls.** The entire BTS table (131,739 rows) is pulled
into SQLite on setup and queried locally. Ranking touches every airport at once; doing
that over HTTP would be slow and rude. Cost: data is a snapshot until refreshed. Bought:
instant responses and a demo that survives Socrata being down.

**Fixed anchors over peer-relative scoring.** Scores stay comparable across every query
at a given data vintage, rather than shifting with whatever peer set a question selects;
the price is a judgement call, made visible in one config table. Note the anchors are
percentile-calibrated to the rolling baseline's distribution, so they are frozen *per
vintage*, not for all time — see §"Why the baseline choice is load-bearing".

**One coherent baseline over per-signal baselines.** Growth and constraint both use the
rolling 3-year window. Growth against fixed calendar 2019 is computed and *reported* —
it is genuinely informative as long-run context — but never scored, so a single window
drives every scored comparison.

**Flag rather than exclude.** Small-baseline airports (PVU, HVN) stay in the ranking with
a `partial` marker. Excluding them would hide real change; ranking them silently would
mislead.

**In-memory sessions.** No database, no auth. Right for a single-analyst tool; a
production deployment would move sessions to Redis and put the API key behind real auth.

**Congestion ≠ investment score.** `compare_airports` ranks on a separate congestion
index (0.5 utilisation + 0.3 constraint + 0.2 unmet). An airport can be congested
without being a good expansion candidate — congested but shrinking is a bad
investment — and the two questions deserve two answers.

---

## 6. Assumptions and limitations

**Stated in the output, not just here** — every estimate carries a `basis` field, and
the agent is instructed to surface it.

1. **Unmet demand is modelled, never measured.** BTS reports passengers *carried*; no
   public dataset reports passengers turned away. The estimate infers spill from load
   factor above a comfort threshold and from passenger growth outpacing seat growth.
   Measuring it directly would need fare data and schedule-level booking curves.
2. **83% is a judgement call.** The load-factor comfort threshold is calibrated so ~9%
   of eligible airports clear it. Defensible, not derived.
3. **Long-haul share is a cohort estimate** unless segment data is loaded (§3).
4. **International figures are derived** as total minus domestic; BTS publishes the
   total and domestic cohorts only.
5. **Upgauging has confounders.** Regional-jet retirement and the post-2021 pilot
   shortage push gauge up for reasons unrelated to any specific airport's gates. The
   signal identifies *candidates for investigation*, not a finished thesis — which is
   why growth and utilisation gate the verdict.
6. **~2-month reporting lag** in T-100. Live FAA status is included as corroboration
   but is explicitly never scored: a ground stop is today's weather, not capacity.
7. **Origin-airport view only.** Connecting-passenger share is invisible here, and it
   matters — a hub's terminal load is not the same as its origin traffic.
8. **No cost side.** This ranks where demand presses against capacity, not where
   construction is cheap, land is available, or politics are favourable. It is one
   input to an investment decision, not the decision.

---

## 7. Architecture

```
web/index.html          single-file chat UI, SSE, tool-transparency panel
        │
app/main.py             FastAPI: POST /chat (SSE), /health, /reset
        │
app/agent.py            claude-sonnet-5 tool-use loop, streaming, prompt caching
        │  reads app/prompt.py: system prompt + tool schemas, the cached prefix
        │  the model chooses a tool; it never computes a number
app/tools.py            5 tools: resolve free text → query → score → explain
        │
app/kpis.py             PURE deterministic scoring. No I/O. 28 invariant tests.
        │
app/airports.py         free text → IATA code, never resolving ambiguity silently
app/store.py            SQLite: one schema, all queries        app/faa.py (live)
        │                        ▲
        │               app/ingest.py — BTS + OurAirports + optional segment CSVs
        │               (the only module that talks to the network on the way in)
app/config.py           every weight, anchor, threshold + SCORING_VERSION
```

The module split follows two cuts rather than a layer stack. Data is split by
**direction** — `store.py` reads, `ingest.py` writes — because the two run at
different moments and fail for different reasons: a failing query is a bug, a
failing ingest is an upstream outage. The model's interface is split by **what the
model sees** — `prompt.py` holds the system prompt and the tool schemas together,
since they are one contract, tuned in the same sittings, and both must stay
byte-stable for the cached prefix to survive a turn.

**Test coverage: 88 tests, ~3s.** Scoring invariants (monotonicity, verdict gates,
degenerate input, haul-mix range containment), resolution (every exam-style query),
the four exam questions end-to-end, and the agent loop against a fake SDK client
(parallel tool results in one message, refusal handling, loop cap, cache stability) —
so the agent's mechanics are verified without an API key or network.
