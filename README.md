# Airport Investment Intelligence Agent

An AI agent for a firm investing in US airport modernisation. It identifies airports
where a terminal renovation would be most profitable, using deterministic scoring over
public aviation data with a conversational interface on top.

**Live demo:** <https://airport-investment-agent-2l5f.onrender.com/>
(hosted on Render's free tier — the first request after idling can take ~30s to wake up)

See **[DESIGN.md](DESIGN.md)** for the scoring methodology, tradeoffs, and where AI is
and is not used.

```
Which airports in New England are strong candidates for terminal expansion?
Compare LA and Santa Ana airport congestion levels.
What is the percentage of long haul flights out of Anchorage airport?
What is the unmet flight demand in SFO airport and why?
```

---

## Setup

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

Populate the local data cache (one-time, ~30s — pulls 131,739 rows from BTS and the
OurAirports reference file):

```bash
python -m app.ingest --source all
```

This step is optional: the server builds the cache itself on startup if it is empty,
which is what makes the app deployable to a host with no persistent disk. Running it
up front just moves the ~30s off the first boot.

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Run it:

```bash
python -m uvicorn app.main:app --port 8000
```

Open <http://localhost:8000>.

---

## What it does

Every answer is built from **deterministic scoring** — the model chooses which analysis
to run and explains the result, but computes none of the numbers. Four signals are
scored against a pre-pandemic baseline:

| Signal | Measures | Weight |
|---|---|---:|
| Utilisation | how full departing aircraft are | 0.25 |
| Demand growth | passenger CAGR vs a rolling 3-year baseline | 0.30 |
| **Capacity constraint** | upgauging vs added frequency | 0.30 |
| Unmet demand | modelled spill | 0.15 |

The capacity-constraint signal is the differentiator. Load factor alone measures airline
scheduling, not terminal capacity; carriers flying **bigger aircraft while flight counts
stay flat** is the fingerprint of an airport that has run out of gates or slots — the
condition a renovation actually relieves.

The right-hand panel shows every tool call and the model's reasoning, so you can see
which data produced each claim.

---

## Voice input

The microphone button in the composer dictates a question via the browser's built-in
`SpeechRecognition` API — no audio leaves the browser, no extra API key. The transcript
lands in the input box rather than sending itself, because recognition mishears airport
codes often enough that a silent auto-send is a real risk.

A deterministic pass repairs what dictation reliably breaks before the text is handed
over: `s f o` becomes `SFO`, `lacks` becomes `LAX`, `seatac` becomes `SEA`. Ordinary
sentences pass through untouched.

Chrome or Edge only, over HTTPS or localhost. Elsewhere the button is not rendered.

---

## Data sources

All public, no API keys.

| Source | Used for |
|---|---|
| [BTS T-100 Segment Summary by Origin Airport](https://data.bts.gov/resource/r495-tyji.json) | traffic, capacity, stage length (131,739 rows, 2014 → present) |
| [OurAirports](https://davidmegginson.github.io/ourairports-data/airports.csv) | region, city and metro resolution |
| [FAA NAS Status](https://nasstatus.faa.gov/api/airport-status-information) | live ground stops and delay programs |

---

## Optional: exact long-haul figures

BTS publishes a *mean stage length* per airport, not per-flight distances, so
`haul_mix` answers with a clearly-labelled cohort estimate plus an uncertainty range
(see [DESIGN.md §3](DESIGN.md)). To get exact figures instead, download a **T-100
Segment** export from
[BTS TranStats](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIM)
— include `ORIGIN`, `DEST`, `DISTANCE`, `DEPARTURES_PERFORMED`, `SEATS`, `PASSENGERS`,
`YEAR`, `MONTH` — then:

```bash
python -m app.ingest --source segments
```

Drop the CSV in `cache/segments/` first. The tool switches to `basis: "exact"`
automatically and says so in its answer.

---

## Tests

```bash
python -m pytest
```

88 tests, ~3 seconds, no API key or network required. Scoring invariants run against
synthetic aggregates; resolution and the four exam questions run against the local
cache; the agent loop runs against a fake SDK client.

---

## Project layout

```
app/
  config.py           weights, anchor tables, thresholds, SCORING_VERSION
  prompt.py           system prompt + tool schemas — everything the model reads
  agent.py            claude-sonnet-5 tool-use loop with streaming
  tools.py            the 5 tools exposed to the model
  kpis.py             pure deterministic scoring — no I/O
  airports.py         free-text → IATA code, metro and region names
  store.py            SQLite: schema, connections, all queries
  ingest.py           BTS + OurAirports + optional segment CSVs → SQLite
  faa.py              live FAA status
  main.py             FastAPI server
web/index.html        chat UI + browser dictation (no build step)
tests/                88 tests
DESIGN.md             methodology, tradeoffs, where AI is used
```

The two axes: `store.py` reads and `ingest.py` writes, `prompt.py` is what the model
is told and `tools.py` is what it can do. `kpis.py` never touches I/O, and the model
never touches arithmetic.

Useful commands:

```bash
python -m app.ingest
```

```bash
python -m app.airports "Santa Ana"
```

```bash
curl -s localhost:8000/health | python -m json.tool
```

---

## Notes and limitations

- Unmet demand is a **model estimate**, not a measurement — no public dataset reports
  passengers turned away. The agent says so whenever it quotes one.
- T-100 runs roughly two months behind. Live FAA status is shown as corroboration but
  is never scored: a ground stop is today's weather, not structural capacity.
- Upgauging has confounders (regional-jet retirement, pilot shortage). The model
  identifies candidates for investigation, not finished theses.
- This ranks where demand presses against capacity. It says nothing about construction
  cost, land availability or local politics — one input to a decision, not the decision.
- Sessions are held in memory and are lost on restart.
- Voice is input only — spoken answers are not implemented.
