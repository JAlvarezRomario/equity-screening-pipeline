# Architecture

A batch data pipeline that turns about 9,900 companies' raw financial statements into comparable scores, a static
website and a self-checking forward test. It runs on one machine. This document covers the design and the
reasons for it. The docstrings are the source of truth for each function.

## Contents

- [Summary](#summary)
- [System context](#system-context)
- [Data flow](#data-flow)
- [One run, step by step](#one-run-step-by-step)
- [Data model](#data-model)
- [Layers](#layers)
- [The LLM boundary](#the-llm-boundary)
- [Quality attributes](#quality-attributes)
- [Trade-offs](#trade-offs)
- [Failure modes](#failure-modes)
- [Operations and testing](#operations-and-testing)
- [How the project is developed](#how-the-project-is-developed)

## Summary

Three ideas shape the design:

1. **The scores are deterministic and auditable; AI stays at the edge.** Every score is arithmetic on reported
   figures. An LLM grade only reorders what the lists show, and a test makes it impossible for the scoring code to
   import the LLM.
2. **The pipeline is incremental and fault tolerant.** A planner fetches each company only when something changed,
   within a per-run budget. Writes are atomic, and an optional stage that fails never blocks the site.
3. **Evaluation is honest by construction.** Each run's scores are tagged with a hash of the scoring code, and a
   frozen, append-only forward test measures the lists against the market in real time.

| Scale (Nov 2026) | |
|---|---|
| Benchmark | 9,902 tickers from 52 index and market lists, 69 countries |
| Raw cache | 9,938 companies × 482 columns (32 MB parquet) |
| Ranked table | 9,670 companies × 863 columns; 9,485 ranked |
| Code | ~10,000 lines of Python, 378 offline tests |
| LLM spend so far | $42.80 for 391 graded companies |

## System context

**Everything runs on the operator's machine, which is the trust boundary (dashed). Only three things leave it:** requests to Yahoo, company pages sent
to Gemini, and the static site uploaded to Cloudflare.

```mermaid
flowchart TB
    OP(["Operator"]) -->|"commands"| FQ
    SCH(["Windows scheduler"]) -->|"monthly: prices,<br/>forward mark"| FQ
    subgraph M["Operator's machine"]
        direction LR
        FQ["<b>fq pipeline</b><br/>Python 3.11 · uv"] <--> ST[("Local stores<br/>parquet + JSON<br/>never committed")]
    end
    FQ ---->|"HTTPS pull:<br/>statements, prices"| Y(["Yahoo Finance<br/>via yfinance"])
    FQ ---->|"page text out,<br/>grades back"| G(["Google Gemini API"])
    FQ ---->|"by hand:<br/>static HTML, no CSV"| CF(["Cloudflare Pages<br/>behind Access"])
    R(["Invited readers"]) -->|"email one-time code"| CF

    classDef ext fill:#fef3c7,stroke:#b45309,color:#4a2a04
    classDef pure fill:#dbeafe,stroke:#1d4ed8,color:#0b1f4d
    classDef store fill:#e5e7eb,stroke:#374151,color:#111827
    classDef person fill:#ede9fe,stroke:#6d28d9,color:#2e1065
    class Y,G,CF ext
    class FQ pure
    class ST store
    class OP,SCH,R person
    style M stroke-dasharray:6 4,stroke-width:2px
```

- No secret is stored in the repo: `GEMINI_API_KEY` comes from the environment.
- Private inputs (`portfolio.txt`) and every store (`data/`, `output/`, `logs/`) are gitignored.
- The bulk CSV is built locally but never deployed. Yahoo's text is untrusted, so Jinja2 autoescape stays on.

[Back to contents](#contents)

## Data flow

**Data moves one way, from source to published page.** Each store is labelled with its format, grain and write
pattern.

```mermaid
flowchart TB
    Y(["Yahoo Finance"]) -->|"statements · quotes"| F["<code>fetch</code><br/>planner + download"]
    F -->|"upsert, atomic"| RAW[("<code>raw.parquet</code><br/>1 row / ticker · 482 cols")]
    RAW --> CORE["<code>metrics</code> → <code>filters</code> → <code>ranking</code><br/>pure functions"]
    RAW -->|"copy per run date"| RA[("<code>snapshots/raw/</code><br/>dated archive")]
    CORE -->|"overwrite"| RK[("<code>ranked_‹label›.parquet</code><br/>1 row / ticker · 863 cols")]
    CORE -->|"one per run date"| SN[("<code>snapshots/</code><br/>scores + method hash")]
    RK --> HL["<code>highlights</code><br/>the 14 lists"]
    HL --> GR["<code>grading</code>"]
    GR <-->|"only when due"| GM(["Gemini"])
    GR -->|"upsert"| GD[("<code>grades.parquet</code><br/>1 row / ticker")]
    GD -->|"orders the lists,<br/>never the scores"| HL
    HL -->|"frozen once"| FW["<code>forward</code>"]
    Y -->|"month-end prices"| FW
    FW -->|"append-only"| LG[("<code>forward/</code><br/>cohort + ledger")]
    HL --> SITE["<code>report</code><br/>Jinja2 + Plotly"]
    LG --> SITE
    SITE --> OUT[("<code>output/benchmark/</code><br/>static HTML")]
    SN --> EV["<code>evaluate</code>"]
    RA -->|"rescore with today's method"| EV

    classDef pure fill:#dbeafe,stroke:#1d4ed8,color:#0b1f4d
    classDef ext fill:#fef3c7,stroke:#b45309,color:#4a2a04
    classDef store fill:#e5e7eb,stroke:#374151,color:#111827
    class CORE,HL,EV pure
    class Y,GM,F,GR,FW ext
    class RAW,RA,RK,SN,GD,LG,OUT store
```

Blue boxes are pure functions with no I/O, amber boxes reach outside the machine, grey cylinders are stored
data, and the site builder (uncoloured) only writes local files. The metric core is a fixed sequence of steps, each writing named columns that later steps read
(`uv run python -m fq.columns` generates the column catalogue):

| Step | What it does |
|---|---|
| `cleaning` | blanks unit errors and unconfirmed TTM anomalies (audited annual figures are only noted, never blanked) |
| `capital` | leases, enterprise value, book value, the shareholders' base |
| `flows` | owner earnings (cash profit after the investment that keeps the business as it is), ROIC, acquisitions, dividends |
| `inflation` | drops unrestated years in high-inflation reporters |
| `normalise` | normal-year earnings power, growth profiles |
| `quality` | ROIC level, trend and stability; leverage; per-share growth; share reduction |
| `valuation` | the Value inputs (owner yield on a normal year) |
| `flags` | suspect-data and model-review flags that keep a company off every list |

Each company is scored as a percentile within its peer group (operating, asset-heavy, bank, insurer) across the
whole benchmark, so its score doesn't depend on which run it is in. Each criterion is a named entry in
`settings.py`, and its input is computed in the matching `metrics/` step.

[Back to contents](#contents)

## One run, step by step

**A run does only the work that is due, and every optional stage can fail without losing the site.** This is
`fq run -u benchmark`, as implemented in `pipeline.py`:

```mermaid
%%{init: {'sequence': {'actorMargin': 24, 'width': 110, 'mirrorActors': false, 'messageFontSize': 15, 'noteFontSize': 14, 'actorFontSize': 15}}}%%
sequenceDiagram
    autonumber
    actor Op as Operator
    participant P as pipeline
    participant Y as Yahoo
    participant D as stores
    participant C as core
    participant F as forward
    participant S as site
    participant G as grading
    participant Gm as Gemini
    Op->>P: fq run -u benchmark
    P->>D: load cache, check schema
    Note over P: refresh.plan: due companies<br/>and why, at most 60
    loop each due company, 3 threads
        P->>Y: full fetch
    end
    P->>Y: batch price refresh (> 7 days old)
    P->>D: raw.parquet (temp file + rename)
    P->>C: metrics → filters → ranking
    C-->>P: ranked table
    P->>D: ranked, snapshot, raw archive
    P->>F: month-end marks
    F->>Y: total-return prices
    Note over F: on error: logged,<br/>the run goes on
    P->>S: build site
    Note over S: unchanged pages<br/>skipped
    opt benchmark, online, API key set
        P->>G: grade due (at most 60)
        G->>Gm: 2 passes + synthesis
        G->>D: grades.parquet
        P->>S: rebuild changed pages
    end
    P->>D: published lists
```

**When a company is re-fetched** (`refresh.reason`; otherwise it only gets a price refresh):

| Reason, in priority order | Trigger |
|---|---|
| new | not in the cache (the run's own new tickers bypass the budget) |
| earnings passed | an earnings date passed since the last fetch, plus 2 days |
| quarter due | last quarter end + 3 months + 45 days |
| annual due | last fiscal year end + 1 year + 90 days |
| new fields | the fetched field set changed since (`FIELDS_VERSION`) |
| incomplete | TTM or a critical line missing, retried after 7 days |
| old | older than 90 days |

Download and ranking can also run apart: `fq fetch --max-fetch 0` as often as needed, then `fq run --offline`.
Every command and option: `uv run fq --help`.

[Back to contents](#contents)

## Data model

**Eight datasets. Two of them can't be rebuilt and are protected:** the grades, which cost money, and the forward
test, which can't be re-run once Yahoo forgets a delisted company.

```mermaid
erDiagram
    RAW ||--o| RANKED : "measured, filtered, ranked"
    RAW ||--o{ RAW_ARCHIVE : "copied per run date"
    RANKED ||--o{ SNAPSHOT : "scores per run date"
    RANKED ||--o| GRADE : "listed companies graded"
    GRADE ||--o| GRADE_GATE : "frozen with the cohort"
    COHORT }o--o{ LEDGER : "joined on Ticker"
    RAW {
        string Ticker PK
        datetime Fetched_At
        string Fields_Version
        datetime PeriodEnd_TTM
        string TTM_Source "provenance"
    }
    RANKED {
        string Ticker PK
        string Business_Model
        float Owner_Yield
        float Value_Score
        float Quality_Score
        int Rank
    }
    RAW_ARCHIVE {
        date Run_Date PK "file name"
        string Ticker PK
        bool In_Benchmark "point in time"
    }
    SNAPSHOT {
        date Run_Date PK "file name"
        string Ticker PK
        string Method "code fingerprint"
        float Value_Score
        float Quality_Score
    }
    GRADE {
        string Ticker PK
        float Moat
        float Management
        string Prompt_Hash
        float Cost_USD
        datetime Graded_At
    }
    GRADE_GATE {
        string Ticker PK
        bool Passes
    }
    COHORT {
        string List PK
        string Ticker PK
        int Position
        date Start
        string Method
    }
    LEDGER {
        date Date PK "month-end"
        string Ticker PK
        float Return "USD, with dividends"
        float Price_Return "USD, price only (since Nov 2026)"
        bool Stale "held at last price"
    }
```

| Dataset | Path | Rows today | Written | Kept |
|---|---|---|---|---|
| RAW | `data/raw.parquet` | 9,938 | upsert per fetch, atomic | rebuilt only when `SCHEMA_VERSION` changes |
| RANKED | `data/ranked_<label>.parquet` | 9,670 | overwritten every rank | derived, rebuildable |
| RAW_ARCHIVE | `data/snapshots/raw/<date>.parquet` | 12 days | one per run day | kept, so old dates can be rescored |
| SNAPSHOT | `data/snapshots/<label>_<date>.parquet` | 21 runs | one per day; a same-day run replaces it | kept |
| GRADE | `data/grades.parquet` | 391 | upsert when due, saved per company | kept: costly to rebuild |
| GRADE_GATE, COHORT | `data/forward/` | 387, 381 | frozen once | **never deleted** |
| LEDGER | `data/forward/ledger.parquet` | 2,118 | appended each month-end | **never deleted** |

Periods are relative (`TTM`, `Y0` = latest annual report, `Y1`, ...); real dates live in `PeriodEnd_<period>`. No
calendar year appears in code or column names.

[Back to contents](#contents)

## Layers

**The scoring core imports only the foundations, and a test enforces it.** An arrow means "imports".

```mermaid
flowchart TB
    CLI["<code>cli.py</code><br/>options and prompts"] --> PL["<code>pipeline.py</code><br/>the run as plain functions"]
    PL --> NET["<b>Downloads</b><br/><code>fetch</code> · <code>refresh</code> · <code>health</code>"]
    PL --> SITE["<b>Site</b><br/><code>report/</code>"]
    PL --> EVAL["<b>Evaluation</b><br/><code>evaluate</code> · <code>forward</code>"]
    PL --> LLM["<b>LLM</b> · <code>grading.py</code><br/>Gemini client imported<br/>only inside the call"]
    SITE --> EVAL
    SITE -->|"gate rules only"| LLM
    EVAL -->|"gate rules only"| LLM
    NET --> CORE
    SITE --> CORE
    EVAL --> CORE
    subgraph CORE["Scoring core"]
        direction LR
        ME["<code>metrics/</code>"] --> RA["<code>filters.py</code> · <code>ranking.py</code>"]
    end
    CORE -->|"its only imports"| FND["<b>Foundations</b><br/><code>settings</code> · <code>universe</code> · <code>fx</code> · <code>inflation</code>"]

    classDef pure fill:#dbeafe,stroke:#1d4ed8,color:#0b1f4d
    classDef ext fill:#fef3c7,stroke:#b45309,color:#4a2a04
    class ME,RA,FND pure
    class NET,LLM ext
    style CORE stroke:#1d4ed8,stroke-width:3px
```

`tests/test_layers.py` parses every import, lazy ones included, and fails if any of these rules breaks:

- The scoring core (`metrics/`, `ranking.py`, `filters.py`, `settings.py`) never imports the site, the downloads,
  the LLM, the pipeline or the evaluation.
- The site never imports the command line, the pipeline, the downloads or `evaluate.py`.
- Only the entry points (`cli.py`, `pipeline.py`) import the pipeline or the command line.
- `grading.py` reads nothing from the site except the slug function that names its files.

The site and the evaluation read the grade's gate rules (constants and pure functions) from `grading.py`. The
Gemini SDK is imported only inside the one function that calls it.

[Back to contents](#contents)

## The LLM boundary

**The model judges; it never scores.**

| Aspect | Design |
|---|---|
| Input | The company's own page as markdown, stripped of every conclusion the system draws (scores, ranks, badges), so nothing anchors the grade |
| Process | Two independent grading passes (`gemini-4-argon`) averaged, then a synthesis by a cheaper model (`gemini-3.8-flash`) |
| Output | Moat and Management, 1 to 5. A gate keeps a company at the top of its list or moves it to the bottom. Scores and ranks are untouched |
| Caching | One row per company, keyed on `Prompt_Hash` (prompt, models, passes, page format). Redone only when that hash, the latest annual report or the one-year age limit says so |
| Cost control | At most 60 grades per run, a `--dry-run` estimate before spending, tokens and `Cost_USD` stored per company; rewording the synthesis never re-runs the costly passes |


[Back to contents](#contents)

## Quality attributes

| Attribute | How it is achieved | Where |
|---|---|---|
| Reproducibility | Deterministic core; each snapshot carries a hash of the scoring code; dated raw archives let any past date be rescored with today's method | `evaluate.method_fingerprint`, `pipeline.rescore` |
| Reliability | Atomic parquet writes; schema check on load; the forward test and grading are isolated, so their failure never blocks the site | `fetch.save_raw`, `health.check_schema`, `pipeline.forward_step` |
| Data quality | Unit-error and anomaly checks, provenance per company, suspect companies kept off every list | `metrics/cleaning.py`, `ranking.trusted` |
| Cost | Incremental fetching; grades cached on a hash with a per-run cap | `refresh.select`, `grading.due` |
| Performance | Unchanged company pages skipped by fingerprint; capped render processes for an 8 GB machine; the full ranking's rows load lazily | `company_page.PAGE_INPUTS`, `FQ_RENDER_WORKERS` |
| Security and privacy | Autoescape on untrusted text, secrets from the environment, private stores gitignored, site behind access control | `report/html.py`, `.gitignore` |
| Maintainability | Import layers tested; one registry of lists; ticker lists and overrides as text; a generated column catalogue | `tests/test_layers.py`, `report/highlights.TABS`, `fq.columns` |

[Back to contents](#contents)

## Trade-offs

| Choice | Instead of | Why | Revisit when |
|---|---|---|---|
| Parquet files | A database | One writer, batch access, wide columnar tables, nothing to operate | Several writers or concurrent readers |
| Static site | A web app | No server to secure, free hosting, cacheable pages | Readers need their own queries |
| Batch runs | Streaming | Fundamentals change quarterly, and the scores look years ahead, so weekly prices are enough | Never, for this purpose |
| LLM at the edge | LLM inside the scores | Scores must be reproducible, cheap and explainable line by line | Never for scores |
| One machine | Cloud jobs | The stores total under 100 MB of parquet; the 8 GB memory limit is handled by capping render workers | The universe grows tenfold |
| Criteria chosen by reasoning | Weights fitted to past returns | Fitting to price history overfits; the forward test judges the result instead | — |

[Back to contents](#contents)

## Failure modes

| What fails | How it is detected | What happens |
|---|---|---|
| Yahoo renames or drops a row | Coverage warning in `health.py` | Aliases and composites rebuild it; a new alias fixes the rest |
| Rate limiting | Fetch errors | Threads take turns; the budget defers the rest to the next run |
| Cache from an older schema | `SCHEMA_VERSION` check | `fetch` discards it; `rank` refuses it |
| Run killed mid-write | — | Temp file plus rename: the previous file stays whole |
| Gemini error or no API key | Exception or missing key | Grading skipped and logged; the site builds with the grades it has |
| Forward test error | Exception | Logged; the report is still built |
| Month-end download gap | More than 20% of tickers unpriced | The month is not stored, and is retried next run |
| A company is delisted | No price after a mark | Held at its last value and marked `Stale` |

[Back to contents](#contents)

## Operations and testing

- **Environment:** `uv` with a locked `uv.lock` (yfinance pinned); lint and format with Ruff.
- **Tests:** 378 tests on synthetic data, with no network. They cover method rules, extraction edge cases taken
  from real failures, the layer rules, agreement between the front page and company pages, and forward-test
  accounting.
- **Schedule:** a monthly job refreshes prices and adds the month-end to the forward test. Deploying is always by
  hand.

[Back to contents](#contents)

## How the project is developed

**Built with an AI coding agent inside explicit guard-rails:**

- A short instructions file (`CLAUDE.md`, kept out of this repository) is the map the agent loads every
  session: layout, rules (for example "LLM only in `grading.py`", "never delete `data/forward/`") and recipes for
  adding a list, a criterion or a metric step.
- Private decision records give each choice its date and reason, so a later change can be judged against why
  the current one was made.
- The rules that matter most are tests, not conventions: import layers, the LLM boundary, front-page and
  company-page agreement.
- A backlog holds known, parked work, so scope stays deliberate.

[Back to contents](#contents)
