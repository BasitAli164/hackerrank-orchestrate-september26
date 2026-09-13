# Buy or Wait — AI Financial Affordability Agent

**HackerRank Orchestrate — September 2026 submission.**

A deterministic financial decision engine that answers:

> **"Can this user safely afford this requested expense?"**

The system reconstructs each user's financial situation from the supplied
datasets, projects a 90-day balance forecast, and recommends one of four
affordability outcomes with an eligible payment plan.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Models Used](#models-used)
4. [Setup](#setup)
5. [Running the Agent](#running-the-agent)
6. [Project Structure](#project-structure)
7. [How the Engine Works](#how-the-engine-works)
8. [Key Design Decisions](#key-design-decisions)
9. [Testing](#testing)
10. [Usage & Cost Report](#usage--cost-report)
11. [Submission Artifacts](#submission-artifacts)
12. [Limitations](#limitations)

---

## Overview

The task is a **financial affordability decision system**, not a chatbot.
For each request, the agent determines:

- `amount_safe_to_pay` — largest amount payable today that keeps the user
  above their minimum balance for the next 90 days
- `affordability_status` — one of `affordable_now`, `affordable_with_plan`,
  `affordable_later`, `not_affordable`
- `recommended_payment_method` — `full_payment`, `partial_payment`,
  `installments`, `wait`, or `not_recommended`
- `payment_plan` — dated list of recommended payments
- `earliest_date_for_full_payment` — first date the full amount is safe
- `spending_changes_needed` — flexible recurring expenses to stop or reduce
- `decision_explanation` — short, personalized rationale

Every financial value is produced by **deterministic Python**. The LLM
only assists with **interpretation** (of messages and images) and
**phrasing** (of the final explanation).

---

## Architecture

```
                 ┌──────────────────────┐
                 │      DATASETS        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  DATA LOADING +      │
                 │  CLEANING (pandas)   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  EVENT RESOLUTION    │
                 │  (linked_event_id)   │
                 └──────────┬───────────┘
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
      GPT-OSS 120B                  Qwen 3.6 27B
      TEXT / REASONING              VISION / OCR
             │                             │
             └──────────────┬──────────────┘
                            ▼
                 ┌──────────────────────┐
                 │  STRUCTURED FACTS    │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │  DETERMINISTIC       │
                 │  FINANCIAL ENGINE    │
                 │  - currency          │
                 │  - 90-day forecast   │
                 │  - affordability     │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │  PLAN GENERATION     │
                 │  VALIDATION + RANK   │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │  FINAL DECISION      │
                 └──────────┬───────────┘
                            ▼
                    GPT-OSS 120B
                    EXPLANATION
                            │
                            ▼
                      output.csv
```

**Core principle:** AI for interpretation, Python for financial truth.

---

## Models Used

Both models are accessed through the **Groq API**.

| Role | Model ID | Purpose |
|---|---|---|
| Text / reasoning | `openai/gpt-oss-120b` | Message fact extraction, final explanation |
| Vision / OCR | `qwen/qwen3.6-27b` | Extract amounts from event images |

Model IDs are configurable via environment variables
(`GROQ_LLM_MODEL`, `GROQ_VLM_MODEL`) — no IDs are hardcoded throughout
the codebase.

Every API call is tracked by `code/usage_tracker.py`, and a summary is
written to `code/evaluation/usage_report.md` at the end of every run.

### Why these models

- **GPT-OSS 120B** supports strict JSON Schema mode on Groq, which lets
  us validate message extractions deterministically. It also handles
  English + Indonesian messages cleanly.
- **Qwen 3.6 27B** is Groq's vision-capable model and is only invoked
  for the **16 events** whose `amount` is blank but has a linked image.
  No images are sent to the text model, and no text messages are sent
  to the vision model.

---

## Setup

### Prerequisites

- Python 3.10+
- A Groq API key (free tier works, though rate limits apply)

### Installation

From the **project root** (`hackerrank-orchestrate-september26/`):

```powershell
# Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate          # Git Bash / MINGW64
# or
.venv\Scripts\Activate.ps1             # PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file at the project root:

```powershell
cp .env.example .env
```

Then edit `.env` and set your key:

```text
GROQ_API_KEY=gsk_your_real_key_here
GROQ_LLM_MODEL=openai/gpt-oss-120b
GROQ_VLM_MODEL=qwen/qwen3.6-27b
GROQ_REASONING_EFFORT=medium
LOG_LEVEL=INFO
```

**Never commit `.env`.** Only `.env.example` belongs in version control.

---

## Running the Agent

All commands are run from the `code/` directory.

### Full run (recommended for submission)

Processes all 250 rows in `dataset/requests.csv`, uses LLM explanations:

```powershell
cd code
python main.py
```

Runtime on the free tier: **~8–12 minutes** (rate-limit bound).

### Fast run (deterministic explanations only)

Same 250 rows, no LLM explanation calls:

```powershell
cd code
python main.py --no-llm-exp
```

Runtime: **~30 seconds**.

### Sanity-check run on samples

Runs on `dataset/sample_requests.csv` (25 rows) instead of the full set:

```powershell
cd code
python main.py --samples
```

### Outputs

Every run writes:

| File | Description |
|---|---|
| `dataset/output.csv` | Final predictions for every `request_id` |
| `code/evaluation/usage_report.md` | Token & cost summary for the run |
| `logs/run.log` | Full run log (INFO + warnings) |

---

## Project Structure

```
hackerrank-orchestrate-september26/
├── code/
│   ├── main.py                 # entry point
│   ├── config.py               # paths, model IDs, pricing, constants
│   ├── data_loader.py          # CSV loading
│   ├── data_cleaner.py         # date / numeric / list-field parsing
│   ├── currency.py             # dated conversion + triangulation
│   ├── financial_state.py      # per-user ledger builder
│   ├── forecast.py             # 90-day balance forecast
│   ├── affordability.py        # safe amount + earliest full-payment date
│   ├── payment_planner.py      # candidate plans
│   ├── spending_optimizer.py   # stop / reduce_to candidate sets
│   ├── validator.py            # deterministic plan validation
│   ├── ranking.py              # 6-tier ranking
│   ├── message_analyzer.py     # GPT-OSS 120B fact extraction
│   ├── image_analyzer.py       # Qwen 3.6 27B image OCR
│   ├── explanation.py          # explanation generation + fallback
│   ├── pipeline.py             # per-request orchestration
│   ├── groq_client.py          # Groq wrapper (retries, tokens, caps)
│   ├── usage_tracker.py        # token & cost accumulator
│   └── evaluation/
│       ├── main.py             # eval context stub
│       └── usage_report.md     # generated at end of run
│
├── dataset/                    # provided data (read-only)
│   ├── requests.csv
│   ├── sample_requests.csv
│   ├── financial_profiles.csv
│   ├── financial_events.csv
│   ├── exchange_rates.csv
│   ├── request_payment_options.csv
│   ├── messages.csv
│   ├── images.csv
│   ├── output.csv              # final submission output
│   └── media/images/           # 16 PNGs
│
├── tests/
│   └── test_pipeline.py        # pytest smoke tests
│
├── logs/                       # run.log written at runtime
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## How the Engine Works

### 1. Data loading and cleaning

All CSVs are loaded with pandas. Dates are parsed to date-only
timestamps, numeric columns are coerced, and pipe-delimited preference
fields in `financial_profiles.csv` are split into Python lists.

### 2. Event resolution

`financial_events.csv` contains full lifecycle chains via
`linked_event_id`. Child events (amendments, cancellations, settlements)
supersede their parents. We drop:

- Events with `status` in `{cancelled, failed, unrealized}`
- Events of type `investment_valuation`
- Events with `direction == non_cash`

Settled events from the past are used as **evidence of recurrence** but
are not re-applied to the balance, because the profile's
`current_available_balance` already reflects them.

### 3. Currency normalisation

All amounts are converted to the user's `home_currency` using
`exchange_rates.csv` only. Direct pair → inverse pair → USD
triangulation, all dated. `Decimal` arithmetic throughout.

### 4. Recurring-series detection

Recurring series are grouped by `(direction, category, amount-bucket)`.
A series must have:

- **≥ 2** occurrences
- Amounts within **±2 %** of the median (guards against lumping
  one-off purchases into a subscription)
- A **dominant interval** shared by ≥ 60 % of gaps

Volatile categories such as groceries and dining are **not** treated as
recurring.

### 5. 90-day forecast

For each recurring series, we project future occurrences using the
**latest** amount and the dominant interval. One-off future events are
added as-is. Everything is applied day-by-day against the starting
balance.

A plan is safe only if the balance **never** drops below
`minimum_balance_to_keep` at any point in the window.

### 6. Affordability

- **`amount_safe_to_pay`**: binary search over `[0, requested_amount]`
  using the safety predicate above.
- **`earliest_date_for_full_payment`**: forward scan day-by-day until a
  full payment on that day keeps the forecast safe.

### 7. Candidate plans

For every request we generate:

1. `full_payment` — pay the full amount today
2. `partial_payment` — two payments: `amount_safe_to_pay` today and the
   remainder on `earliest_date_for_full_payment`
3. `installments` — one candidate per matching row in
   `request_payment_options.csv` (the plan must match exactly)
4. `wait` — pay the full amount on the earliest safe date
5. `not_recommended` — the fallback, no payments

### 8. Spending-change candidates

Flexible recurring expenses may be:

- `stop:<event_id>` — if flexibility ∈ {stoppable, reducible_or_stoppable}
- `reduce_to:<event_id>:<amount>` — if flexibility ∈ {reducible,
  reducible_or_stoppable} and `amount >= minimum_allowed_amount`

We generate up to 60 change-sets per request (bounded), sorted by
preference for fewer changes and larger savings.

### 9. Validation

Each `(plan, spending_changes)` combination is validated:

- Method must be allowed by `payment_methods_user_will_consider`
- Payments must be within the 90-day window and ≥ `request_date`
- Plan structure must be legal (e.g. installments must match a supplied
  `payment_option_id`, partial must be exactly 2 payments summing to the
  request amount)
- The forecast must never dip below minimum
- The last payment must be ≤ `desired_completion_date`

Invalid combinations are dropped.

### 10. Ranking

Valid candidates are ranked by the exact 6-tier rule:

1. Complete the full request by `desired_completion_date` (already
   enforced by validation)
2. Require no spending changes
3. Minimize total paid
4. Start earlier
5. Fewer payments
6. Lowest `payment_option_id`

`not_recommended` is **excluded** from ranking and used only when no
other candidate passes validation.

### 11. Explanation

The deterministic decision is passed to GPT-OSS 120B with an explicit
instruction to reuse status and method verbatim and to introduce no new
numbers. If the response is empty, too long, or introduces an
implausible figure, a deterministic template is used instead.

---

## Key Design Decisions

1. **Deterministic engine owns every financial value.** The LLM never
   computes amounts, dates, or plans. It only reads and phrases.
2. **Bounded candidate search.** At most 400 (plan × change-set)
   validations per request.
3. **`Decimal` for money.** No floats where currency is involved.
4. **Strict recurrence detection.** Ambiguous series are treated as
   one-off, which is conservative.
5. **Missing amounts are never zero.** Blank event amounts trigger an
   optional VLM call; if extraction fails, the event is skipped safely.
6. **`not_recommended` never competes on cost.** It's a fallback, not a
   plan.
7. **Explanation has a deterministic fallback.** The engine's decision
   is authoritative.
8. **All AI calls are cached.** Same message content → same extraction.
   Same image → same extraction. This bounds cost and avoids
   inconsistency across requests.
9. **Token caps per call site.** Message extraction, explanation, and
   VLM extraction each have their own `max_tokens` to stay under the
   free-tier per-minute limits.

---

## Testing

```powershell
cd code
pytest ../tests -v
```

The current suite includes smoke tests for state building, forecast
creation, and safe-amount bounding. Additional tests can be added under
`tests/` following the same import pattern.

---

## Usage & Cost Report

After every run, `code/evaluation/usage_report.md` is regenerated with:

- Per-model call counts
- Input / output / total tokens per model
- Estimated cost per model (configurable pricing in `code/config.py`)
- Per-request averages across the full dataset
- Overall totals

**Example for the final 250-request run:**

| Metric | Value |
|---|---|
| Model | `openai/gpt-oss-120b` |
| Calls | 250 |
| Input tokens | ~81,000 |
| Output tokens | ~17,000 |
| Total tokens | ~98,000 |
| Estimated cost | **~$0.022 USD** |

Pricing is intentionally kept in `MODEL_PRICING` inside `code/config.py`
so it can be updated without touching any call site.

---

## Submission Artifacts

| Artifact | Description |
|---|---|
| `dataset/output.csv` | Predictions for every row in `dataset/requests.csv` |
| `code.zip` | Full runnable source, prompts, README, evaluation folder |
| `chat_transcript` | Development conversation (provided separately) |

`code.zip` includes:

```
code/                (all source)
tests/               (pytest smoke tests)
evaluation/          (nested under code/)
README.md
requirements.txt
.env.example
.gitignore
```

`code.zip` excludes:

```
.venv/
.git/
.env
__pycache__/
.pytest_cache/
logs/run.log
dataset/media/
```

---

## Limitations

- **Recurring detection is heuristic.** Some variable-amount series are
  conservatively treated as one-off, which can under-estimate a user's
  future expenses.
- **VLM OCR occasionally fails** on dense receipts. When it does, the
  affected event is skipped — never treated as a zero-amount event.
- **Free-tier Groq rate limits** cause occasional retries and slow down
  the LLM-enabled run. The client backs off and retries up to 3 times.
- **Explanation LLM is optional.** `--no-llm-exp` produces a fully
  deterministic `output.csv` in ~30 seconds.
- **Investment requests** are treated purely as affordability questions.
  No market data or investment advice is produced.

---

## License

Submission for HackerRank Orchestrate — September 2026.