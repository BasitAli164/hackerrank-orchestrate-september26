"""
Central configuration: paths, model IDs, pricing, constants.
No secrets here. Secrets come from environment via .env.
"""
from pathlib import Path
import os

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
# This file lives in code/. Project root is one level up.
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MEDIA_DIR = DATASET_DIR / "media" / "images"
LOGS_DIR = PROJECT_ROOT / "logs"
EVAL_DIR = CODE_DIR / "evaluation"

# Input CSVs
REQUESTS_CSV = DATASET_DIR / "requests.csv"
SAMPLE_REQUESTS_CSV = DATASET_DIR / "sample_requests.csv"
FINANCIAL_PROFILES_CSV = DATASET_DIR / "financial_profiles.csv"
FINANCIAL_EVENTS_CSV = DATASET_DIR / "financial_events.csv"
EXCHANGE_RATES_CSV = DATASET_DIR / "exchange_rates.csv"
REQUEST_PAYMENT_OPTIONS_CSV = DATASET_DIR / "request_payment_options.csv"
MESSAGES_CSV = DATASET_DIR / "messages.csv"
IMAGES_CSV = DATASET_DIR / "images.csv"
OUTPUT_CSV = DATASET_DIR / "output.csv"

# Output artifacts
USAGE_REPORT_MD = EVAL_DIR / "usage_report.md"
RUN_LOG = LOGS_DIR / "run.log"

# ---------------------------------------------------------------------------
# MODELS (Groq)
# ---------------------------------------------------------------------------
LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
VLM_MODEL = os.getenv("GROQ_VLM_MODEL", "qwen/qwen3.6-27b")

# Reasoning effort for GPT-OSS 120B: low | medium | high
LLM_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "medium")

# ---------------------------------------------------------------------------
# PRICING (USD per 1M tokens) — configurable, keep in one place
# ---------------------------------------------------------------------------
MODEL_PRICING = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "qwen/qwen3.6-27b":    {"input": 0.60, "output": 3.00},
}

# ---------------------------------------------------------------------------
# FINANCIAL CONSTANTS
# ---------------------------------------------------------------------------
FORECAST_DAYS = 90            # safety horizon
MAX_SPENDING_CHANGES = 3      # per problem statement
MAX_VLM_IMAGE_BYTES = 15 * 1024 * 1024   # stay well under 20 MB Groq limit

# ---------------------------------------------------------------------------
# ENUM VALUES (from problem_statement.md — do not invent new ones)
# ---------------------------------------------------------------------------
AFFORDABILITY_STATUSES = {
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
}

PAYMENT_METHODS = {
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
}

# Event statuses we IGNORE for cash-flow forecasting
IGNORED_EVENT_STATUSES = {"cancelled", "failed", "unrealized"}

# Event types we IGNORE entirely for cash balance
IGNORED_EVENT_TYPES = {"investment_valuation"}

# Direction values
DIRECTION_DEBIT = "debit"
DIRECTION_CREDIT = "credit"
DIRECTION_NON_CASH = "non_cash"

# Flexibility values that permit STOP
STOPPABLE_FLEXIBILITIES = {"stoppable", "reducible_or_stoppable"}
# Flexibility values that permit REDUCE_TO
REDUCIBLE_FLEXIBILITIES = {"reducible", "reducible_or_stoppable"}

# ---------------------------------------------------------------------------
# OUTPUT COLUMNS (exact order — do not change)
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")