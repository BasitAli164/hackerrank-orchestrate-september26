"""
Entry point for the Buy-or-Wait agent.

Usage:
  python main.py                  # full run on dataset/requests.csv
  python main.py --samples        # run on dataset/sample_requests.csv
  python main.py --no-llm-exp     # skip LLM explanation (faster, cheaper)
"""
from __future__ import annotations

import argparse
import logging
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

from config import (
    REQUESTS_CSV, SAMPLE_REQUESTS_CSV, OUTPUT_CSV,
    EVAL_DIR, LOGS_DIR, USAGE_REPORT_MD, OUTPUT_COLUMNS,
)
from data_cleaner import get as get_clean
from pipeline import process_request
from usage_tracker import get_tracker
from groq_client import GroqClient


def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "run.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Silence noisy HTTP logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("groq._base_client").setLevel(logging.WARNING)


def _fmt_amount(d: Decimal) -> str:
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def run(
    use_samples: bool,
    use_llm_explanation: bool,
) -> None:
    setup_logging()
    log = logging.getLogger("main")

    log.info("=== Buy-or-Wait starting ===")
    log.info("Sample mode: %s | LLM explanation: %s",
             use_samples, use_llm_explanation)

    # Use sample_requests for sanity check, else requests
    if use_samples:
        req_df = pd.read_csv(SAMPLE_REQUESTS_CSV)
        # Only keep the input columns
        input_cols = [
            "request_id", "user_id", "request_date", "request_type",
            "requested_amount", "desired_completion_date",
            "allows_partial_payment", "request_text",
        ]
        req_df = req_df[input_cols].copy()
    else:
        req_df = get_clean("requests").copy()

    # Normalize date / bool columns as in data_cleaner
    req_df["request_date"] = pd.to_datetime(req_df["request_date"], errors="coerce").dt.normalize()
    req_df["desired_completion_date"] = pd.to_datetime(
        req_df["desired_completion_date"], errors="coerce"
    ).dt.normalize()
    req_df["requested_amount"] = pd.to_numeric(req_df["requested_amount"], errors="coerce")
    req_df["allows_partial_payment"] = (
        req_df["allows_partial_payment"].astype(str).str.lower() == "true"
    )
    log.info("Loaded %d requests", len(req_df))

    # Set up LLM client (single instance, reused)
    try:
        client = GroqClient() if use_llm_explanation else None
    except Exception as e:  # noqa: BLE001
        log.warning("Could not initialize Groq client: %s", e)
        client = None

    # Process
    rows = []
    failures = 0
    for i, (_, r) in enumerate(req_df.iterrows(), start=1):
        rid = str(r["request_id"])
        try:
            decision = process_request(
                r,
                llm_client=client,
                use_llm_explanation=(use_llm_explanation and client is not None),
            )
            rows.append({
                "request_id": decision.request_id,
                "amount_safe_to_pay": _fmt_amount(decision.amount_safe_to_pay),
                "affordability_status": decision.affordability_status,
                "recommended_payment_method": decision.recommended_payment_method,
                "payment_plan": decision.payment_plan,
                "earliest_date_for_full_payment": (
                    decision.earliest_date_for_full_payment.isoformat()
                    if decision.earliest_date_for_full_payment else ""
                ),
                "spending_changes_needed": decision.spending_changes_needed,
                "decision_explanation": decision.decision_explanation,
            })
        except Exception as e:  # noqa: BLE001
            failures += 1
            log.exception("Request %s failed: %s", rid, e)
            rows.append({
                "request_id": rid,
                "amount_safe_to_pay": "0",
                "affordability_status": "not_affordable",
                "recommended_payment_method": "not_recommended",
                "payment_plan": "none",
                "earliest_date_for_full_payment": "",
                "spending_changes_needed": "none",
                "decision_explanation": "Unable to complete safety check.",
            })
        if i % 25 == 0 or i == len(req_df):
            log.info("Processed %d/%d (failures=%d)", i, len(req_df), failures)

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(OUTPUT_CSV, index=False)
    log.info("Wrote %s (%d rows)", OUTPUT_CSV, len(out_df))

    # Usage report
    tracker = get_tracker()
    tracker.write_report(total_requests=len(out_df), path=USAGE_REPORT_MD)
    log.info("Wrote usage report: %s", USAGE_REPORT_MD)

    log.info("=== Done. Failures: %d ===", failures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", action="store_true",
                        help="Run on sample_requests.csv instead of requests.csv")
    parser.add_argument("--no-llm-exp", action="store_true",
                        help="Skip LLM explanation (use deterministic templates)")
    args = parser.parse_args()

    run(
        use_samples=args.samples,
        use_llm_explanation=(not args.no_llm_exp),
    )


if __name__ == "__main__":
    main()