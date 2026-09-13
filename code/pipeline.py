"""
Orchestrate the full per-request pipeline:

  1. Load request row + user profile
  2. Build user financial state (deterministic)
  3. Compute amount_safe_to_pay and earliest_date_for_full_payment
  4. Generate candidate plans
  5. Generate candidate spending-change sets (bounded)
  6. Validate all (candidate, change-set) combinations
  7. Rank valid combinations (not_recommended excluded from ranking)
  8. Pick the best, or fall back to not_recommended
  9. Determine final affordability_status
 10. Build the output row
 11. Optionally ask the LLM for a nicer explanation

All financial decisions happen in deterministic Python.
AI only assists with interpretation and phrasing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Optional

import pandas as pd

from config import FORECAST_DAYS, MAX_SPENDING_CHANGES
from data_cleaner import get as get_clean
from financial_state import build_user_state, UserFinancialState
from forecast import build_forecast
from affordability import (
    compute_amount_safe_to_pay,
    compute_earliest_full_payment_date,
)
from payment_planner import Candidate, generate_candidates
from spending_optimizer import (
    generate_spending_change_candidates,
    format_changes,
    SpendingChange,
)
from validator import validate_candidate, format_plan
from ranking import rank_candidates
from explanation import generate_explanation, build_fallback_explanation
from groq_client import GroqClient

log = logging.getLogger(__name__)

# Bounded candidate search to prevent combinatorial explosion
_MAX_CHANGE_SETS_PER_REQUEST = 60
_MAX_TOTAL_CANDIDATES = 400


@dataclass
class Decision:
    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: Optional[date]
    spending_changes_needed: str
    decision_explanation: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _score_change_set(changes: list[SpendingChange]) -> tuple:
    """Rank change-sets: prefer fewer changes and higher savings."""
    total_savings = sum(float(c.savings) for c in changes)
    return (len(changes), -total_savings)


def _bounded_change_sets(state: UserFinancialState) -> list[list[SpendingChange]]:
    combos = generate_spending_change_candidates(state)
    combos.sort(key=_score_change_set)
    return combos[:_MAX_CHANGE_SETS_PER_REQUEST]


def _status_from_decision(
    method: str,
    amount_safe_to_pay: Decimal,
    requested_amount: Decimal,
    earliest_full: Optional[date],
    request_date: date,
    best_candidate: Optional[Candidate],
) -> str:
    """
    Map the deterministic outcome to one of the four allowed statuses.

    Rules:
      - not_affordable: no valid plan OR method == not_recommended
      - affordable_now: full amount safe today AND user accepts full_payment
      - affordable_with_plan: valid partial/installments/full-with-changes
      - affordable_later: wait is the recommendation
    """
    if best_candidate is None or method == "not_recommended":
        return "not_affordable"

    if method == "wait":
        return "affordable_later"

    if method == "full_payment":
        # Full payment today only classifies as affordable_now if the safe
        # amount covers the full request AND no spending changes are needed.
        if (
            amount_safe_to_pay >= requested_amount
            and earliest_full is not None
            and earliest_full == request_date
            and not best_candidate.spending_changes
        ):
            return "affordable_now"
        # full_payment with spending changes → plan
        return "affordable_with_plan"

    if method in ("partial_payment", "installments"):
        return "affordable_with_plan"

    return "not_affordable"


# ---------------------------------------------------------------------------
# Main per-request entry point
# ---------------------------------------------------------------------------
def process_request(
    request_row: pd.Series,
    llm_client: Optional[GroqClient] = None,
    use_llm_explanation: bool = True,
) -> Decision:
    request_id = str(request_row["request_id"])
    user_id = str(request_row["user_id"])
    request_date = request_row["request_date"].date()
    requested_amount = Decimal(str(request_row["requested_amount"]))
    desired_completion_date = request_row["desired_completion_date"].date()

    # ---- 1. Profile + state ------------------------------------------------
    profiles = get_clean("profiles")
    prof_rows = profiles[profiles["user_id"] == user_id]
    if prof_rows.empty:
        raise ValueError(f"No profile for {user_id}")
    prof = prof_rows.iloc[0]
    home_ccy = str(prof["home_currency"])
    min_balance = Decimal(str(prof["minimum_balance_to_keep"]))
    max_install = (
        float(prof["max_installment_months"])
        if pd.notna(prof["max_installment_months"]) else None
    )

    state = build_user_state(user_id, request_date)

    # ---- 2. Core metrics ---------------------------------------------------
    safe_amount = compute_amount_safe_to_pay(state, request_date, requested_amount)
    earliest_full = compute_earliest_full_payment_date(
        state, request_date, requested_amount
    )

    # ---- 3. Candidate plans ------------------------------------------------
    plan_candidates = generate_candidates(
        request_row, safe_amount, earliest_full, home_ccy, max_install
    )

    # ---- 4. Candidate spending change-sets ---------------------------------
    change_sets = _bounded_change_sets(state)

    # ---- 5. Validate all combos -------------------------------------------
    valid: list[tuple[Candidate, list[SpendingChange]]] = []
    total_tested = 0
    for plan in plan_candidates:
        if plan.method in ("wait", "not_recommended"):
            ok, reason = validate_candidate(plan, request_row, state, earliest_full)
            total_tested += 1
            if ok:
                valid.append((plan, []))
            continue

        for cs in change_sets:
            if total_tested >= _MAX_TOTAL_CANDIDATES:
                break
            plan_with_changes = Candidate(
                method=plan.method,
                payments=plan.payments,
                spending_changes=cs,
                payment_option_id=plan.payment_option_id,
            )
            ok, reason = validate_candidate(
                plan_with_changes, request_row, state, earliest_full
            )
            total_tested += 1
            if ok:
                valid.append((plan_with_changes, cs))
        if total_tested >= _MAX_TOTAL_CANDIDATES:
            break

    # ---- 6. Rank ----------------------------------------------------------
    # Exclude not_recommended from ranking: it must never compete with real
    # plans on total_paid (which is 0). It's only used as final fallback.
    real_candidates = [c for c, _ in valid if c.method != "not_recommended"]

    best: Optional[Candidate] = None
    if real_candidates:
        ranked = rank_candidates(real_candidates)
        best = ranked[0]

    # ---- 7. Determine status, plan, changes -------------------------------
    if best is None:
        status = "not_affordable"
        method = "not_recommended"
        plan_str = "none"
        changes_str = "none"
    else:
        method = best.method
        plan_str = format_plan(best.payments)
        changes_str = format_changes(best.spending_changes)
        status = _status_from_decision(
            method, safe_amount, requested_amount,
            earliest_full, request_date, best,
        )
        if method == "not_recommended":
            status = "not_affordable"

    earliest_str = earliest_full.isoformat() if earliest_full else None

    # ---- 8. Explanation ---------------------------------------------------
    if llm_client is not None and use_llm_explanation:
        explanation = generate_explanation(
            client=llm_client,
            request_id=request_id,
            home_currency=home_ccy,
            status=status,
            method=method,
            requested_amount=requested_amount,
            safe_amount=safe_amount,
            plan=plan_str,
            earliest_full=earliest_str,
            spending_changes=changes_str,
            min_balance=min_balance,
        )
    else:
        explanation = build_fallback_explanation(
            home_ccy, status, method,
            requested_amount, safe_amount, plan_str,
            earliest_str, min_balance,
        )

    return Decision(
        request_id=request_id,
        amount_safe_to_pay=safe_amount,
        affordability_status=status,
        recommended_payment_method=method,
        payment_plan=plan_str,
        earliest_date_for_full_payment=earliest_full,
        spending_changes_needed=changes_str,
        decision_explanation=explanation,
    )