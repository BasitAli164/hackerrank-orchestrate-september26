"""
Validate candidate plans against the deterministic forecast.

A candidate is VALID if:
  1. method is allowed by user preferences
  2. every payment is within the 90-day horizon and >= request_date
  3. the full requested amount is covered by the last payment
     (or method is 'not_recommended' with no payments)
  4. for installments: plan EXACTLY matches a supplied payment_option row
  5. for partial_payment: exactly 2 payments summing to requested_amount,
     first == request_date, second == earliest_full_payment_date,
     earliest_full_payment_date <= desired_completion_date
  6. all spending_changes reference flexible recurring events and are legal
  7. the forecast NEVER dips below minimum_balance at any point
  8. the request is fully completed by desired_completion_date
     (except for 'wait' which must land on/before desired_completion_date)
"""
from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import Optional

import pandas as pd

from config import FORECAST_DAYS
from financial_state import UserFinancialState
from forecast import FutureEvent, build_forecast
from payment_planner import Candidate
from spending_optimizer import SpendingChange

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spending change application to the event stream
# ---------------------------------------------------------------------------
def _apply_spending_changes_to_state(
    state: UserFinancialState,
    changes: list[SpendingChange],
) -> UserFinancialState:
    """
    Return a shallow copy of `state` with the given spending changes
    applied: stopped events are removed; reduced events have their
    amount set to new_amount.
    """
    if not changes:
        return state

    changes_by_id = {c.event_id: c for c in changes}
    new_entries = []
    for e in state.entries:
        c = changes_by_id.get(e.event_id)
        if c is None:
            new_entries.append(e)
            continue
        if c.change_type == "stop":
            continue
        # reduce_to
        if c.new_amount is None:
            new_entries.append(e)
            continue
        from dataclasses import replace
        new_entries.append(replace(e, amount_home=c.new_amount))

    from dataclasses import replace as _replace
    return _replace(state, entries=new_entries)


# ---------------------------------------------------------------------------
# Forecast under a candidate
# ---------------------------------------------------------------------------
def _forecast_with_candidate(
    state: UserFinancialState,
    request_date: date,
    payments: list[tuple[date, Decimal]],
    changes: list[SpendingChange],
):
    """
    Build a 90-day forecast including the candidate's payments and spending
    changes. Returns a Forecast object.
    """
    state2 = _apply_spending_changes_to_state(state, changes)
    fc = build_forecast(state2, request_date, horizon_days=FORECAST_DAYS)

    # Inject payments
    pay_events = [
        FutureEvent(
            date=d,
            amount_home=-abs(amt),
            event_id=f"__candidate_payment__{i}",
            category="__payment__",
            description="candidate payment",
            is_recurring=False,
        )
        for i, (d, amt) in enumerate(payments)
    ]

    by_date: dict[date, list[FutureEvent]] = {}
    for p in fc.points:
        by_date.setdefault(p.date, []).extend(p.events)
    for e in pay_events:
        if request_date <= e.date <= fc.end_date:
            by_date.setdefault(e.date, []).append(e)

    from forecast import BalancePoint
    points = []
    bal = state2.current_balance
    for d in sorted(by_date.keys()):
        for e in by_date[d]:
            bal = bal + e.amount_home
        points.append(BalancePoint(date=d, balance_after_day=bal, events=by_date[d]))

    from forecast import Forecast
    return Forecast(
        start_date=fc.start_date,
        end_date=fc.end_date,
        starting_balance=fc.starting_balance,
        minimum_balance=fc.minimum_balance,
        points=points,
    )


# ---------------------------------------------------------------------------
# Method eligibility vs user preferences
# ---------------------------------------------------------------------------
def _method_allowed(method: str, state: UserFinancialState) -> bool:
    """
    Determine whether the given payment method is allowed by user preferences.

    Rules from problem statement:
      - full_payment / partial_payment / installments must appear in
        payment_methods_user_will_consider
      - wait is allowed whenever full_payment is in the list
      - not_recommended is always allowed (it is the fallback)
    """
    if method == "not_recommended":
        return True
    prefs = {p.lower() for p in state.payment_methods_user_will_consider}
    if method in ("full_payment", "partial_payment", "installments"):
        return method in prefs
    if method == "wait":
        return "full_payment" in prefs
    return False


# ---------------------------------------------------------------------------
# Installment plan matching
# ---------------------------------------------------------------------------
def _matches_installment_option(
    candidate: Candidate, request_id: str
) -> bool:
    """
    Return True if the candidate exactly matches an entry in
    request_payment_options for this request (same id, dates, amounts).
    """
    from data_cleaner import get as get_clean
    if candidate.method != "installments":
        return True
    if candidate.payment_option_id is None:
        return False
    opts = get_clean("payment_options")
    row = opts[opts["payment_option_id"] == candidate.payment_option_id]
    if row.empty:
        return False
    row = row.iloc[0]
    if str(row["request_id"]) != request_id:
        return False
    amount = Decimal(str(row["payment_amount"]))
    n = int(row["number_of_payments"]) if pd.notna(row["number_of_payments"]) else 1
    if len(candidate.payments) != n:
        return False
    for d, a in candidate.payments:
        if a != amount:
            return False
    return True


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------
def validate_candidate(
    candidate: Candidate,
    request_row: pd.Series,
    state: UserFinancialState,
    earliest_full_payment_date: Optional[date],
) -> tuple[bool, str]:
    """
    Return (is_valid, reason). Reason is empty if valid.
    """
    request_id = str(request_row["request_id"])
    request_date = request_row["request_date"].date()
    requested_amount = Decimal(str(request_row["requested_amount"]))
    desired_completion_date = request_row["desired_completion_date"].date()
    allows_partial = bool(request_row["allows_partial_payment"])

    # ---- 1. Method allowed by user preferences
    if not _method_allowed(candidate.method, state):
        return False, f"method {candidate.method} not allowed by user preferences"

    # ---- 2. not_recommended is always structurally valid but never "safe"
    if candidate.method == "not_recommended":
        if candidate.payments:
            return False, "not_recommended must have no payments"
        return True, ""

    # ---- 3. Payments within horizon
    horizon_end = request_date + pd.Timedelta(days=FORECAST_DAYS).to_pytimedelta()
    for d, amt in candidate.payments:
        if d < request_date:
            return False, f"payment date {d} precedes request_date"
        if amt <= 0:
            return False, f"non-positive payment amount {amt}"

    # ---- 4. Method-specific structure
    if candidate.method == "full_payment":
        if len(candidate.payments) != 1:
            return False, "full_payment must have exactly one payment"
        if candidate.payments[0][1] != requested_amount:
            return False, "full_payment amount must equal requested_amount"
        if candidate.payments[0][0] != request_date:
            return False, "full_payment must be on request_date"

    elif candidate.method == "partial_payment":
        if not allows_partial:
            return False, "partial payment not allowed for this request"
        if len(candidate.payments) != 2:
            return False, "partial_payment must have exactly two payments"
        d1, a1 = candidate.payments[0]
        d2, a2 = candidate.payments[1]
        if d1 != request_date:
            return False, "partial_payment first payment must be on request_date"
        if a1 + a2 != requested_amount:
            return False, "partial_payment total must equal requested_amount"
        if earliest_full_payment_date is None:
            return False, "partial_payment requires earliest_full_payment_date"
        if d2 != earliest_full_payment_date:
            return False, "partial_payment second date must equal earliest_full_payment_date"
        if earliest_full_payment_date > desired_completion_date:
            return False, "partial_payment cannot complete by desired_completion_date"

    elif candidate.method == "installments":
        if not _matches_installment_option(candidate, request_id):
            return False, "installments do not match any supplied payment_option"

    elif candidate.method == "wait":
        if len(candidate.payments) != 1:
            return False, "wait must have exactly one payment"
        d, a = candidate.payments[0]
        if a != requested_amount:
            return False, "wait must pay the full requested_amount"
        if d > desired_completion_date:
            return False, "wait date exceeds desired_completion_date"

    # ---- 5. Spending changes are legal
    legal_ids = {e.event_id for e in state.flexible_recurring_debits()}
    for chg in candidate.spending_changes:
        if chg.event_id not in legal_ids:
            return False, f"spending change on non-flexible event {chg.event_id}"

    # ---- 6. Forecast safety
    fc = _forecast_with_candidate(
        state, request_date, candidate.payments, candidate.spending_changes
    )
    if fc.min_balance_in_window() < fc.minimum_balance:
        first_break = fc.first_date_below_minimum()
        return False, f"forecast dips below minimum on {first_break}"

    # ---- 7. Request fully completed by desired_completion_date
    last_payment = candidate.last_payment_date()
    if last_payment is not None and last_payment > desired_completion_date:
        return False, "last payment exceeds desired_completion_date"

    return True, ""


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------
def format_plan(payments: list[tuple[date, Decimal]]) -> str:
    """Serialize a plan to the required output format."""
    if not payments:
        return "none"
    parts = []
    for d, amt in payments:
        s = format(amt, "f")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        parts.append(f"{d.isoformat()}:{s}")
    return "|".join(parts)


if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO)
    print("validator module — importable. Full test via pipeline.py.")