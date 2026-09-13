"""
Generate candidate payment plans for a request.

A "candidate" is a fully-specified proposal:
  - method: full_payment | partial_payment | installments | wait | not_recommended
  - plan: list of (date, amount) payments
  - spending_changes: list of change strings (stop/reduce_to)
  - plan_id: tie-breaker (payment_option_id or synthetic)

This module ONLY generates candidates. It does NOT validate or rank them.
Validation lives in validator.py; ranking lives in ranking.py.

Candidate types generated:
  1. full_payment       — pay everything on request_date
  2. partial_payment    — if allowed and safe (2 payments)
  3. installments       — one candidate per matching payment_option row
  4. wait               — pay full amount on earliest_date_for_full_payment
  5. not_recommended    — the fallback (no payments)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional

import pandas as pd

from data_cleaner import get as get_clean

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    """A single proposed plan, not yet validated."""
    method: str                                  # allowed payment_method value
    payments: list[tuple[date, Decimal]]         # chronological (date, amount)
    spending_changes: list[str] = field(default_factory=list)
    # For installments: the payment_option_id it derives from
    payment_option_id: Optional[str] = None

    def total_paid(self) -> Decimal:
        return sum((amt for _, amt in self.payments), Decimal("0"))

    def n_payments(self) -> int:
        return len(self.payments)

    def first_payment_date(self) -> Optional[date]:
        return self.payments[0][0] if self.payments else None

    def last_payment_date(self) -> Optional[date]:
        return self.payments[-1][0] if self.payments else None


# ---------------------------------------------------------------------------
# Full payment candidate
# ---------------------------------------------------------------------------
def candidate_full_payment(
    request_date: date,
    requested_amount: Decimal,
) -> Candidate:
    """Pay the full amount on request_date, no spending changes."""
    return Candidate(
        method="full_payment",
        payments=[(request_date, requested_amount)],
        spending_changes=[],
    )


# ---------------------------------------------------------------------------
# Partial payment candidate
# ---------------------------------------------------------------------------
def candidate_partial_payment(
    request_date: date,
    requested_amount: Decimal,
    amount_safe_to_pay: Decimal,
    earliest_full_payment_date: Optional[date],
    allows_partial_payment: bool,
) -> Optional[Candidate]:
    """
    Return a partial-payment candidate if it is structurally possible:
      - request allows partial payment
      - 0 < amount_safe_to_pay < requested_amount
      - earliest_full_payment_date is known

    The plan is always exactly 2 payments:
      (request_date, amount_safe_to_pay)
      (earliest_full_payment_date, requested_amount - amount_safe_to_pay)

    NOTE: the problem statement additionally requires that earliest_full_payment
    is <= desired_completion_date. That check happens in validator.py.
    """
    if not allows_partial_payment:
        return None
    if amount_safe_to_pay <= 0:
        return None
    if amount_safe_to_pay >= requested_amount:
        return None
    if earliest_full_payment_date is None:
        return None

    remainder = requested_amount - amount_safe_to_pay
    return Candidate(
        method="partial_payment",
        payments=[
            (request_date, amount_safe_to_pay),
            (earliest_full_payment_date, remainder),
        ],
        spending_changes=[],
    )


# ---------------------------------------------------------------------------
# Installment candidates
# ---------------------------------------------------------------------------
def _expand_installment_dates(
    first_date: date,
    n_payments: int,
    frequency_days: Optional[float],
) -> list[date]:
    """
    Return the list of payment dates for an installment plan.
      - n_payments == 1: just [first_date]
      - else: use frequency_days (fall back to 30 if missing)
    """
    if n_payments <= 1:
        return [first_date]
    freq = int(round(frequency_days)) if frequency_days and frequency_days > 0 else 30
    return [first_date + timedelta(days=freq * i) for i in range(n_payments)]


def candidate_installments(
    request_id: str,
    request_date: date,
    home_currency: str,
    max_installment_months: Optional[float],
) -> list[Candidate]:
    """
    Return one Candidate per payment_options row matching this request_id
    with payment_method == 'installments'.

    The plan must EXACTLY match the supplied option:
      - payment_amount per occurrence
      - number_of_payments
      - first_payment_date
      - payment_frequency_days
    """
    opts = get_clean("payment_options")
    rows = opts[(opts["request_id"] == request_id) &
                (opts["payment_method"] == "installments")]
    candidates: list[Candidate] = []

    for _, r in rows.iterrows():
        n_payments = int(r["number_of_payments"]) if pd.notna(r["number_of_payments"]) else 1
        freq_days = r["payment_frequency_days"]
        first_date = r["first_payment_date"].date()
        amount = Decimal(str(r["payment_amount"]))

        # Optional: skip options whose first payment is before request_date.
        # (Defensive; should not happen with clean data.)
        if first_date < request_date:
            # still valid — the plan may predate the request evaluation.
            # We keep it; validator will decide.
            pass

        # Optional max_installment_months filter:
        if max_installment_months is not None and n_payments > 1:
            # approximate: installment count must not exceed months * 30 / freq
            freq_actual = freq_days if freq_days and freq_days > 0 else 30
            max_payments = int((max_installment_months * 30) // freq_actual)
            if max_payments > 0 and n_payments > max_payments:
                continue

        dates = _expand_installment_dates(first_date, n_payments, freq_days)
        payments = [(d, amount) for d in dates]

        candidates.append(Candidate(
            method="installments",
            payments=payments,
            spending_changes=[],
            payment_option_id=str(r["payment_option_id"]),
        ))

    return candidates


# ---------------------------------------------------------------------------
# Wait candidate
# ---------------------------------------------------------------------------
def candidate_wait(
    earliest_full_payment_date: Optional[date],
    requested_amount: Decimal,
) -> Optional[Candidate]:
    """
    Wait candidate: pay the full amount on earliest_full_payment_date.
    Only valid if that date is known.
    """
    if earliest_full_payment_date is None:
        return None
    return Candidate(
        method="wait",
        payments=[(earliest_full_payment_date, requested_amount)],
        spending_changes=[],
    )


# ---------------------------------------------------------------------------
# Not-recommended candidate (fallback)
# ---------------------------------------------------------------------------
def candidate_not_recommended() -> Candidate:
    """The fallback: no payments, no spending changes."""
    return Candidate(
        method="not_recommended",
        payments=[],
        spending_changes=[],
    )


# ---------------------------------------------------------------------------
# Aggregate: all candidates for a request
# ---------------------------------------------------------------------------
def generate_candidates(
    request_row: pd.Series,
    amount_safe_to_pay: Decimal,
    earliest_full_payment_date: Optional[date],
    home_currency: str,
    max_installment_months: Optional[float],
) -> list[Candidate]:
    """
    Generate the full candidate set for a request.
    Order is deterministic; ranking handles final selection.
    """
    request_id = str(request_row["request_id"])
    request_date = request_row["request_date"].date()
    requested_amount = Decimal(str(request_row["requested_amount"]))
    allows_partial = bool(request_row["allows_partial_payment"])

    candidates: list[Candidate] = []

    # 1. Full payment
    candidates.append(candidate_full_payment(request_date, requested_amount))

    # 2. Partial payment (if structurally possible)
    pp = candidate_partial_payment(
        request_date, requested_amount, amount_safe_to_pay,
        earliest_full_payment_date, allows_partial,
    )
    if pp is not None:
        candidates.append(pp)

    # 3. Installments (one per eligible option)
    candidates.extend(
        candidate_installments(
            request_id, request_date, home_currency, max_installment_months
        )
    )

    # 4. Wait
    w = candidate_wait(earliest_full_payment_date, requested_amount)
    if w is not None:
        candidates.append(w)

    # 5. Not recommended (always present as fallback)
    candidates.append(candidate_not_recommended())

    return candidates


if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO)
    from datetime import date as _d
    from decimal import Decimal as D

    # Manual candidate generation for request_01
    cands = candidate_installments("request_01", _d(2024, 3, 3), "ZAR", None)
    print(f"request_01 installment candidates: {len(cands)}")
    for c in cands[:3]:
        print(f"  {c.payment_option_id} -> {len(c.payments)} payments, total {c.total_paid()}")