"""
Computes the two key scalar metrics for a request:

  1. amount_safe_to_pay: the largest amount the user can pay ON request_date
     without the 90-day forecast ever dipping below minimum_balance_to_keep.
     Bounded by [0, requested_amount].

  2. earliest_date_for_full_payment: the first date D >= request_date on which
     paying the FULL requested_amount as a single payment is safe.

Both computations are PURE deterministic Python — no LLM involvement.

Approach:
  - A "hypothetical payment" is added to the forecast as a negative FutureEvent
    on the target date.
  - The forecast is re-evaluated. If min balance stays >= minimum, it's safe.

This is O(points) per test, and binary search gives ~log2(requested) calls
to it. Fine for 250 requests.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from config import FORECAST_DAYS
from financial_state import UserFinancialState
from forecast import Forecast, FutureEvent, build_forecast

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core primitive: simulate adding one hypothetical payment
# ---------------------------------------------------------------------------
def _simulate_with_payment(
    state: UserFinancialState,
    request_date: date,
    extra_payments: list[tuple[date, Decimal]],
    horizon_days: int = FORECAST_DAYS,
) -> Forecast:
    """
    Build a 90-day forecast and inject `extra_payments` (list of (date, amount)
    where amount is positive magnitude; internally treated as a debit).
    Returns the resulting Forecast.
    """
    fc = build_forecast(state, request_date, horizon_days=horizon_days)

    # Build the payment FutureEvents
    pay_events = [
        FutureEvent(
            date=d,
            amount_home=-abs(amt),
            event_id=f"__hypothetical_payment__{i}",
            category="__payment__",
            description="hypothetical request payment",
            is_recurring=False,
        )
        for i, (d, amt) in enumerate(extra_payments)
    ]

    # Rebuild points with payments merged
    by_date: dict[date, list[FutureEvent]] = {}
    for p in fc.points:
        by_date.setdefault(p.date, []).extend(p.events)
    for e in pay_events:
        if request_date <= e.date <= fc.end_date:
            by_date.setdefault(e.date, []).append(e)

    points = []
    bal = state.current_balance
    for d in sorted(by_date.keys()):
        for e in by_date[d]:
            bal = bal + e.amount_home
        points.append(type(fc.points[0])(date=d, balance_after_day=bal, events=by_date[d])) if fc.points else None

    # If fc.points was empty, still produce a valid Forecast
    from forecast import BalancePoint
    points = []
    bal = state.current_balance
    for d in sorted(by_date.keys()):
        for e in by_date[d]:
            bal = bal + e.amount_home
        points.append(BalancePoint(date=d, balance_after_day=bal, events=by_date[d]))

    return Forecast(
        start_date=fc.start_date,
        end_date=fc.end_date,
        starting_balance=fc.starting_balance,
        minimum_balance=fc.minimum_balance,
        points=points,
    )


def is_safe_with_payments(
    state: UserFinancialState,
    request_date: date,
    extra_payments: list[tuple[date, Decimal]],
) -> bool:
    fc = _simulate_with_payment(state, request_date, extra_payments)
    return fc.is_safe()


# ---------------------------------------------------------------------------
# amount_safe_to_pay (binary search)
# ---------------------------------------------------------------------------
def compute_amount_safe_to_pay(
    state: UserFinancialState,
    request_date: date,
    requested_amount: Decimal,
) -> Decimal:
    """
    Largest single payment on request_date that keeps the 90-day forecast safe.
    Returns Decimal in [0, requested_amount].

    Binary search over a Decimal range, stopping at cent precision.
    """
    if requested_amount <= 0:
        return Decimal("0")

    # Fast path: full amount is safe?
    if is_safe_with_payments(
        state, request_date, [(request_date, requested_amount)]
    ):
        return requested_amount

    # Fast path: even 0 extra payment is unsafe? (i.e. the base forecast itself
    # dips below min — meaning the user cannot pay ANYTHING on request_date
    # without some other intervention. In that case safe amount is 0.)
    if not is_safe_with_payments(state, request_date, []):
        # Base forecast is unsafe; no positive payment is safe.
        return Decimal("0")

    lo = Decimal("0")
    hi = requested_amount
    cent = Decimal("0.01")

    # ~40 iterations gives well below cent precision for realistic magnitudes
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid - lo < cent:
            break
        if is_safe_with_payments(state, request_date, [(request_date, mid)]):
            lo = mid
        else:
            hi = mid

    # Floor to cents (never round UP — would risk unsafe)
    return (lo).quantize(Decimal("0.01"))


# ---------------------------------------------------------------------------
# earliest_date_for_full_payment
# ---------------------------------------------------------------------------
def compute_earliest_full_payment_date(
    state: UserFinancialState,
    request_date: date,
    requested_amount: Decimal,
    max_search_days: int = FORECAST_DAYS,
) -> date | None:
    """
    Earliest date D in [request_date, request_date + max_search_days] such
    that paying requested_amount in full on D is safe.

    Returns None if no such date is found within the horizon.
    """
    if requested_amount <= 0:
        return request_date

    for offset in range(0, max_search_days + 1):
        d = request_date + timedelta(days=offset)
        # When checking a future date D, the forecast should still start at
        # request_date (its base window is defined by the request).
        if is_safe_with_payments(
            state, request_date, [(d, requested_amount)]
        ):
            return d
    return None


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO)
    from datetime import date as _d
    from financial_state import build_user_state

    cases = [
        ("user_01", _d(2024, 3, 3), Decimal("25256")),
        ("user_02", _d(2025, 8, 5), Decimal("46018000")),
        ("user_03", _d(2019, 9, 3), Decimal("5491000")),
    ]
    for uid, req_dt, amt in cases:
        st = build_user_state(uid, req_dt)
        safe = compute_amount_safe_to_pay(st, req_dt, amt)
        earliest = compute_earliest_full_payment_date(st, req_dt, amt)
        print(f"\n=== {uid} request on {req_dt} for {amt} ===")
        print(f"  amount_safe_to_pay: {safe}")
        print(f"  earliest_full_date: {earliest}")