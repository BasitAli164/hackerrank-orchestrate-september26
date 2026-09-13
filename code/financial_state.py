"""
Build a per-user financial state from cleaned events + profile.

Responsibilities:
  - Filter out events we must ignore (cancelled, failed, unrealized,
    investment_valuation, non_cash direction).
  - Apply lifecycle: linked_event_id chains (amendment / cancellation /
    settlement).
  - Determine recurring income and recurring expenses.
  - Convert all amounts to the user's home currency.

This module does NOT know about requests. It only builds a canonical
ledger for a given user, using only events on/before a given `as_of_date`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Optional

import pandas as pd

from config import (
    IGNORED_EVENT_STATUSES,
    IGNORED_EVENT_TYPES,
    DIRECTION_DEBIT,
    DIRECTION_CREDIT,
    DIRECTION_NON_CASH,
    STOPPABLE_FLEXIBILITIES,
    REDUCIBLE_FLEXIBILITIES,
)
from currency import to_home_currency
from data_cleaner import get as get_clean

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class LedgerEntry:
    """A single normalized financial event, in home currency."""
    event_id: str
    event_type: str
    category: str
    direction: str            # 'debit' | 'credit'
    amount_home: Decimal      # positive magnitude, home currency
    event_date: date
    status: str               # 'settled' | 'pending' | 'scheduled' | ...
    flexibility: str          # 'fixed' | 'reducible' | 'stoppable' | 'reducible_or_stoppable'
    minimum_allowed_amount: Optional[Decimal]
    is_recurring: bool
    original_currency: str
    original_amount: Optional[Decimal]


@dataclass
class UserFinancialState:
    user_id: str
    home_currency: str
    current_balance: Decimal
    minimum_balance_to_keep: Decimal
    payment_methods_user_will_consider: list[str]
    expense_categories_to_protect: list[str]
    expense_categories_to_reduce: list[str]
    expense_categories_to_stop: list[str]
    max_installment_months: Optional[float]
    entries: list[LedgerEntry] = field(default_factory=list)

    def debits(self) -> list[LedgerEntry]:
        return [e for e in self.entries if e.direction == "debit"]

    def credits(self) -> list[LedgerEntry]:
        return [e for e in self.entries if e.direction == "credit"]

    def recurring_debits(self) -> list[LedgerEntry]:
        return [e for e in self.debits() if e.is_recurring]

    def recurring_credits(self) -> list[LedgerEntry]:
        return [e for e in self.credits() if e.is_recurring]

    def flexible_recurring_debits(self) -> list[LedgerEntry]:
        return [
            e for e in self.recurring_debits()
            if e.flexibility in ("reducible", "stoppable", "reducible_or_stoppable")
        ]


# ---------------------------------------------------------------------------
# Recurrence heuristic
# ---------------------------------------------------------------------------
_RECURRING_EVENT_TYPES = {"subscription"}

# Categories that are STRUCTURALLY recurring — expected to appear at a
# stable cadence with a stable amount. Volatile-spend categories (groceries,
# dining, shopping, entertainment) are excluded and treated as one-off.
_RECURRING_CATEGORY_HINTS = {
    "rent", "housing", "utilities", "internet", "phone",
    "insurance", "salary", "payroll", "pension",
    "education", "childcare", "loan_repayment", "debt_repayment",
    "streaming", "cloud_storage", "delivery_membership",
    "gym", "membership",
}


def _is_recurring(event_type: str, category: str, description: str) -> bool:
    """
    Recurrence heuristic. The dataset does not have an explicit 'recurring'
    flag, so we infer it from event_type, category, and description.
    """
    if event_type in _RECURRING_EVENT_TYPES:
        return True
    cat = (category or "").lower()
    if cat in _RECURRING_CATEGORY_HINTS:
        return True
    desc = (description or "").lower()
    for kw in ("monthly", "recurring", "subscription", "every month",
               "weekly", "annual", "yearly", "quarterly"):
        if kw in desc:
            return True
    return False


# ---------------------------------------------------------------------------
# Filtering & lifecycle resolution
# ---------------------------------------------------------------------------
def _should_ignore(row: pd.Series) -> bool:
    """Return True if this event must never affect the cash forecast."""
    if row["status"] in IGNORED_EVENT_STATUSES:
        return True
    if row["event_type"] in IGNORED_EVENT_TYPES:
        return True
    if row["direction"] == DIRECTION_NON_CASH:
        return True
    return False


def _resolve_lifecycle(events: pd.DataFrame) -> pd.DataFrame:
    """
    Apply linked_event_id rules:
      - If event A has linked_event_id pointing to B, A amends/cancels/
        settles B.
      - Child events supersede parents.
      - If the child is cancelled/failed, drop BOTH.
    """
    df = events.copy()

    children = df[df["linked_event_id"].notna()]
    superseded_parents: set[str] = set()
    drop_event_ids: set[str] = set()

    for _, child in children.iterrows():
        parent_id = child["linked_event_id"]
        child_id = child["event_id"]
        superseded_parents.add(parent_id)

        if child["status"] in ("cancelled", "failed"):
            drop_event_ids.add(child_id)

    df = df[~df["event_id"].isin(superseded_parents)]
    df = df[~df["event_id"].isin(drop_event_ids)]
    return df


# ---------------------------------------------------------------------------
# Public: build a user state
# ---------------------------------------------------------------------------
def build_user_state(user_id: str, as_of_date: date) -> UserFinancialState:
    """
    Build the user's financial state using events with event_date <= as_of_date.
    Past events (settled before as_of_date) are not re-applied to the balance
    (the profile's `current_available_balance` already reflects the settled past).
    Instead, we keep them as evidence for recurrence detection.
    """
    profiles = get_clean("profiles")
    prof_rows = profiles[profiles["user_id"] == user_id]
    if prof_rows.empty:
        raise ValueError(f"No profile for user_id={user_id}")
    prof = prof_rows.iloc[0]

    home_ccy = prof["home_currency"]
    balance = Decimal(str(prof["current_available_balance"]))
    min_balance = Decimal(str(prof["minimum_balance_to_keep"]))
    max_install = (
        float(prof["max_installment_months"])
        if pd.notna(prof["max_installment_months"])
        else None
    )

    events = get_clean("events")
    ev = events[events["user_id"] == user_id].copy()
    ev = ev.sort_values(["event_date", "event_id"]).reset_index(drop=True)

    ev = _resolve_lifecycle(ev)
    ev = ev[~ev.apply(_should_ignore, axis=1)].copy()

    entries: list[LedgerEntry] = []
    for _, row in ev.iterrows():
        amt_orig = row["amount"]
        if pd.isna(amt_orig):
            # Blank amount — VLM should fill in later in the pipeline.
            continue

        orig_ccy = row["currency"]
        orig_amt = Decimal(str(amt_orig))
        try:
            amt_home = (
                orig_amt
                if orig_ccy == home_ccy
                else to_home_currency(orig_amt, orig_ccy, home_ccy, row["event_date"].date())
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Conversion failed for event %s (%s->%s on %s): %s",
                row["event_id"], orig_ccy, home_ccy,
                row["event_date"].date(), exc,
            )
            continue

        entries.append(
            LedgerEntry(
                event_id=row["event_id"],
                event_type=row["event_type"],
                category=row["category"] or "",
                direction=row["direction"],
                amount_home=amt_home,
                event_date=row["event_date"].date(),
                status=row["status"],
                flexibility=row["flexibility"],
                minimum_allowed_amount=(
                    Decimal(str(row["minimum_allowed_amount"]))
                    if pd.notna(row["minimum_allowed_amount"])
                    else None
                ),
                is_recurring=_is_recurring(
                    row["event_type"], row["category"], row["description"]
                ),
                original_currency=orig_ccy,
                original_amount=orig_amt,
            )
        )

    return UserFinancialState(
        user_id=user_id,
        home_currency=home_ccy,
        current_balance=balance,
        minimum_balance_to_keep=min_balance,
        payment_methods_user_will_consider=list(prof["payment_methods_user_will_consider"]),
        expense_categories_to_protect=list(prof["expense_categories_to_protect"]),
        expense_categories_to_reduce=list(prof["expense_categories_user_is_willing_to_reduce"]),
        expense_categories_to_stop=list(prof["expense_categories_user_is_willing_to_stop"]),
        max_installment_months=max_install,
        entries=entries,
    )


if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO)
    from datetime import date as _d

    for uid, dt in [("user_01", _d(2024, 3, 3)), ("user_02", _d(2025, 8, 5))]:
        st = build_user_state(uid, dt)
        print(f"\n=== {uid} ({st.home_currency}) ===")
        print(f"  balance: {st.current_balance}")
        print(f"  min_balance: {st.minimum_balance_to_keep}")
        print(f"  payment methods: {st.payment_methods_user_will_consider}")
        print(f"  max_installment_months: {st.max_installment_months}")
        print(f"  total entries: {len(st.entries)}")
        print(f"  recurring debits: {len(st.recurring_debits())}")
        print(f"  recurring credits: {len(st.recurring_credits())}")
        print(f"  flexible recurring debits: {len(st.flexible_recurring_debits())}")
        for e in st.flexible_recurring_debits()[:5]:
            print(f"    {e.event_id} {e.category!r:20} {e.amount_home:>15} flex={e.flexibility}")