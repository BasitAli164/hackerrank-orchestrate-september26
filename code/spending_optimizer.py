"""
Propose candidate spending changes (stop / reduce_to) that could make
a full payment safe when it otherwise wouldn't be.

Only flexible recurring expenses may be changed:
  - stop:<event_id>           — allowed if flexibility in {stoppable, reducible_or_stoppable}
  - reduce_to:<event_id>:<amt> — allowed if flexibility in {reducible, reducible_or_stoppable}
                                  and new_amount >= minimum_allowed_amount

Rules:
  - Max 3 changes total
  - stop and reduce_to on the SAME event are mutually exclusive
  - Categories the user has marked as willing to stop/reduce are preferred
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from config import (
    MAX_SPENDING_CHANGES,
    STOPPABLE_FLEXIBILITIES,
    REDUCIBLE_FLEXIBILITIES,
)
from financial_state import LedgerEntry, UserFinancialState

log = logging.getLogger(__name__)


@dataclass
class SpendingChange:
    event_id: str
    change_type: str                    # 'stop' | 'reduce_to'
    new_amount: Decimal | None          # only for reduce_to
    category: str
    savings: Decimal                    # amount saved per occurrence

    def format(self) -> str:
        if self.change_type == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{_fmt_amount(self.new_amount)}"


def _fmt_amount(x: Decimal | None) -> str:
    if x is None:
        return "0"
    # Drop trailing zeros while preserving precision
    s = format(x, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


# ---------------------------------------------------------------------------
# Generate candidate spending changes
# ---------------------------------------------------------------------------
def _eligible_for_stop(e: LedgerEntry, state: UserFinancialState) -> bool:
    if e.flexibility not in STOPPABLE_FLEXIBILITIES:
        return False
    # Category must be in the user's "willing to stop" list if that list exists
    if state.expense_categories_to_stop:
        if e.category.lower() not in [c.lower() for c in state.expense_categories_to_stop]:
            return False
    # Protected categories can never be touched
    if e.category.lower() in [c.lower() for c in state.expense_categories_to_protect]:
        return False
    return True


def _eligible_for_reduce(e: LedgerEntry, state: UserFinancialState) -> bool:
    if e.flexibility not in REDUCIBLE_FLEXIBILITIES:
        return False
    if e.minimum_allowed_amount is None:
        return False
    if e.minimum_allowed_amount >= e.amount_home:
        return False  # nothing to reduce
    if state.expense_categories_to_reduce:
        if e.category.lower() not in [c.lower() for c in state.expense_categories_to_reduce]:
            return False
    if e.category.lower() in [c.lower() for c in state.expense_categories_to_protect]:
        return False
    return True


def generate_spending_change_candidates(
    state: UserFinancialState,
    max_changes: int = MAX_SPENDING_CHANGES,
) -> list[list[SpendingChange]]:
    """
    Return a list of candidate change-sets. Each set is a list of
    SpendingChange objects (length 0..max_changes).

    We generate:
      - [] (no changes)
      - [each single stop]
      - [each single reduce_to]
      - greedy combos of size 2..max_changes (stops first, then reduces)

    The returned combinations are ordered deterministically.
    """
    flexible = state.flexible_recurring_debits()

    stops: list[SpendingChange] = []
    reduces: list[SpendingChange] = []

    for e in flexible:
        if _eligible_for_stop(e, state):
            stops.append(SpendingChange(
                event_id=e.event_id,
                change_type="stop",
                new_amount=None,
                category=e.category,
                savings=e.amount_home,
            ))
        if _eligible_for_reduce(e, state):
            # Reduce down to minimum_allowed_amount
            new_amt = e.minimum_allowed_amount
            reduces.append(SpendingChange(
                event_id=e.event_id,
                change_type="reduce_to",
                new_amount=new_amt,
                category=e.category,
                savings=e.amount_home - new_amt,
            ))

    # Deterministic ordering by savings descending, then event_id
    stops.sort(key=lambda c: (-float(c.savings), c.event_id))
    reduces.sort(key=lambda c: (-float(c.savings), c.event_id))

    candidates: list[list[SpendingChange]] = [[]]

    # Singles
    for c in stops + reduces:
        candidates.append([c])

    # Greedy combos
    def _extend(base: list[SpendingChange], pool: list[SpendingChange],
                depth: int) -> None:
        if depth <= 0:
            return
        used_ids = {c.event_id for c in base}
        for c in pool:
            if c.event_id in used_ids:
                continue
            new_base = base + [c]
            candidates.append(new_base)
            _extend(new_base, pool, depth - 1)

    # Build combos up to max_changes
    for c in stops + reduces:
        if max_changes < 2:
            break
        _extend([c], stops + reduces, max_changes - 1)

    # Deduplicate (order-independent) while preserving first occurrence
    seen: set[frozenset[str]] = set()
    deduped: list[list[SpendingChange]] = []
    for combo in candidates:
        key = frozenset(c.format() for c in combo)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(combo)

    return deduped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_changes(changes: list[SpendingChange]) -> str:
    """Serialize a change list to the required output format."""
    if not changes:
        return "none"
    return "|".join(c.format() for c in changes)


if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO)
    from datetime import date as _d
    from financial_state import build_user_state

    for uid in ["user_01", "user_02", "user_21"]:
        st = build_user_state(uid, _d(2024, 3, 3))
        combos = generate_spending_change_candidates(st)
        print(f"\n=== {uid} : {len(combos)} change-sets ===")
        for combo in combos[:5]:
            print(f"  {format_changes(combo)}")