"""
Rank valid candidates by the exact 6-tier rule from the problem statement:

  1. Complete the full request by desired_completion_date  (validated upstream)
  2. Require no spending changes
  3. Minimize the total amount paid
  4. Start payment earlier
  5. Use fewer payments
  6. Use the lowest payment_option_id (final tie-breaker)

Deterministic Python only — never delegate this to an LLM.
"""
from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import Optional

from payment_planner import Candidate

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ranking key
# ---------------------------------------------------------------------------
def _ranking_key(candidate: Candidate) -> tuple:
    """
    Return a tuple that sorts ascending in the desired order.
    """
    no_spending_changes = 0 if not candidate.spending_changes else 1
    total_paid = candidate.total_paid()
    first_payment = candidate.first_payment_date() or date.max
    n_payments = candidate.n_payments()

    # For tie-break #6: payment_option_id — put option-bearing methods
    # before synthetic ones. Synthetic methods get a large sentinel.
    option_id = candidate.payment_option_id or "zzzzz"

    return (
        no_spending_changes,
        float(total_paid),
        first_payment,
        n_payments,
        option_id,
    )


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """
    Return candidates sorted best-first.
    Caller should pre-filter to only VALID candidates.
    """
    return sorted(candidates, key=_ranking_key)


def pick_best(candidates: list[Candidate]) -> Optional[Candidate]:
    ranked = rank_candidates(candidates)
    return ranked[0] if ranked else None


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------
def describe(c: Candidate) -> str:
    plan = "|".join(
        f"{d.isoformat()}:{format(a, 'f').rstrip('0').rstrip('.') if '.' in format(a,'f') else format(a,'f')}"
        for d, a in c.payments
    ) or "none"
    changes = "|".join(ch.format() for ch in c.spending_changes) or "none"
    return f"[{c.method}] plan={plan} changes={changes} total={c.total_paid()}"


if __name__ == "__main__":
    # Smoke test with synthetic candidates
    from datetime import date as _d
    from decimal import Decimal as D
    from payment_planner import Candidate

    c1 = Candidate(method="full_payment",
                   payments=[(_d(2024,3,3), D("100"))])
    c2 = Candidate(method="installments",
                   payments=[(_d(2024,3,10), D("50")), (_d(2024,4,10), D("50"))],
                   payment_option_id="payment_option_03")
    c3 = Candidate(method="wait",
                   payments=[(_d(2024,4,15), D("100"))])
    for c in rank_candidates([c2, c3, c1]):
        print(describe(c))