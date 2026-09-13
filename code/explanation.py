"""
Generate the final human-readable explanation.

Input: the DETERMINISTIC decision (already made by the engine).
Output: a short, personalized explanation string.

The LLM is NEVER authoritative. If the LLM output contradicts the
decision, or the call fails, we fall back to a deterministic template.
"""
from __future__ import annotations

import logging
import re
from decimal import Decimal
from typing import Optional

from groq_client import GroqClient, GroqError

log = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You write ONE short, personalized sentence explaining a
completed financial recommendation to a user.

You MUST use the given affordability_status EXACTLY as provided:
- If status is 'affordable_now', say the payment is affordable now.
- If status is 'affordable_with_plan', say a plan is recommended.
- If status is 'affordable_later', say to wait until the earliest date.
- If status is 'not_affordable', say it is not affordable.

You MUST use the given recommended_payment_method EXACTLY as provided.
NEVER change any numeric value, date, or method.
NEVER introduce new amounts.
NEVER expose reasoning.
Reply with only the explanation sentence — no quotes, no prefix."""


def _fmt_money(amount: Decimal, ccy: str) -> str:
    s = f"{amount:,.2f}".rstrip("0").rstrip(".")
    return f"{ccy} {s}"


def _fmt_money_short(amount: Decimal, ccy: str) -> str:
    """Round to whole units for readability in text."""
    try:
        whole = int(amount.quantize(Decimal("1")))
    except Exception:  # noqa: BLE001
        whole = int(amount)
    return f"{ccy} {whole:,}"


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------
def build_fallback_explanation(
    home_currency: str,
    status: str,
    method: str,
    requested_amount: Decimal,
    safe_amount: Decimal,
    plan: str,
    earliest_full: Optional[str],
    min_balance: Decimal,
) -> str:
    """A deterministic explanation matching the sample style."""
    ccy = home_currency
    req = _fmt_money_short(requested_amount, ccy)
    safe = _fmt_money_short(safe_amount, ccy)
    mn = _fmt_money_short(min_balance, ccy)

    if status == "affordable_now":
        return (
            f"Pay {req} today. This leaves at least {mn} available "
            f"over the next 90 days."
        )
    if status == "affordable_with_plan":
        if method == "full_payment":
            return (
                f"Pay {req} today. This is safe after the recommended "
                f"spending changes and keeps at least {mn} available."
            )
        if method == "partial_payment":
            return (
                f"Pay {safe} today and the remaining balance on the "
                f"earliest safe date. This keeps at least {mn} available."
            )
        if method == "installments":
            return (
                f"Use installments to spread {req}. This keeps at least "
                f"{mn} available over the next 90 days."
            )
        return (
            f"Complete {req} safely by the deadline while keeping at "
            f"least {mn} available."
        )
    if status == "affordable_later":
        when = earliest_full or "later"
        return (
            f"Pay {req} in full on {when}. Paying earlier would take "
            f"the balance below the {mn} minimum."
        )
    if status == "not_affordable":
        if safe_amount > 0:
            return (
                f"Do not proceed with this request. Although {safe} is "
                f"available today, the full {req} cannot be completed "
                f"safely within 90 days while keeping {mn}."
            )
        return (
            f"Do not proceed with this request. None of the available "
            f"options keeps the {mn} minimum protected within 90 days."
        )
    return f"Recommendation: {method} for {req}."


# ---------------------------------------------------------------------------
# Consistency check (loose)
# ---------------------------------------------------------------------------
def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d[\d,\u202f]*(?:\.\d+)?", text))


def _is_consistent(
    llm_text: str,
    requested_amount: Decimal,
    safe_amount: Decimal,
    min_balance: Decimal,
    ccy: str,
) -> bool:
    """
    Loose sanity check: reject if the LLM introduces a number far larger
    than the requested amount (potential hallucination).
    Accept formatting variations of amounts we supplied.
    """
    if not llm_text or len(llm_text) < 15:
        return False

    cap = max(abs(float(requested_amount)), abs(float(safe_amount))) * 3 + 1.0

    for raw in _extract_numbers(llm_text):
        s = raw.replace(",", "").replace("\u202f", "").replace(" ", "")
        try:
            val = float(s)
        except ValueError:
            continue
        if val > cap:
            return False
    return True


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def generate_explanation(
    client: GroqClient,
    request_id: str,
    home_currency: str,
    status: str,
    method: str,
    requested_amount: Decimal,
    safe_amount: Decimal,
    plan: str,
    earliest_full: Optional[str],
    spending_changes: str,
    min_balance: Decimal,
) -> str:
    """
    Try to get an LLM-generated explanation. Fall back to the deterministic
    template if the LLM fails or produces contradictory numbers.
    """
    fallback = build_fallback_explanation(
        home_currency, status, method,
        requested_amount, safe_amount,
        plan, earliest_full, min_balance,
    )

    facts = (
        f"home_currency={home_currency}\n"
        f"affordability_status={status}\n"
        f"recommended_payment_method={method}\n"
        f"requested_amount={requested_amount}\n"
        f"amount_safe_to_pay={safe_amount}\n"
        f"payment_plan={plan}\n"
        f"earliest_date_for_full_payment={earliest_full}\n"
        f"spending_changes_needed={spending_changes}\n"
        f"minimum_balance_to_keep={min_balance}\n"
    )

    try:
        text = client.llm_text(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=(
                "Write ONE short explanation for this decision. "
                "Use affordability_status and recommended_payment_method "
                "verbatim from the facts below. Do not introduce new numbers.\n\n"
                + facts
            ),
            purpose="explanation",
            request_id=request_id,
        )
    except GroqError as e:
        log.warning("Explanation LLM failed for %s: %s", request_id, e)
        return fallback

    if not _is_consistent(text, requested_amount, safe_amount, min_balance, home_currency):
        log.warning(
            "Explanation for %s rejected (numeric inconsistency): %r",
            request_id, text[:120],
        )
        return fallback

    text = text.strip().strip('"').strip("'")
    if len(text) > 400:
        text = text[:400].rsplit(".", 1)[0] + "."
    return text