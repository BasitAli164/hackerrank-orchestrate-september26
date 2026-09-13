"""Minimal smoke tests for the Buy-or-Wait pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from decimal import Decimal
from datetime import date

from affordability import compute_amount_safe_to_pay
from financial_state import build_user_state
from forecast import build_forecast


def test_build_state_for_known_user():
    state = build_user_state("user_01", date(2024, 3, 3))
    assert state.home_currency == "ZAR"
    assert state.minimum_balance_to_keep == Decimal("18000")
    assert len(state.entries) > 0


def test_forecast_returns_safe_or_unsafe():
    state = build_user_state("user_01", date(2024, 3, 3))
    fc = build_forecast(state, date(2024, 3, 3))
    assert isinstance(fc.is_safe(), bool)
    assert fc.min_balance_in_window() is not None


def test_safe_amount_bounded():
    state = build_user_state("user_01", date(2024, 3, 3))
    safe = compute_amount_safe_to_pay(state, date(2024, 3, 3), Decimal("25256"))
    assert Decimal("0") <= safe <= Decimal("25256")
