"""
Dated currency conversion using dataset/exchange_rates.csv only.
No live APIs. No hardcoded rates.

Strategy:
  - Build a lookup: (from_ccy, to_ccy, date) -> rate
  - If a direct pair on/before the target date exists, use it.
  - Otherwise triangulate via a hub currency (USD) if both legs exist.
  - Otherwise raise ConversionError — callers must decide how to handle it.
"""
from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal, getcontext
from functools import lru_cache

import pandas as pd

from config import DATASET_DIR
from data_cleaner import get as get_clean

getcontext().prec = 28  # high precision for money math

log = logging.getLogger(__name__)


class ConversionError(RuntimeError):
    """Raised when a required conversion cannot be performed from dataset rates."""


# ---------------------------------------------------------------------------
# Internal index
# ---------------------------------------------------------------------------
def _build_index(rates: pd.DataFrame) -> pd.DataFrame:
    """
    Return rates sorted by (from, to, date) for fast as-of lookup.
    """
    df = rates.copy()
    df = df.sort_values(["from_currency", "to_currency", "rate_date"])
    return df


@lru_cache(maxsize=1)
def _rates_df() -> pd.DataFrame:
    return _build_index(get_clean("rates"))


def _as_of_rate(from_ccy: str, to_ccy: str, on_date: date) -> Decimal | None:
    """
    Return the most recent rate for (from -> to) on or before on_date.
    None if no such rate exists.
    """
    df = _rates_df()
    mask = (
        (df["from_currency"] == from_ccy)
        & (df["to_currency"] == to_ccy)
        & (df["rate_date"] <= pd.Timestamp(on_date))
    )
    sub = df.loc[mask]
    if sub.empty:
        return None
    row = sub.iloc[-1]  # latest on-or-before
    return Decimal(str(row["rate"]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def convert(
    amount: float | int | Decimal,
    from_ccy: str,
    to_ccy: str,
    on_date: date,
) -> Decimal:
    """
    Convert `amount` from `from_ccy` to `to_ccy` using the most recent
    rate on or before `on_date`. Triangulates via USD when needed.

    Raises ConversionError if no path exists.
    """
    from_ccy = from_ccy.upper()
    to_ccy = to_ccy.upper()
    amt = Decimal(str(amount))

    if from_ccy == to_ccy:
        return amt

    # Direct
    r = _as_of_rate(from_ccy, to_ccy, on_date)
    if r is not None:
        return amt * r

    # Inverse
    r_inv = _as_of_rate(to_ccy, from_ccy, on_date)
    if r_inv is not None and r_inv != 0:
        return amt / r_inv

    # Triangulate via USD
    if from_ccy != "USD" and to_ccy != "USD":
        leg1 = _as_of_rate(from_ccy, "USD", on_date)
        leg2 = _as_of_rate("USD", to_ccy, on_date)
        if leg1 is not None and leg2 is not None:
            return amt * leg1 * leg2
        # Try inverse legs
        leg1_inv = _as_of_rate("USD", from_ccy, on_date)
        leg2_inv = _as_of_rate(to_ccy, "USD", on_date)
        if leg1_inv is not None and leg2_inv is not None and leg1_inv != 0 and leg2_inv != 0:
            return (amt / leg1_inv) / leg2_inv

    raise ConversionError(
        f"No conversion path for {from_ccy}->{to_ccy} on/before {on_date}"
    )


def to_home_currency(
    amount: float | int | Decimal,
    from_ccy: str,
    home_ccy: str,
    on_date: date,
) -> Decimal:
    """Convenience wrapper — always returns a Decimal in home currency."""
    return convert(amount, from_ccy, home_ccy, on_date)


if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO)
    from datetime import date as _d
    print("USD->IDR on 2025-08-05:", convert(100, "USD", "IDR", _d(2025, 8, 5)))
    print("IDR->USD on 2025-08-05:", convert(1_000_000, "IDR", "USD", _d(2025, 8, 5)))
    print("EUR->ZAR on 2024-03-03:", convert(100, "EUR", "ZAR", _d(2024, 3, 3)))