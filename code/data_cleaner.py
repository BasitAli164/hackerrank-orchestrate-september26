"""
Cleans loaded DataFrames:
  - parse date columns to pandas Timestamp (date only, no tz)
  - parse numeric columns (coerce invalid to NaN)
  - normalize string columns (strip whitespace)
  - split pipe-delimited preference fields in profiles
"""
from __future__ import annotations

import logging

import pandas as pd

from data_loader import (
    load_financial_events,
    load_financial_profiles,
    load_requests,
    load_messages,
    load_images,
    load_request_payment_options,
    load_exchange_rates,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_date(s: pd.Series) -> pd.Series:
    """Parse YYYY-MM-DD string to date-only Timestamp (tz-naive)."""
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _split_pipe(value) -> list[str]:
    """'a|b|c' -> ['a','b','c']; NaN -> []"""
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]


# ---------------------------------------------------------------------------
# Public cleaners
# ---------------------------------------------------------------------------
def clean_requests() -> pd.DataFrame:
    df = load_requests().copy()
    df["request_date"] = _to_date(df["request_date"])
    df["desired_completion_date"] = _to_date(df["desired_completion_date"])
    df["requested_amount"] = _to_num(df["requested_amount"])
    df["allows_partial_payment"] = (
        df["allows_partial_payment"].astype(str).str.lower() == "true"
    )
    df["request_type"] = df["request_type"].astype(str).str.strip()
    df["request_text"] = df["request_text"].fillna("").astype(str)
    return df


def clean_profiles() -> pd.DataFrame:
    df = load_financial_profiles().copy()
    df["current_available_balance"] = _to_num(df["current_available_balance"])
    df["minimum_balance_to_keep"] = _to_num(df["minimum_balance_to_keep"])
    df["max_installment_months"] = _to_num(df["max_installment_months"])

    # Split pipe-delimited preference lists into Python lists
    for col in [
        "financial_priorities",
        "expense_categories_to_protect",
        "expense_categories_user_is_willing_to_reduce",
        "expense_categories_user_is_willing_to_stop",
        "payment_methods_user_will_consider",
    ]:
        df[col] = df[col].apply(_split_pipe)
    return df


def clean_events() -> pd.DataFrame:
    df = load_financial_events().copy()
    df["event_date"] = _to_date(df["event_date"])
    df["settlement_date"] = _to_date(df["settlement_date"])
    df["amount"] = _to_num(df["amount"])
    df["minimum_allowed_amount"] = _to_num(df["minimum_allowed_amount"])
    df["event_type"] = df["event_type"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip().replace("nan", "")
    df["direction"] = df["direction"].astype(str).str.strip()
    df["status"] = df["status"].astype(str).str.strip()
    df["flexibility"] = df["flexibility"].astype(str).str.strip()
    df["currency"] = df["currency"].astype(str).str.strip()
    return df


def clean_messages() -> pd.DataFrame:
    df = load_messages().copy()
    # sent_at is ISO 8601 with 'Z' — parse to UTC then drop tz
    df["sent_at"] = (
        pd.to_datetime(df["sent_at"], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )
    df["message_text"] = df["message_text"].fillna("").astype(str)
    df["source_type"] = df["source_type"].astype(str).str.strip()
    return df


def clean_images() -> pd.DataFrame:
    df = load_images().copy()
    for col in ["image_id", "user_id", "request_id", "related_event_id"]:
        df[col] = df[col].astype("object")
    return df


def clean_payment_options() -> pd.DataFrame:
    df = load_request_payment_options().copy()
    df["first_payment_date"] = _to_date(df["first_payment_date"])
    df["payment_frequency_days"] = _to_num(df["payment_frequency_days"])
    df["number_of_payments"] = _to_num(df["number_of_payments"]).astype("Int64")
    for col in ["payment_amount", "financing_fee", "total_payable_amount"]:
        df[col] = _to_num(df[col])
    df["payment_method"] = df["payment_method"].astype(str).str.strip()
    return df


def clean_exchange_rates() -> pd.DataFrame:
    df = load_exchange_rates().copy()
    df["rate_date"] = _to_date(df["rate_date"])
    df["rate"] = _to_num(df["rate"])
    df["from_currency"] = df["from_currency"].astype(str).str.strip().str.upper()
    df["to_currency"] = df["to_currency"].astype(str).str.strip().str.upper()
    return df


# ---------------------------------------------------------------------------
# Cached public accessors
# ---------------------------------------------------------------------------
_CACHE: dict[str, pd.DataFrame] = {}


def get(name: str) -> pd.DataFrame:
    """
    Cached cleaned DataFrame by name.
    Valid names: requests, profiles, events, messages, images,
                 payment_options, rates
    """
    if name in _CACHE:
        return _CACHE[name]
    fn = {
        "requests": clean_requests,
        "profiles": clean_profiles,
        "events": clean_events,
        "messages": clean_messages,
        "images": clean_images,
        "payment_options": clean_payment_options,
        "rates": clean_exchange_rates,
    }[name]
    df = fn()
    _CACHE[name] = df
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for name in ["requests", "profiles", "events", "messages", "images",
                 "payment_options", "rates"]:
        df = get(name)
        print(f"\n=== {name} ===")
        print(df.dtypes.to_string())
        print(df.head(2).to_string())