"""
Loads all dataset CSVs into pandas DataFrames with correct dtypes.
Does NOT modify data — cleaning happens in data_cleaner.py.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd

from config import (
    REQUESTS_CSV,
    SAMPLE_REQUESTS_CSV,
    FINANCIAL_PROFILES_CSV,
    FINANCIAL_EVENTS_CSV,
    EXCHANGE_RATES_CSV,
    REQUEST_PAYMENT_OPTIONS_CSV,
    MESSAGES_CSV,
    IMAGES_CSV,
    OUTPUT_CSV,
)

log = logging.getLogger(__name__)


def _read(path, **kwargs) -> pd.DataFrame:
    log.info("Loading %s", path.name)
    df = pd.read_csv(path, **kwargs)
    log.info("  -> %d rows, %d cols", len(df), len(df.columns))
    return df


@lru_cache(maxsize=1)
def load_requests() -> pd.DataFrame:
    return _read(REQUESTS_CSV)


@lru_cache(maxsize=1)
def load_sample_requests() -> pd.DataFrame:
    return _read(SAMPLE_REQUESTS_CSV)


@lru_cache(maxsize=1)
def load_financial_profiles() -> pd.DataFrame:
    return _read(FINANCIAL_PROFILES_CSV)


@lru_cache(maxsize=1)
def load_financial_events() -> pd.DataFrame:
    return _read(FINANCIAL_EVENTS_CSV)


@lru_cache(maxsize=1)
def load_exchange_rates() -> pd.DataFrame:
    return _read(EXCHANGE_RATES_CSV)


@lru_cache(maxsize=1)
def load_request_payment_options() -> pd.DataFrame:
    return _read(REQUEST_PAYMENT_OPTIONS_CSV)


@lru_cache(maxsize=1)
def load_messages() -> pd.DataFrame:
    return _read(MESSAGES_CSV)


@lru_cache(maxsize=1)
def load_images() -> pd.DataFrame:
    return _read(IMAGES_CSV)


@lru_cache(maxsize=1)
def load_output_template() -> pd.DataFrame:
    return _read(OUTPUT_CSV)


def load_all() -> dict[str, pd.DataFrame]:
    """Convenience loader for the whole dataset."""
    return {
        "requests": load_requests(),
        "sample_requests": load_sample_requests(),
        "profiles": load_financial_profiles(),
        "events": load_financial_events(),
        "rates": load_exchange_rates(),
        "payment_options": load_request_payment_options(),
        "messages": load_messages(),
        "images": load_images(),
    }


def inspect() -> None:
    """Print a quick summary of every dataset — used for debugging."""
    for name, df in load_all().items():
        print(f"\n=== {name} ===")
        print(f"shape: {df.shape}")
        print(f"columns: {list(df.columns)}")
        print(f"nulls per col:\n{df.isna().sum().to_string()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    inspect()