"""
90-day balance forecast.

Key insight: the events dataset contains MONTHLY HISTORIES of recurring
events (e.g. 5 rows for the same monthly subscription). We must NOT sum
all rows into the future. Instead:

  1. Group recurring events into "recurring series" by
     (direction, category, amount-bucket).
  2. Validate each series:
       - >=2 occurrences
       - amounts must not vary wildly (allow a raise/drop at most
         _AMOUNT_MAX_DEVIATION from the LATEST amount)
       - the DOMINANT interval (rounded to nearest 5 days) must be shared
         by a majority of gaps
  3. Project future occurrences using the LATEST amount and the dominant
     interval, starting AFTER as_of_date.
  4. Add one-off future events (settled/scheduled/pending with future dates).
  5. Apply everything day-by-day against the starting balance.

Volatile-spend categories (groceries, dining) are intentionally NOT
treated as recurring because their amounts vary wildly. They are one-off.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from statistics import median

from config import FORECAST_DAYS
from financial_state import LedgerEntry, UserFinancialState

log = logging.getLogger(__name__)

# Tolerance for grouping recurring amounts (relative bucket width)
_AMOUNT_BUCKET_REL = 0.05

# Minimum history occurrences to infer recurrence
_MIN_RECURRENCE_OCCURRENCES = 2

# Interval bounds
_MIN_INTERVAL_DAYS = 5
_MAX_INTERVAL_DAYS = 45

# Rounding granularity for the dominant interval (days)
_INTERVAL_ROUNDING = 5

# Required fraction of gaps that must fall into the dominant interval bucket
_DOMINANT_INTERVAL_MIN_FRACTION = 0.6

# Sanity: latest amount must be within this factor of the historical median
# (guards against one-off huge credits being lumped into a recurring series)
_AMOUNT_MAX_RATIO_TO_MEDIAN = 5.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FutureEvent:
    """A concrete event scheduled to occur on a given date in the forecast."""
    date: date
    amount_home: Decimal       # signed: negative for debit, positive for credit
    event_id: str              # source event_id (or synthetic)
    category: str
    description: str           # human-readable label
    is_recurring: bool


@dataclass
class BalancePoint:
    date: date
    balance_after_day: Decimal
    events: list[FutureEvent]


@dataclass
class Forecast:
    start_date: date
    end_date: date
    starting_balance: Decimal
    minimum_balance: Decimal
    points: list[BalancePoint] = field(default_factory=list)

    def min_balance_in_window(self) -> Decimal:
        if not self.points:
            return self.starting_balance
        return min(p.balance_after_day for p in self.points)

    def first_date_below_minimum(self) -> date | None:
        for p in self.points:
            if p.balance_after_day < self.minimum_balance:
                return p.date
        return None

    def balance_on(self, d: date) -> Decimal:
        """Balance at end of day `d`."""
        bal = self.starting_balance
        for p in self.points:
            if p.date <= d:
                bal = p.balance_after_day
            else:
                break
        return bal

    def is_safe(self) -> bool:
        return self.min_balance_in_window() >= self.minimum_balance


# ---------------------------------------------------------------------------
# Recurrence interval inference
# ---------------------------------------------------------------------------
def _infer_interval_days(sorted_dates: list[date]) -> int | None:
    """
    Given >=2 historical dates, return the DOMINANT interval in days,
    rounded to the nearest _INTERVAL_ROUNDING days.

    The dominant interval must account for at least
    _DOMINANT_INTERVAL_MIN_FRACTION of all gaps. A stray bonus or
    irregular payment will not kill the series.
    """
    if len(sorted_dates) < _MIN_RECURRENCE_OCCURRENCES:
        return None
    gaps = [
        (sorted_dates[i + 1] - sorted_dates[i]).days
        for i in range(len(sorted_dates) - 1)
    ]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return None

    # Bucket gaps by rounding to nearest _INTERVAL_ROUNDING
    buckets = [int(round(g / _INTERVAL_ROUNDING)) * _INTERVAL_ROUNDING for g in gaps]
    counter = Counter(buckets)
    dominant, count = counter.most_common(1)[0]

    if dominant < _MIN_INTERVAL_DAYS or dominant > _MAX_INTERVAL_DAYS:
        return None
    if count / len(gaps) < _DOMINANT_INTERVAL_MIN_FRACTION:
        return None
    return dominant


# ---------------------------------------------------------------------------
# Grouping recurring series
# ---------------------------------------------------------------------------
def _series_key(e: LedgerEntry) -> tuple:
    """
    Group key for recurring events. Two events belong to the same series
    only if they share direction, category, AND have amounts within
    roughly +-2 *_AMOUNT_BUCKET_REL of each other.
    """
    cat = (e.category or e.event_type).lower()
    bucket_size = max(1.0, float(e.amount_home) * _AMOUNT_BUCKET_REL * 2)
    bucket = int(round(float(e.amount_home) / bucket_size))
    return (e.direction, cat, bucket)


def _build_series(entries: list[LedgerEntry]) -> dict[tuple, list[LedgerEntry]]:
    """
    Build candidate series by bucket key, then VALIDATE each series:
      - >=_MIN_RECURRENCE_OCCURRENCES occurrences
      - latest amount within _AMOUNT_MAX_RATIO_TO_MEDIAN of the median
        (prevents merging a huge one-off credit into a small series)
    Interval consistency is checked later in _infer_interval_days.
    """
    candidates: dict[tuple, list[LedgerEntry]] = {}
    for e in entries:
        if not e.is_recurring:
            continue
        candidates.setdefault(_series_key(e), []).append(e)

    series: dict[tuple, list[LedgerEntry]] = {}
    for key, members in candidates.items():
        members.sort(key=lambda x: x.event_date)
        if len(members) < _MIN_RECURRENCE_OCCURRENCES:
            continue
        amounts = [float(m.amount_home) for m in members]
        med_amount = median(amounts)
        if med_amount <= 0:
            continue
        latest = amounts[-1]
        # Guard: latest amount should be within a reasonable ratio of the median
        ratio = latest / med_amount if med_amount > 0 else 0
        if ratio > _AMOUNT_MAX_RATIO_TO_MEDIAN:
            # Outlier latest — drop it and re-check with the rest
            members = members[:-1]
            if len(members) < _MIN_RECURRENCE_OCCURRENCES:
                continue
        series[key] = members
    return series


# ---------------------------------------------------------------------------
# Project future events
# ---------------------------------------------------------------------------
def _project_recurring(
    series: dict[tuple, list[LedgerEntry]],
    start_date: date,
    end_date: date,
) -> list[FutureEvent]:
    """
    For each recurring series, project future occurrences using the MEDIAN
    historical amount (robust to one-off bonuses/raises) and the dominant
    interval.
    """
    projected: list[FutureEvent] = []

    for key, entries in series.items():
        hist_dates = [e.event_date for e in entries]
        interval = _infer_interval_days(hist_dates)
        if interval is None:
            continue

        last = entries[-1]
        # Use the MEDIAN amount so one-off bonuses don't skew the projection
        amt_home = Decimal(str(median(float(e.amount_home) for e in entries)))
        direction = last.direction
        signed = -amt_home if direction == "debit" else amt_home

        # Anchor on the most recent date so we don't repeat the past
        anchor_date = last.event_date

        next_date = anchor_date + timedelta(days=interval)
        while next_date <= end_date:
            if next_date > start_date:
                projected.append(
                    FutureEvent(
                        date=next_date,
                        amount_home=signed,
                        event_id=last.event_id,
                        category=last.category or last.event_type,
                        description=f"projected {last.category or last.event_type}",
                        is_recurring=True,
                    )
                )
            next_date = next_date + timedelta(days=interval)

    return projected

def _project_one_off_future(
    entries: list[LedgerEntry],
    start_date: date,
    end_date: date,
) -> list[FutureEvent]:
    """
    Non-recurring events with start_date < event_date <= end_date are
    applied as-is. These represent one-off scheduled future payments.
    """
    out: list[FutureEvent] = []
    for e in entries:
        if e.is_recurring:
            continue
        if start_date < e.event_date <= end_date:
            signed = -e.amount_home if e.direction == "debit" else e.amount_home
            out.append(
                FutureEvent(
                    date=e.event_date,
                    amount_home=signed,
                    event_id=e.event_id,
                    category=e.category or e.event_type,
                    description=e.category or e.event_type,
                    is_recurring=False,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Main: build forecast
# ---------------------------------------------------------------------------
def build_forecast(
    state: UserFinancialState,
    start_date: date,
    horizon_days: int = FORECAST_DAYS,
) -> Forecast:
    """
    Build a 90-day balance forecast for `state`, starting at `start_date`.
    The starting balance is the profile's `current_available_balance`.

    NOTE: We assume `current_available_balance` in the profile already
    reflects all SETTLED events on/before `start_date`. So we do NOT
    retroactively apply settled historical events. We only project FUTURE
    recurring and one-off events.
    """
    end_date = start_date + timedelta(days=horizon_days)

    all_entries = state.entries
    series = _build_series(all_entries)

    projected_recurring = _project_recurring(series, start_date, end_date)
    projected_oneoff = _project_one_off_future(all_entries, start_date, end_date)

    all_future = projected_recurring + projected_oneoff
    all_future.sort(key=lambda e: (e.date, e.event_id))

    by_date: dict[date, list[FutureEvent]] = {}
    for e in all_future:
        by_date.setdefault(e.date, []).append(e)

    points: list[BalancePoint] = []
    bal = state.current_balance
    for d in sorted(by_date.keys()):
        for e in by_date[d]:
            bal = bal + e.amount_home
        points.append(BalancePoint(date=d, balance_after_day=bal, events=by_date[d]))

    return Forecast(
        start_date=start_date,
        end_date=end_date,
        starting_balance=state.current_balance,
        minimum_balance=state.minimum_balance_to_keep,
        points=points,
    )


if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO)
    from datetime import date as _d
    from financial_state import build_user_state

    for uid, dt in [("user_01", _d(2024, 3, 3)), ("user_02", _d(2025, 8, 5)),
                    ("user_03", _d(2019, 9, 3))]:
        st = build_user_state(uid, dt)
        fc = build_forecast(st, dt)
        print(f"\n=== Forecast for {uid} from {dt} to {fc.end_date} ===")
        print(f"  starting balance: {fc.starting_balance}")
        print(f"  minimum balance : {fc.minimum_balance}")
        print(f"  #forecast points: {len(fc.points)}")
        print(f"  #events total   : {sum(len(p.events) for p in fc.points)}")
        print(f"  min balance in window: {fc.min_balance_in_window()}")
        print(f"  first date below min : {fc.first_date_below_minimum()}")
        print(f"  safe? {fc.is_safe()}")
        print("  first 15 events:")
        shown = 0
        for p in fc.points:
            for e in p.events:
                print(f"    {p.date}  {e.amount_home:>15}  {e.description}")
                shown += 1
                if shown >= 15:
                    break
            if shown >= 15:
                break