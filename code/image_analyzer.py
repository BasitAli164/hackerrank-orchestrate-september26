"""
Extract financial amounts from event images using Qwen 3.6 27B (vision).

Called ONLY when:
  - financial_events.amount is blank, AND
  - an image in images.csv links to that event via related_event_id

Returns a structured amount + currency + date + description, or None if
the extraction fails or the amount is not clearly visible.

Never invents a value: if the model is uncertain, the fact is dropped
and the caller falls back to the deterministic engine's safe handling.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Optional

import pandas as pd

from config import MEDIA_DIR
from groq_client import GroqClient, GroqError
from data_cleaner import get as get_clean

log = logging.getLogger(__name__)


_IMAGE_PROMPT = """Extract financial information from this image.

Return a JSON object with EXACTLY these keys:
  - "amount":   number or null    (positive magnitude; null if not visible)
  - "currency": string or null    (3-letter ISO code; null if not visible)
  - "date":     "YYYY-MM-DD" or null
  - "description": string or null (short label for what this is)

Rules:
  - Do NOT guess. If a value is not clearly visible, use null.
  - The image may be a receipt, invoice, screenshot, or bank statement.
  - Prefer the TOTAL or final amount when multiple amounts appear.
  - Return ONLY the JSON object.
"""


@dataclass
class ImageFact:
    image_id: str
    event_id: str
    user_id: str
    amount: Decimal
    currency: str
    date: Optional[date]
    description: str


class ImageAnalyzer:
    def __init__(self, client: Optional[GroqClient] = None) -> None:
        self.client = client or GroqClient()
        self._cache: dict[str, Optional[ImageFact]] = {}
        self._images_by_event: dict[str, str] | None = None

    def _event_to_image(self) -> dict[str, str]:
        if self._images_by_event is not None:
            return self._images_by_event
        df = get_clean("images")
        mapping: dict[str, str] = {}
        for _, r in df.iterrows():
            ev = r["related_event_id"]
            img = r["image_id"]
            if pd.notna(ev) and pd.notna(img):
                mapping[str(ev)] = str(img)
        self._images_by_event = mapping
        return mapping

    def find_image_for_event(self, event_id: str) -> Optional[str]:
        return self._event_to_image().get(event_id)

    def analyze_for_event(
        self,
        event_id: str,
        user_id: str,
        request_id: Optional[str] = None,
    ) -> Optional[ImageFact]:
        if event_id in self._cache:
            return self._cache[event_id]

        image_id = self.find_image_for_event(event_id)
        if image_id is None:
            self._cache[event_id] = None
            return None

        path = MEDIA_DIR / f"{image_id}.png"
        if not path.is_file():
            log.warning("Image file missing for %s: %s", image_id, path)
            self._cache[event_id] = None
            return None

        try:
            raw = self.client.vlm_json_from_image(
                image_path=str(path),
                prompt=_IMAGE_PROMPT,
                purpose="image_extraction",
                request_id=request_id,
            )
        except GroqError as e:
            log.warning("VLM call failed for %s: %s", image_id, e)
            self._cache[event_id] = None
            return None

        fact = self._validate(image_id, event_id, user_id, raw)
        self._cache[event_id] = fact
        return fact

    def _validate(
        self,
        image_id: str,
        event_id: str,
        user_id: str,
        raw: dict,
    ) -> Optional[ImageFact]:
        if not isinstance(raw, dict):
            return None

        raw_amount = raw.get("amount")
        if raw_amount is None:
            return None
        try:
            amount = Decimal(str(raw_amount))
        except (InvalidOperation, ValueError):
            return None
        if amount <= 0:
            return None

        currency = raw.get("currency")
        if not currency or not isinstance(currency, str):
            return None
        currency = currency.strip().upper()[:8]
        if len(currency) != 3 or not currency.isalpha():
            return None

        date_val: Optional[date] = None
        raw_date = raw.get("date")
        if raw_date:
            try:
                date_val = pd.to_datetime(raw_date, errors="raise").date()
            except Exception:
                date_val = None

        description = str(raw.get("description") or "").strip()[:200]

        return ImageFact(
            image_id=image_id,
            event_id=event_id,
            user_id=user_id,
            amount=amount,
            currency=currency,
            date=date_val,
            description=description,
        )


if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO)

    analyzer = ImageAnalyzer()
    events = get_clean("events")
    blank = events[events["amount"].isna()]
    print(f"Events with blank amount:  {len(blank)}")
    print(f"Events with linked image: {len(analyzer._event_to_image())}")

    for _, row in blank.head(5).iterrows():
        ev_id = str(row["event_id"])
        user_id = str(row["user_id"])
        print(f"\n=== {ev_id} ({user_id}) ===")
        print(f"  desc: {row['description']}")
        fact = analyzer.analyze_for_event(ev_id, user_id)
        if fact:
            print(f"  amount: {fact.amount} {fact.currency}")
            print(f"  date:   {fact.date}")
            print(f"  desc:   {fact.description}")
        else:
            print("  (no fact extracted)")
