"""
Tracks every Groq API call: model, purpose, tokens, cost.
Writes a summary to code/evaluation/usage_report.md at the end of a run.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional

from config import MODEL_PRICING, USAGE_REPORT_MD


@dataclass
class CallRecord:
    model: str
    purpose: str
    request_id: Optional[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float


@dataclass
class _Bucket:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class UsageTracker:
    """
    Thread-safe (in case we parallelize later) usage accumulator.
    One instance per process, injected into the Groq client.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[CallRecord] = []

    def record(
        self,
        model: str,
        purpose: str,
        request_id: Optional[str],
        input_tokens: int,
        output_tokens: int,
    ) -> CallRecord:
        total = int(input_tokens or 0) + int(output_tokens or 0)
        pricing = MODEL_PRICING.get(model)
        if pricing:
            cost = (
                (input_tokens or 0) / 1_000_000 * pricing["input"]
                + (output_tokens or 0) / 1_000_000 * pricing["output"]
            )
        else:
            cost = 0.0

        rec = CallRecord(
            model=model,
            purpose=purpose,
            request_id=request_id,
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            total_tokens=total,
            estimated_cost_usd=round(cost, 8),
        )
        with self._lock:
            self._records.append(rec)
        return rec

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def _aggregate(self) -> dict[str, _Bucket]:
        buckets: dict[str, _Bucket] = {}
        for r in self._records:
            b = buckets.setdefault(r.model, _Bucket())
            b.calls += 1
            b.input_tokens += r.input_tokens
            b.output_tokens += r.output_tokens
            b.total_tokens += r.total_tokens
            b.cost_usd += r.estimated_cost_usd
        return buckets

    def write_report(self, total_requests: int, path=USAGE_REPORT_MD) -> None:
        """
        Write usage_report.md summarizing the final full-dataset run.
        Distinguishes per-model breakdown and totals.
        """
        buckets = self._aggregate()
        path.parent.mkdir(parents=True, exist_ok=True)

        grand_calls = sum(b.calls for b in buckets.values())
        grand_in = sum(b.input_tokens for b in buckets.values())
        grand_out = sum(b.output_tokens for b in buckets.values())
        grand_tok = sum(b.total_tokens for b in buckets.values())
        grand_cost = sum(b.cost_usd for b in buckets.values())

        lines: list[str] = []
        lines.append("# Groq Token & Cost Usage Report\n")
        lines.append(f"Requests processed: **{total_requests}**\n")
        lines.append("## Per-model breakdown\n")

        for model, b in sorted(buckets.items()):
            lines.append(f"### `{model}`\n")
            lines.append(f"- Calls: **{b.calls}**")
            lines.append(f"- Input tokens: **{b.input_tokens:,}**")
            lines.append(f"- Output tokens: **{b.output_tokens:,}**")
            lines.append(f"- Total tokens: **{b.total_tokens:,}**")
            lines.append(f"- Estimated cost (USD): **${b.cost_usd:.6f}**")
            if total_requests:
                lines.append(
                    f"- Average tokens/request: **{b.total_tokens / total_requests:.2f}**"
                )
                lines.append(
                    f"- Average cost/request (USD): **${b.cost_usd / total_requests:.6f}**"
                )
            lines.append("")

        lines.append("## Overall\n")
        lines.append(f"- Total calls: **{grand_calls}**")
        lines.append(f"- Total input tokens: **{grand_in:,}**")
        lines.append(f"- Total output tokens: **{grand_out:,}**")
        lines.append(f"- Total tokens: **{grand_tok:,}**")
        lines.append(f"- Total estimated cost (USD): **${grand_cost:.6f}**")
        if total_requests:
            lines.append(
                f"- Average tokens/request: **{grand_tok / total_requests:.2f}**"
            )
            lines.append(
                f"- Average cost/request (USD): **${grand_cost / total_requests:.6f}**"
            )
        lines.append("")
        lines.append(
            "_Pricing is configurable in `code/config.py` (`MODEL_PRICING`). "
            "Token counts come directly from Groq API responses._"
        )

        path.write_text("\n".join(lines), encoding="utf-8")

    def dump_json(self) -> str:
        return json.dumps([asdict(r) for r in self._records], indent=2)


# Module-level singleton
_tracker: Optional[UsageTracker] = None


def get_tracker() -> UsageTracker:
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker()
    return _tracker