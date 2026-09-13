"""
Single reusable Groq client.
Handles:
  - loading GROQ_API_KEY from .env
  - text calls to GPT-OSS 120B (LLM_MODEL)
  - vision calls to Qwen 3.6 27B (VLM_MODEL)
  - usage tracking on every call
  - retry with bounded attempts on transient failures

Token caps are enforced at each call site to stay under per-minute limits
on the free tier (especially Qwen's 1000 OTPM cap).
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any, Optional

from dotenv import load_dotenv
from groq import Groq

from config import (
    LLM_MODEL,
    VLM_MODEL,
    LLM_REASONING_EFFORT,
    MAX_VLM_IMAGE_BYTES,
)
from usage_tracker import get_tracker

log = logging.getLogger(__name__)

# Load .env once at import
load_dotenv()

# Per-call max_tokens caps (safety against OTPM limits on free tier)
_LLM_JSON_MAX_TOKENS = 2000
_LLM_TEXT_MAX_TOKENS = 600
_VLM_JSON_MAX_TOKENS = 400


class GroqError(RuntimeError):
    """Raised when Groq call fails after retries or config is invalid."""


class GroqClient:
    """
    Thin wrapper around groq.Groq with retries and usage tracking.
    Instantiate once per run and reuse.
    """

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise GroqError("GROQ_API_KEY is not configured. Add it to .env")
        self._client = Groq(api_key=api_key)
        self.llm_model = LLM_MODEL
        self.vlm_model = VLM_MODEL
        self.tracker = get_tracker()

    # ------------------------------------------------------------------
    # TEXT (GPT-OSS 120B)
    # ------------------------------------------------------------------
    def llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
        purpose: str,
        request_id: Optional[str] = None,
        schema: Optional[dict] = None,
        max_retries: int = 3,
    ) -> dict:
        """
        Call GPT-OSS 120B expecting a JSON object response.

        If `schema` is provided, uses Groq structured-outputs json_schema mode.
        Otherwise falls back to json_object mode.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_format: dict[str, Any]
        if schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_extraction",
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            response_format = {"type": "json_object"}

        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    response_format=response_format,
                    reasoning_effort=LLM_REASONING_EFFORT,
                    temperature=0,
                    max_tokens=_LLM_JSON_MAX_TOKENS,
                )
                self._track(resp, purpose, request_id)
                content = resp.choices[0].message.content or "{}"
                return _safe_json_loads(content)
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning(
                    "LLM call failed (attempt %d/%d, purpose=%s): %s",
                    attempt, max_retries, purpose, e,
                )
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)
        raise GroqError(f"LLM call failed after {max_retries} attempts: {last_err}")

    def llm_text(
        self,
        system_prompt: str,
        user_prompt: str,
        purpose: str,
        request_id: Optional[str] = None,
        max_retries: int = 3,
    ) -> str:
        """Plain text response (used for final explanation)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=0,
                    reasoning_effort="low",
                    max_tokens=_LLM_TEXT_MAX_TOKENS,
                )
                self._track(resp, purpose, request_id)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning(
                    "LLM text call failed (attempt %d/%d, purpose=%s): %s",
                    attempt, max_retries, purpose, e,
                )
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)
        raise GroqError(f"LLM text call failed after {max_retries} attempts: {last_err}")

    # ------------------------------------------------------------------
    # VISION (Qwen 3.6 27B)
    # ------------------------------------------------------------------
    def vlm_json_from_image(
        self,
        image_path: str,
        prompt: str,
        purpose: str,
        request_id: Optional[str] = None,
        max_retries: int = 3,
    ) -> dict:
        """
        Send one local PNG to Qwen 3.6 27B, request JSON object output.
        Enforces image size limit before sending.
        """
        path = os.fspath(image_path)
        if not os.path.isfile(path):
            raise GroqError(f"Image not found: {path}")

        size = os.path.getsize(path)
        if size > MAX_VLM_IMAGE_BYTES:
            raise GroqError(
                f"Image too large: {size} bytes (limit {MAX_VLM_IMAGE_BYTES})"
            )

        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.vlm_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=_VLM_JSON_MAX_TOKENS,
                )
                self._track(resp, purpose, request_id)
                content = resp.choices[0].message.content or "{}"
                return _safe_json_loads(content)
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning(
                    "VLM call failed (attempt %d/%d, purpose=%s): %s",
                    attempt, max_retries, purpose, e,
                )
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)
        raise GroqError(f"VLM call failed after {max_retries} attempts: {last_err}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _track(self, resp: Any, purpose: str, request_id: Optional[str]) -> None:
        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        model = getattr(resp, "model", None) or self.llm_model
        self.tracker.record(
            model=model,
            purpose=purpose,
            request_id=request_id,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )


def _safe_json_loads(text: str) -> dict:
    """
    Robust JSON parse for LLM output. Strips code fences if present.
    Returns empty dict if unrecoverable — callers MUST validate.
    """
    t = text.strip()
    if t.startswith("```"):
        # strip ```json ... ```
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
        return {"_value": obj}
    except json.JSONDecodeError:
        # last resort: extract first {...} block
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(t[start : end + 1])
            except json.JSONDecodeError:
                pass
    log.warning("Could not parse LLM JSON output: %r", text[:200])
    return {}