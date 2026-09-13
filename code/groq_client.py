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

VLM strategy:
  Qwen 3.6 27B has a small OTPM cap on the free tier and its
  `response_format=json_object` mode sometimes returns empty output when
  the model's reasoning consumes the max_tokens budget. We therefore:
    - use a generous max_tokens (800)
    - keep json_object mode, but retry once WITHOUT it if we get an
      empty/invalid response, relying on _safe_json_loads to extract
      the JSON from plain text.
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
_VLM_JSON_MAX_TOKENS = 800


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
        Send one local PNG to Qwen 3.6 27B, request JSON.

        Strategy:
          1. First try with response_format=json_object
          2. If the response is empty/unparseable, retry WITHOUT
             response_format (plain text), relying on _safe_json_loads.
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
            # Alternate: json_object on odd attempts, plain on even
            use_json_mode = (attempt % 2 == 1)
            try:
                kwargs: dict[str, Any] = dict(
                    model=self.vlm_model,
                    messages=messages,
                    temperature=0,
                    max_tokens=_VLM_JSON_MAX_TOKENS,
                )
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                resp = self._client.chat.completions.create(**kwargs)
                self._track(resp, purpose, request_id)
                content = (resp.choices[0].message.content or "").strip()
                if not content:
                    raise GroqError("empty VLM response")
                parsed = _safe_json_loads(content)
                if not parsed:
                    raise GroqError(f"unparseable VLM response: {content[:120]!r}")
                return parsed
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning(
                    "VLM call failed (attempt %d/%d, json_mode=%s, purpose=%s): %s",
                    attempt, max_retries, use_json_mode, purpose, e,
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
    Robust JSON parse for LLM output.

    Handles:
      - ```json ... ``` code fences
      - <think>...</think> reasoning blocks (Qwen 3.x)
      - leading prose before the JSON object
      - trailing prose after the JSON object

    Returns empty dict if unrecoverable — callers MUST validate.
    """
    if not text:
        return {}

    import re
    t = text.strip()

    # 1. Strip any <think>...</think> block (Qwen reasoning)
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE).strip()
    # Also strip a lone dangling <think> with no close
    t = re.sub(r"<think>", "", t, flags=re.IGNORECASE).strip()

    # 2. Strip ```json ... ``` or ``` ... ``` fences
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()

    # 3. Direct parse
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
        return {"_value": obj}
    except json.JSONDecodeError:
        pass

    # 4. Last resort: extract the first balanced {...} block
    start = t.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = t[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break

    log.warning("Could not parse LLM JSON output: %r", text[:200])
    return {}