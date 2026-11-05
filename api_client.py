"""Async OpenAI-compatible chat client used by the text-CoT pipeline.

Design notes
------------
* One process owns one event loop; concurrency is bounded by a semaphore.
* Multiple endpoints are supported and rotated round-robin (each endpoint has
  its own semaphore, mirroring `src/rl/scripts/mmd_judge_client.py`).
* Retries with exponential backoff + jitter for 429 / 5xx / timeout / network
  errors. 429 waits >= 5s (data-safety rule 7).
* Reasoning is normalized from the several shapes providers use:
    - message.reasoning            (vLLM DeepSeek / R1 style)
    - message.reasoning_content    (some gateways)
    - content containing  thinking...</think>
    - content without tags         (everything before <answer>)
* Token accounting is accumulated so the caller can report cost/length stats.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import aiohttp

THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


@dataclass
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 16384
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class Completion:
    uid: str
    sample_idx: int
    content: str
    reasoning: str
    reasoning_source: str  # "field" | "think_tag" | "answer_split" | "none"
    answer_text: str
    finish_reason: str
    usage: dict[str, int]
    latency_s: float
    model: str
    endpoint: str
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "sample_idx": self.sample_idx,
            "content": self.content,
            "reasoning": self.reasoning,
            "answer_text": self.answer_text,
            "reasoning_source": self.reasoning_source,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "latency_s": round(self.latency_s, 3),
            "model": self.model,
            "endpoint": self.endpoint,
            "error": self.error,
        }


def split_reasoning(message: dict[str, Any], content: str) -> tuple[str, str, str]:
    """Return (reasoning, answer_text, reasoning_source)."""
    for key in ("reasoning", "reasoning_content", "thinking"):
        val = message.get(key)
        if isinstance(val, str) and val.strip():
            ans = ANSWER_RE.search(content)
            answer_text = ans.group(1).strip() if ans else content.strip()
            return val.strip(), answer_text, "field"

    m = THINK_RE.search(content)
    if m:
        answer_text = content[m.end():].strip()
        ans = ANSWER_RE.search(answer_text)
        if ans:
            answer_text = ans.group(1).strip()
        return m.group(1).strip(), answer_text, "think_tag"

    ans = ANSWER_RE.search(content)
    if ans and content[: ans.start()].strip():
        return content[: ans.start()].strip(), ans.group(1).strip(), "answer_split"

    return "", content.strip(), "none"


class Endpoint:
    def __init__(self, url: str, max_concurrency: int):
        self.url = url
        self.cap = max_concurrency
        self.sem = asyncio.Semaphore(max_concurrency)
        self.inflight = 0
        self.ok = 0
        self.fail = 0


class ChatClient:
    def __init__(
        self,
        base_urls: Iterable[str],
        model: str,
        api_key: str,
        *,
        max_concurrency_per_endpoint: int = 16,
        rpm: int | None = None,
        timeout_s: float = 600.0,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        proxy: str | None = None,
    ):
        urls = [u.rstrip("/") for u in base_urls if u.strip()]
        if not urls:
            raise ValueError("at least one base_url is required")
        self.endpoints = [Endpoint(u, max_concurrency_per_endpoint) for u in urls]
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.proxy = proxy
        self._rr = 0
        self._rpm = rpm
        self._next_slot = 0.0
        self._rpm_lock = asyncio.Lock()
        self.stats = {
            "requests": 0,
            "ok": 0,
            "failed": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
        }
        self._session: aiohttp.ClientSession | None = None

    @property
    def capacity(self) -> int:
        """Total in-flight budget across endpoints."""
        return sum(ep.cap for ep in self.endpoints)

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout_s),
            connector=aiohttp.TCPConnector(limit=0, force_close=False),
        )
        return self

    async def __aexit__(self, *exc):
        if self._session:
            await self._session.close()

    def _pick(self) -> Endpoint:
        ep = self.endpoints[self._rr % len(self.endpoints)]
        self._rr += 1
        return ep

    async def _respect_rpm(self):
        if not self._rpm:
            return
        async with self._rpm_lock:
            interval = 60.0 / self._rpm
            now = time.monotonic()
            wait = self._next_slot - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_slot = max(now, self._next_slot) + interval

    async def chat(
        self,
        messages: list[dict[str, Any]],
        sampling: SamplingParams,
        *,
        uid: str = "",
        sample_idx: int = 0,
        extra_fields: dict[str, Any] | None = None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "stream": False,
        }
        payload.update(sampling.extra_body or {})
        if extra_fields:
            payload.update(extra_fields)

        last_err = None
        for attempt in range(self.max_retries + 1):
            ep = self._pick()
            await self._respect_rpm()
            t0 = time.monotonic()
            try:
                async with ep.sem:
                    ep.inflight += 1
                    self.stats["requests"] += 1
                    try:
                        async with self._session.post(
                            f"{ep.url}/chat/completions",
                            json=payload,
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                            },
                            proxy=self.proxy,
                        ) as resp:
                            body = await resp.text()
                            status = resp.status
                    finally:
                        ep.inflight -= 1

                if status == 200:
                    data = json.loads(body)
                    msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
                    finish = (data.get("choices") or [{}])[0].get("finish_reason", "") or ""
                    content = msg.get("content") or ""
                    if isinstance(content, list):
                        content = "".join(
                            p.get("text", "") for p in content if isinstance(p, dict)
                        )
                    reasoning, answer_text, src = split_reasoning(msg, content)
                    usage = data.get("usage") or {}
                    cdetails = usage.get("completion_tokens_details") or {}
                    dt = time.monotonic() - t0
                    ep.ok += 1
                    self.stats["ok"] += 1
                    self.stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                    self.stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                    self.stats["reasoning_tokens"] += int(cdetails.get("reasoning_tokens") or 0)
                    return Completion(
                        uid=uid,
                        sample_idx=sample_idx,
                        content=content,
                        reasoning=reasoning,
                        reasoning_source=src,
                        answer_text=answer_text,
                        finish_reason=finish,
                        usage={
                            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                            "completion_tokens": int(usage.get("completion_tokens") or 0),
                            "reasoning_tokens": int(cdetails.get("reasoning_tokens") or 0),
                        },
                        latency_s=dt,
                        model=data.get("model", self.model),
                        endpoint=ep.url,
                    )

                # Non-retryable client errors
                if 400 <= status < 500 and status not in (408, 409, 429):
                    ep.fail += 1
                    self.stats["failed"] += 1
                    return Completion(
                        uid=uid, sample_idx=sample_idx, content="", reasoning="",
                        answer_text="", reasoning_source="none", finish_reason="error", usage={},
                        latency_s=time.monotonic() - t0, model=self.model,
                        endpoint=ep.url, error=f"HTTP {status}: {body[:500]}",
                    )

                last_err = f"HTTP {status}: {body[:300]}"
                if status == 429:
                    wait = max(5.0, self.backoff_base ** attempt)
                    wait = max(wait, 5.0 + random.random())
                else:
                    wait = min(60.0, self.backoff_base ** attempt) + random.random()
            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                wait = min(60.0, self.backoff_base ** attempt) + random.random()
            except Exception as exc:  # noqa: BLE001 - never kill the whole batch
                last_err = f"{type(exc).__name__}: {exc}"
                wait = min(60.0, self.backoff_base ** attempt) + random.random()

            ep.fail += 1
            self.stats["failed"] += 1
            if attempt < self.max_retries:
                print(f"[api] retry {attempt + 1}/{self.max_retries} uid={uid} "
                      f"sample={sample_idx} in {wait:.1f}s: {last_err}", flush=True)
                await asyncio.sleep(wait)

        return Completion(
            uid=uid, sample_idx=sample_idx, content="", reasoning="",
            answer_text="", reasoning_source="none", finish_reason="error", usage={},
            latency_s=0.0, model=self.model, endpoint="", error=last_err,
        )
