#!/usr/bin/env python3
"""Text-QA chain-of-thought construction pipeline (teacher-API distillation).

Stages
------
  ingest    raw QA (parquet / jsonl / json) -> canonical records.jsonl
  generate  N sampled CoTs per question from an OpenAI-compatible teacher API
  verify    answer-check + quality filters + best-CoT selection
  build     accepted traces -> SFT jsonl/parquet in OmniRL format + report
  stats     print pipeline statistics
  all       ingest -> generate -> verify -> build

Everything is resumable: finished (uid, sample_idx) pairs are skipped by
re-reading the generation shards, so you can Ctrl-C and re-run at any time.

Examples
--------
  python pipeline.py all --config config.yaml --limit 500
  python pipeline.py generate --config config.yaml --num-shards 8 --shard-id 0
  python pipeline.py verify --config config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import random
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml  # noqa: E402

from answer_check import (  # noqa: E402
    check_answer,
    extract_answer,
    guess_problem_type,
    quality_check,
)
from api_client import (  # noqa: E402
    THINK_RE,
    ChatClient,
    Completion,
    SamplingParams,
    split_reasoning,
)


# ── config / io helpers ───────────────────────────────────────────────────────
def load_config(path: str | None) -> dict[str, Any]:
    cfg_path = Path(path or (Path(__file__).parent / "config.yaml"))
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def resolve_path(p: str | os.PathLike) -> Path:
    """Relative paths resolve against the current working directory, so the
    pipeline is runnable from any project checkout."""
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else (Path.cwd() / p)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ── stage: ingest ─────────────────────────────────────────────────────────────
def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text", "")) for p in content if isinstance(p, dict)
        )
    return "" if content is None else str(content)


def _guess_problem_type(question: str, gt: str) -> str:
    return guess_problem_type(question, gt)


def _record_from_row(row: dict[str, Any], idx: int, source: str) -> dict[str, Any] | None:
    question, gt, ptype, dsource = None, None, None, None

    # verl RL parquet schema
    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        for m in reversed(prompt):
            if isinstance(m, dict) and m.get("role") == "user":
                question = _content_to_text(m.get("content"))
                break
        if question is None:
            question = _content_to_text(prompt[-1].get("content") if isinstance(prompt[-1], dict) else prompt[-1])
    rm = row.get("reward_model")
    if isinstance(rm, dict):
        gt = rm.get("ground_truth")
    ei = row.get("extra_info")
    if isinstance(ei, dict):
        ptype = ei.get("problem_type")
        dsource = ei.get("data_source")
        question = question or ei.get("question") or ei.get("query")

    # plain jsonl schema
    if question is None:
        for k in ("question", "query", "instruction", "problem", "input"):
            if isinstance(row.get(k), str) and row[k].strip():
                question = row[k]
                break
    if question is None and isinstance(row.get("messages"), list):
        for m in reversed(row["messages"]):
            if isinstance(m, dict) and m.get("role") == "user":
                question = _content_to_text(m.get("content"))
                break
    if gt is None:
        for k in ("answer", "ground_truth", "gt", "solution", "label", "target", "output"):
            if isinstance(row.get(k), str) and row[k].strip():
                gt = row[k]
                break
    if gt is None and isinstance(row.get("messages"), list):
        for m in reversed(row["messages"]):
            if isinstance(m, dict) and m.get("role") == "assistant":
                gt = _content_to_text(m.get("content"))
                break

    if not question or gt is None or str(gt).strip() == "":
        return None
    gt = str(gt).strip()
    ptype = (ptype or row.get("problem_type") or _guess_problem_type(question, gt)).strip()
    guessed = _guess_problem_type(question, gt)
    # Dataset metadata is usually right, but a bare-letter gold answer is
    # unambiguously multiple-choice even if the source labelled it free-form.
    if guessed == "multiple choice" and ptype.lower() not in ("multiple choice", "multiple_choice", "mc"):
        ptype = guessed
    dsource = (dsource or row.get("data_source") or row.get("source") or source).strip()
    return {
        "uid": f"{source}:{idx}",
        "question": question.strip(),
        "ground_truth": gt,
        "problem_type": ptype,
        "data_source": dsource,
    }


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    suffix = path.suffix.lower()
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for f in sorted(path.glob("*.parquet")):
            rows.extend(_load_rows(f)[0])
        return rows, path.name
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist(), path.stem
    if suffix in (".jsonl", ".json"):
        if suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            rows = data if isinstance(data, list) else data.get("data", [])
            return rows, path.stem
        return read_jsonl(path), path.stem
    raise ValueError(f"unsupported input: {path}")


def stage_ingest(cfg: dict[str, Any], args) -> Path:
    work_dir = resolve_path(cfg["paths"]["work_dir"])
    inputs = [resolve_path(p) for p in cfg["paths"]["inputs"]]
    out = work_dir / "records.jsonl"
    if out.exists() and not args.force:
        print(f"[ingest] reuse existing {out}")
        return out

    ingest_cfg = cfg.get("ingest", {})
    max_records = int(ingest_cfg.get("max_records", 0) or 0)
    dedup_questions = bool(ingest_cfg.get("dedup_questions", False))
    max_question_chars = int(ingest_cfg.get("max_question_chars", 0) or 0)

    records, skipped, dup = [], collections.Counter(), 0
    seen_questions: set[str] = set()
    for path in inputs:
        rows, stem = _load_rows(path)
        for i, row in enumerate(rows):
            rec = _record_from_row(row, i, stem)
            if rec is None:
                skipped["empty"] += 1
                continue
            if max_question_chars and len(rec["question"]) > max_question_chars:
                skipped["question_too_long"] += 1
                continue
            if dedup_questions:
                qkey = _norm_key(rec["question"])
                if qkey in seen_questions:
                    skipped["duplicate_question"] += 1
                    dup += 1
                    continue
                seen_questions.add(qkey)
            records.append(rec)

    # NOTE: --limit is applied at generate/verify time, never persisted here,
    # otherwise a later full run would silently reuse the truncated file.
    if max_records > 0:
        records = records[:max_records]
    write_jsonl(out, records)
    by_type = collections.Counter(r["problem_type"] for r in records)
    print(f"[ingest] {len(records)} records -> {out} (skipped {dict(skipped) or 'none'})")
    print(f"[ingest] problem_type: {dict(by_type)}")
    return out


# ── stage: generate ───────────────────────────────────────────────────────────
def _build_messages(cfg: dict[str, Any], question: str) -> list[dict[str, str]]:
    gen = cfg.get("generate", {})
    msgs = []
    if gen.get("system_prompt"):
        msgs.append({"role": "system", "content": gen["system_prompt"]})
    msgs.append({"role": "user", "content": f"{question}{gen.get('user_suffix', '')}"})
    return msgs


def _sampling_for(cfg: dict[str, Any], sample_idx: int) -> SamplingParams:
    s = cfg.get("sampling", {})
    temps = s.get("temperatures") or [s.get("temperature", 0.7)]
    return SamplingParams(
        temperature=float(temps[sample_idx % len(temps)]),
        top_p=float(s.get("top_p", 0.95)),
        max_tokens=int(s.get("max_tokens", 16384)),
        extra_body=s.get("extra_body") or {},
    )


def _shard_path(work_dir: Path, shard_id: int) -> Path:
    return work_dir / "gen" / f"shard_{shard_id:03d}.jsonl"


def _shard_of(uid: str, num_shards: int) -> int:
    return zlib.crc32(uid.encode("utf-8")) % num_shards


def _seed_for(uid: str, sample_idx: int, base: int = 0) -> int:
    """Deterministic per-(question, sample) seed so a rerun reproduces the same trace."""
    return (base + zlib.crc32(uid.encode("utf-8")) * 131 + sample_idx * 7919) % (2**31 - 1)


class JsonlWriter:
    """Line-buffered append writer; safe to kill at any point (last line may be partial)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", buffering=1)

    def write(self, obj: dict[str, Any]) -> None:
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def close(self) -> None:
        try:
            self.f.flush()
            os.fsync(self.f.fileno())
        finally:
            self.f.close()


def _completed_keys(work_dir: Path, shard_id: int | None) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    files = ([_shard_path(work_dir, shard_id)] if shard_id is not None
             else sorted((work_dir / "gen").glob("shard_*.jsonl")))
    for f in files:
        if not f.exists():
            continue
        with open(f) as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not o.get("error"):
                    done.add((o["uid"], int(o["sample_idx"])))
    return done


async def stage_generate(cfg: dict[str, Any], args) -> None:
    work_dir = resolve_path(cfg["paths"]["work_dir"])
    records = read_jsonl(work_dir / "records.jsonl")
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    n_samples = int(cfg["sampling"].get("n_samples", 4))
    num_shards = max(1, int(args.num_shards))
    shard_id = int(args.shard_id)
    assert 0 <= shard_id < num_shards

    # Scan every shard (not just this one): if sharding was changed between runs
    # (e.g. first unsharded, then --num-shards 4) we must not re-bill completed
    # (uid, sample_idx) pairs.
    done = _completed_keys(work_dir, None)
    tasks = []
    for r in records:
        if num_shards > 1 and _shard_of(r["uid"], num_shards) != shard_id:
            continue
        for k in range(n_samples):
            if (r["uid"], k) not in done:
                tasks.append((r["uid"], k))
    random.shuffle(tasks)

    out_path = _shard_path(work_dir, shard_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    api_cfg = cfg.get("api", {}) or {}

    # ── endpoint / model / key come from the environment (never committed) ──
    #   TEXT_COT_API_BASES   comma-separated OpenAI-compatible base URLs (wins)
    #   TEXT_COT_API_BASE    single base URL
    #   TEXT_COT_API_MODEL   model name served by the endpoint
    #   TEXT_COT_API_KEY     bearer token
    # Config keys api.base_urls / api.base_url / api.model / api.api_key remain
    # as optional overrides for setups that prefer explicit config files.
    base_s = os.environ.get("TEXT_COT_API_BASES", "").strip()
    base_1 = os.environ.get("TEXT_COT_API_BASE", "").strip()
    env_model = os.environ.get("TEXT_COT_API_MODEL", "").strip()
    env_key = os.environ.get("TEXT_COT_API_KEY", "").strip()

    urls: list[str] = []
    if base_s:
        urls = [u.strip() for u in base_s.split(",") if u.strip()]
    elif base_1:
        urls = [base_1]
    elif api_cfg.get("base_urls"):
        urls = list(api_cfg["base_urls"])
    elif api_cfg.get("base_url"):
        urls = [api_cfg["base_url"]]
    model = env_model or args.model or api_cfg.get("model")
    key = env_key or os.environ.get(api_cfg.get("api_key_env", "") or "_", "").strip() \
        or api_cfg.get("api_key", "")

    if not urls:
        raise SystemExit("[generate] no API endpoint: export TEXT_COT_API_BASES=url1,url2 "
                         "(or set api.base_urls / api.base_url in the config)")
    if not model:
        raise SystemExit("[generate] no model: export TEXT_COT_API_MODEL=<name> "
                         "(or set api.model in the config)")
    if not key:
        raise SystemExit("[generate] no API key: export TEXT_COT_API_KEY=<token> "
                         "(or set api.api_key_env / api.api_key in the config)")

    client = ChatClient(
        urls,
        model=model,
        api_key=key,
        max_concurrency_per_endpoint=int(args.concurrency or api_cfg.get("max_concurrency", 16)),
        rpm=api_cfg.get("rpm"),
        timeout_s=float(api_cfg.get("timeout", 600)),
        max_retries=int(api_cfg.get("max_retries", 5)),
        proxy=api_cfg.get("proxy"),
    )

    rec_by_uid = {r["uid"]: r for r in records}
    lock = asyncio.Lock()
    counter = {"done": 0, "err": 0, "t0": time.monotonic()}
    total = len(tasks)
    seed_base = int(cfg.get("sampling", {}).get("seed_base", 0) or 0)
    send_seed = bool(cfg.get("sampling", {}).get("send_seed", True))
    print(f"[generate] shard {shard_id}/{num_shards}: {total} tasks "
          f"({len(records)} questions x {n_samples} samples, {len(done)} already done)")

    queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    for t in tasks:
        queue.put_nowait(t)
    writer = JsonlWriter(out_path)

    async with client:
        async def one(uid: str, k: int):
            rec = rec_by_uid[uid]
            extra = {"seed": _seed_for(uid, k, seed_base)} if send_seed else None
            comp: Completion = await client.chat(
                _build_messages(cfg, rec["question"]),
                _sampling_for(cfg, k),
                uid=uid,
                sample_idx=k,
                extra_fields=extra,
            )
            obj = comp.to_json()
            obj["problem_type"] = rec["problem_type"]
            async with lock:
                writer.write(obj)
                counter["done"] += 1
                if obj.get("error"):
                    counter["err"] += 1
                if counter["done"] % 50 == 0 or counter["done"] == total:
                    dt = time.monotonic() - counter["t0"]
                    rate = counter["done"] / dt if dt else 0
                    eta = (total - counter["done"]) / rate if rate else 0
                    print(f"[generate] {counter['done']}/{total} "
                          f"err={counter['err']} {rate:.1f} req/s "
                          f"eta={eta / 60:.1f}min tokens={client.stats['completion_tokens']}",
                          flush=True)

        async def worker():
            # Bounded pool: never materialize one coroutine per (question, sample)
            # pair, which would blow up memory on 100k+ question datasets.
            while True:
                try:
                    uid, k = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                await one(uid, k)

        try:
            await asyncio.gather(*(worker() for _ in range(max(1, client.capacity))))
        except asyncio.CancelledError:
            print("[generate] cancelled, shard kept for resume")
            raise
        finally:
            writer.close()

    print(f"[generate] finished shard {shard_id}: {counter['done']} written, "
          f"{counter['err']} errors -> {out_path}")


# ── stage: verify ─────────────────────────────────────────────────────────────
def _norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())[:120]


# Lines that are instructions-to-self rather than reasoning ("We need answer...",
# "ensure the final answer is inside <answer>").  Stripped only when short so we
# never delete substantive derivation.
_META_LINE_RE = re.compile(
    r"^(?:we need|need (?:to )?(?:final|answer|reason)|okay[, ]|alright[, ]|"
    r"ensure (?:the )?(?:final|answer)|put (?:the )?final|include (?:reasoning|final)|"
    r"final (?:answer )?(?:inside|in) |i need to (?:answer|reason))",
    re.IGNORECASE,
)


def _clean_trace(text: str, strip_meta: bool = False) -> str:
    text = (text or "").strip()
    text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<answer>.*?</answer>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Stray unpaired tags (e.g. the model writing "put the answer in <answer>"
    # mid-derivation) would corrupt later <answer> extraction — drop them too.
    text = re.sub(r"</?answer>", "", text, flags=re.IGNORECASE)
    if strip_meta:
        kept = [
            ln for ln in text.splitlines()
            if not (len(ln.strip()) < 200 and _META_LINE_RE.match(ln.strip()))
        ]
        text = "\n".join(kept)
    return text.strip()


def _trace_from_sample(sample: dict[str, Any], trace_source: str) -> str:
    """Pick the text that becomes the SFT  thinking block.

    `content` (default) is the teacher's polished, visible derivation; the
    `reasoning` field is often an informal scratch pad ("We need answer...")
    that is worse as distillation target.  Fall back to reasoning when content
    is answer-only.
    """
    if trace_source == "reasoning":
        return sample.get("reasoning") or ""
    content = sample.get("content") or ""
    m = THINK_RE.search(content)
    if m:
        return m.group(1)
    cut = content.lower().find("<answer>")
    text = content[:cut] if cut >= 0 else content
    return text.strip() or (sample.get("reasoning") or "")


def _select(samples: list[dict[str, Any]], strategy: str) -> dict[str, Any] | None:
    correct = [s for s in samples if s["correct"]]
    if not correct:
        return None
    if strategy == "first_correct":
        return correct[0]
    if strategy == "majority_then_shortest":
        cnt = collections.Counter(_norm_key(s["answer"]) for s in correct)
        top = cnt.most_common(1)[0][0]
        pool = [s for s in correct if _norm_key(s["answer"]) == top]
        return min(pool, key=lambda s: s["reasoning_chars"])
    # default: shortest_correct (DART-Math style: shorter traces distill better)
    return min(correct, key=lambda s: s["reasoning_chars"])


def stage_verify(cfg: dict[str, Any], args) -> None:
    work_dir = resolve_path(cfg["paths"]["work_dir"])
    records = read_jsonl(work_dir / "records.jsonl")
    if args.limit and args.limit > 0:
        records = records[: args.limit]
    rec_by_uid = {r["uid"]: r for r in records}
    vcfg = cfg.get("verify", {})
    strategy = vcfg.get("select", "shortest_correct")
    trace_source = vcfg.get("trace_source", "content")
    strip_meta = bool(vcfg.get("strip_meta", False))
    min_correct = int(vcfg.get("min_correct", 0) or 0)
    max_correct_raw = vcfg.get("max_correct")
    max_correct = int(max_correct_raw) if max_correct_raw is not None else None
    drop_too_easy = bool(vcfg.get("drop_too_easy", False))
    dedup_traces = bool(vcfg.get("dedup_traces", True))

    gens: dict[str, dict[tuple[str, int], dict[str, Any]]] = collections.defaultdict(dict)
    for f in sorted((work_dir / "gen").glob("shard_*.jsonl")):
        for o in read_jsonl(f):
            if o.get("uid") in rec_by_uid:
                # de-duplicate identical (uid, sample_idx) across shard retries
                gens[o["uid"]][(o["uid"], int(o["sample_idx"]))] = o

    accepted, rejected, per_type = [], [], collections.defaultdict(lambda: [0, 0])
    reject_reasons = collections.Counter()
    seen_traces: set[str] = set()
    token_usage = collections.Counter()
    for rec in records:
        uid = rec["uid"]
        samples = list(gens.get(uid, {}).values())
        checked = []
        for s in samples:
            if s.get("error"):
                checked.append({**s, "correct": False, "reason": f"api_error:{s['error'][:120]}",
                                "reasoning_chars": 0})
                continue
            trace = _clean_trace(_trace_from_sample(s, trace_source), strip_meta)
            answer = s.get("answer_text") or extract_answer(s.get("content") or "")
            usage = s.get("usage") or {}
            token_usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            token_usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            ok, why = quality_check(
                trace, answer, s.get("finish_reason", ""),
                min_reasoning_chars=int(vcfg.get("min_reasoning_chars", 80)),
                max_reasoning_chars=int(vcfg.get("max_reasoning_chars", 24000)),
                max_repeat_ratio=float(vcfg.get("max_repeat_ratio", 0.35)),
                reject_truncated=bool(vcfg.get("reject_truncated", True)),
                reject_refusal=bool(vcfg.get("reject_refusal", True)),
                require_reasoning=bool(vcfg.get("require_reasoning", True)),
            )
            if not ok:
                checked.append({**s, "trace": trace, "answer": answer, "correct": False,
                                "reason": why, "reasoning_chars": len(trace)})
                continue
            correct, score, cmp_reason = check_answer(
                answer, rec["ground_truth"], rec["problem_type"],
                freeform_min_f1=float(vcfg.get("freeform_min_f1", 0.75)),
            )
            checked.append({**s, "trace": trace, "answer": answer, "correct": correct, "score": score,
                            "reason": f"ok:{cmp_reason}" if correct else f"wrong:{cmp_reason}",
                            "reasoning_chars": len(trace)})

        n_correct = sum(1 for c in checked if c.get("correct"))
        per_type[rec["problem_type"]][1] += 1

        # ── difficulty filter (DART-Math: easy questions add little signal) ──
        filtered_reason = None
        if n_correct < min_correct:
            filtered_reason = f"difficulty_filter(too_few_correct={n_correct})"
        elif max_correct is not None and n_correct > max_correct:
            filtered_reason = f"difficulty_filter(too_many_correct={n_correct})"
        elif drop_too_easy and len(checked) >= 2 and n_correct == len(checked):
            filtered_reason = "difficulty_filter(too_easy)"

        best = None if filtered_reason else _select(checked, strategy)
        if best is not None and dedup_traces:
            tkey = _norm_key(best["trace"])[:4000]
            if tkey in seen_traces:
                best = None
                filtered_reason = "duplicate_trace"
        if best is None:
            reason = filtered_reason or "no_sample"
            reject_reasons[reason.split("(")[0]] += 1
            for c in checked:
                rejected.append({"uid": uid, "sample_idx": c.get("sample_idx"),
                                 "reason": filtered_reason or c.get("reason", "no_sample"),
                                 "answer": c.get("answer", ""),
                                 "ground_truth": rec["ground_truth"],
                                 "problem_type": rec["problem_type"]})
                if not filtered_reason:
                    reject_reasons[c.get("reason", "no_sample").split("(")[0]] += 1
            continue

        seen_traces.add(_norm_key(best["trace"])[:4000])
        per_type[rec["problem_type"]][0] += 1
        accepted.append({
            "uid": uid,
            "question": rec["question"],
            "ground_truth": rec["ground_truth"],
            "problem_type": rec["problem_type"],
            "data_source": rec["data_source"],
            "reasoning": best["trace"],
            "answer": best["answer"],
            "n_samples": len(checked),
            "n_correct": n_correct,
            "teacher": best.get("model", ""),
            "reasoning_chars": best["reasoning_chars"],
            "finish_reason": best.get("finish_reason", ""),
        })
        # rejected siblings still tracked for stats
        for c in checked:
            if c is best:
                continue
            reject_reasons["unused_correct" if c.get("correct") else "unused_wrong"] += 1

    write_jsonl(work_dir / "accepted.jsonl", accepted)
    write_jsonl(work_dir / "rejected.jsonl", rejected)
    stats = {
        "questions_total": len(records),
        "questions_accepted": len(accepted),
        "acceptance_rate": round(len(accepted) / max(1, len(records)), 4),
        "by_problem_type": {k: {"accepted": v[0], "total": v[1]} for k, v in per_type.items()},
        "reject_reasons": dict(reject_reasons.most_common(20)),
        "avg_reasoning_chars": round(
            sum(a["reasoning_chars"] for a in accepted) / max(1, len(accepted)), 1),
        "avg_samples_per_accepted": round(
            sum(a["n_samples"] for a in accepted) / max(1, len(accepted)), 2),
        "difficulty_n_correct": dict(sorted(
            collections.Counter(a["n_correct"] for a in accepted).items())),
        "trace_source": trace_source,
        "teacher_token_usage": dict(token_usage),
    }
    (work_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"[verify] accepted {len(accepted)}/{len(records)} "
          f"({stats['acceptance_rate']:.1%}) -> {work_dir / 'accepted.jsonl'}")
    print(f"[verify] by type: {stats['by_problem_type']}")
    print(f"[verify] reject reasons: {stats['reject_reasons']}".replace("'", ""))


# ── stage: build ──────────────────────────────────────────────────────────────
def _rows_from_accepted(accepted: list[dict[str, Any]], fmt: dict[str, Any]) -> list[dict[str, Any]]:
    think_open = fmt.get("think_open", "<think>")
    think_close = fmt.get("think_close", "</think>")
    rows = []
    for a in accepted:
        trace = _clean_trace(a["reasoning"], strip_meta=False)
        rows.append({
            "messages": [
                {"role": "user", "content": a["question"]},
                {"role": "assistant",
                 "content": f"{think_open}{trace}{think_close}\n"
                            f"<answer>{a['answer']}</answer>"},
            ],
            "images": [],
            "videos": [],
            "audios": [],
            "_source": a["data_source"],
            "extra_info": {
                "data_type": "text",
                "data_source": a["data_source"],
                "problem_type": a["problem_type"],
                "uid": a["uid"],
                "teacher": a.get("teacher", ""),
                "ground_truth": a["ground_truth"],
                "n_samples": int(a.get("n_samples") or 0),
                "n_correct": int(a.get("n_correct") or 0),
            },
        })
    return rows


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    """Explicit schema: empty list columns otherwise fail inference."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    msg = pa.struct([("role", pa.string()), ("content", pa.string())])
    extra = pa.struct([
        ("data_type", pa.string()), ("data_source", pa.string()),
        ("problem_type", pa.string()), ("uid", pa.string()),
        ("teacher", pa.string()), ("ground_truth", pa.string()),
        ("n_samples", pa.int64()), ("n_correct", pa.int64()),
    ])
    schema = pa.schema([
        ("messages", pa.list_(msg)),
        ("images", pa.list_(pa.string())),
        ("videos", pa.list_(pa.string())),
        ("audios", pa.list_(pa.string())),
        ("_source", pa.string()),
        ("extra_info", extra),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def stage_build(cfg: dict[str, Any], args) -> None:
    work_dir = resolve_path(cfg["paths"]["work_dir"])
    accepted = read_jsonl(work_dir / "accepted.jsonl")
    if not accepted:
        raise SystemExit("[build] no accepted samples; run verify first")

    fmt = cfg.get("output", {})
    rows = _rows_from_accepted(accepted, fmt)

    out_dir = resolve_path(fmt.get("dir", str(work_dir / "out")))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = fmt.get("name", "text_cot")
    jsonl_path = out_dir / f"{name}.jsonl"
    write_jsonl(jsonl_path, rows)

    parquet_path = out_dir / f"{name}.parquet"
    try:
        _write_parquet(rows, parquet_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[build] parquet skipped: {exc}")

    if fmt.get("split_by_source"):
        by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for r in rows:
            by_source[r["_source"]].append(r)
        for src, rs in by_source.items():
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", src)
            write_jsonl(out_dir / f"{safe}.json", rs)
            try:
                _write_parquet(rs, out_dir / f"{safe}.parquet")
            except Exception:  # noqa: BLE001
                pass
        print(f"[build] split_by_source -> {len(by_source)} files in {out_dir}")

    stats_path = work_dir / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    teacher = collections.Counter(
        a.get("teacher", "") for a in accepted if a.get("teacher")
    ).most_common(1)
    teacher_name = teacher[0][0] if teacher else "(unknown)"
    report = [
        "# Text QA CoT 构建报告",
        "",
        f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Teacher: `{teacher_name}`",
        f"- 输出: `{jsonl_path}` / `{parquet_path}`",
        f"- 样本数: {len(rows)}",
        f"- 题目接受率: {stats.get('acceptance_rate')}",
        f"- 平均 reasoning 字符数: {stats.get('avg_reasoning_chars')}",
        "",
        "## 分 problem_type",
        "",
        "| problem_type | accepted | total | rate |",
        "|---|---|---|---|",
    ]
    for k, v in (stats.get("by_problem_type") or {}).items():
        rate = v["accepted"] / max(1, v["total"])
        report.append(f"| {k} | {v['accepted']} | {v['total']} | {rate:.1%} |")
    report += ["", "## 拒绝原因", "", "| reason | count |", "|---|---|"]
    for k, v in (stats.get("reject_reasons") or {}).items():
        report.append(f"| {k} | {v} |")
    (out_dir / "REPORT.md").write_text("\n".join(report) + "\n")

    print(f"[build] wrote {len(rows)} rows -> {jsonl_path}")
    print(f"[build] report -> {out_dir / 'REPORT.md'}")


def stage_stats(cfg: dict[str, Any], args) -> None:
    work_dir = resolve_path(cfg["paths"]["work_dir"])
    stats_path = work_dir / "stats.json"
    if stats_path.exists():
        print(stats_path.read_text())
    else:
        records = read_jsonl(work_dir / "records.jsonl")
        done = _completed_keys(work_dir, None)
        print(f"records={len(records)} generated_samples={len(done)} "
              f"expected={len(records) * int(cfg['sampling'].get('n_samples', 4))}")


def stage_validate(cfg: dict[str, Any], args) -> None:
    """Re-check the built SFT file: schema, tags, answer correctness, dedup."""
    work_dir = resolve_path(cfg["paths"]["work_dir"])
    fmt = cfg.get("output", {})
    out_dir = resolve_path(fmt.get("dir", str(work_dir / "out")))
    path = out_dir / f"{fmt.get('name', 'text_cot')}.jsonl"
    if not path.exists():
        raise SystemExit(f"[validate] missing {path}; run build first")
    rows = read_jsonl(path)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    min_chars = int(cfg.get("verify", {}).get("min_reasoning_chars", 80))
    think_open = fmt.get("think_open", "<think>")
    think_close = fmt.get("think_close", "</think>")

    errors: list[str] = []
    seen_uid: set[str] = set()
    seen_trace: set[str] = set()
    lengths: list[int] = []
    for i, r in enumerate(rows):
        uid = (r.get("extra_info") or {}).get("uid", f"row{i}")
        msgs = r.get("messages") or []
        if len(msgs) != 2 or msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
            errors.append(f"[{uid}] bad messages structure")
            continue
        content = msgs[1].get("content") or ""
        if think_open not in content or think_close not in content:
            errors.append(f"[{uid}] missing think tags")
            continue
        trace = content.split(think_open, 1)[1].split(think_close, 1)[0]
        lengths.append(len(trace))
        if len(trace) < min_chars:
            errors.append(f"[{uid}] trace too short ({len(trace)})")
        ei = r.get("extra_info") or {}
        gt = ei.get("ground_truth")
        if gt is None:
            errors.append(f"[{uid}] missing ground_truth")
        else:
            ok, _, why = check_answer(extract_answer(content), gt, ei.get("problem_type", ""))
            if not ok:
                errors.append(f"[{uid}] answer mismatch ({why})")
        if uid in seen_uid:
            errors.append(f"[{uid}] duplicate uid")
        seen_uid.add(uid)
        tkey = _norm_key(trace)[:4000]
        if tkey in seen_trace:
            errors.append(f"[{uid}] duplicate trace")
        seen_trace.add(tkey)

    avg = sum(lengths) / max(1, len(lengths))
    print(f"[validate] {len(rows)} rows checked, {len(errors)} problems, "
          f"avg trace chars={avg:.0f}")
    for e in errors[:20]:
        print(f"[validate]   {e}")
    if errors:
        raise SystemExit(1)
    print("[validate] OK")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["ingest", "generate", "verify", "build",
                                      "validate", "stats", "all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=0, help="first N questions (0=all)")
    ap.add_argument("--force", action="store_true", help="re-run ingest even if records exist")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=0, help="override per-endpoint concurrency")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.stage in ("ingest", "all"):
        stage_ingest(cfg, args)
    if args.stage in ("generate", "all"):
        asyncio.run(stage_generate(cfg, args))
    if args.stage in ("verify", "all"):
        stage_verify(cfg, args)
    if args.stage in ("build", "all"):
        stage_build(cfg, args)
    if args.stage in ("validate", "all"):
        stage_validate(cfg, args)
    if args.stage == "stats":
        stage_stats(cfg, args)


if __name__ == "__main__":
    main()
