"""Answer normalization / comparison and CoT quality filters.

Comparison is dispatched by `problem_type`, aligned with the training-time
reward (`src/rl/scripts/mmd_reward.py`) but stricter: here we only need a
binary "is this trace correct" decision, so thresholds lean conservative
(better to drop a good sample than to train on a wrong one).

problem_type values follow the repo convention:
    numerical | multiple choice | free-form | ocr | regression | math | text
"""

from __future__ import annotations

import math
import re
import unicodedata
from fractions import Fraction

# ── extraction ────────────────────────────────────────────────────────────────
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
MC_LETTER_RE = re.compile(r"(?:^|[^A-Za-z])([A-H])(?:[\.\)\:\,]|\s|$)")
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


def extract_answer(text: str) -> str:
    """Final answer from a completion: <answer> > \\boxed{} > last line."""
    if not text:
        return ""
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    m = BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _unwrap_answer(ans: str) -> str:
    """Strip wrappers that commonly surround a gold answer: \\text{A}, $2$, (B)."""
    s = (ans or "").strip()
    for _ in range(3):
        before = s
        s = re.sub(r"\\(?:text|mathrm|mathbf|mathit)\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", s)
        s = s.strip().strip("$").strip()
        if s == before:
            break
    return s


def guess_problem_type(question: str, gt: str) -> str:
    """Infer problem_type from the gold answer.

    Only the answer is used (never the question's option list): datasets such as
    NuminaMath put the *option value* in ground_truth while the question has
    A-D options, and treating those as multiple-choice would compare the letter
    against the value and reject every correct trace.
    """
    g = _unwrap_answer(extract_answer(gt))
    if re.fullmatch(r"[A-Ha-h]", g) or re.fullmatch(r"\(([A-Ha-h])\)", g):
        return "multiple choice"
    if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", g.replace(",", "")):
        return "numerical"
    if re.fullmatch(r"-?\d+\s*/\s*\d+", g):
        return "numerical"
    return "free-form"


# ── normalization ─────────────────────────────────────────────────────────────
_LATEX_NOISE = [
    (r"\\left", ""), (r"\\right", ""), (r"\\!", ""), (r"\\,", ""), (r"\\;", ""),
    (r"\\ ", ""), (r"\\text\{([^{}]*)\}", r"\1"), (r"\\mathrm\{([^{}]*)\}", r"\1"),
    (r"\\tfrac", r"\\frac"), (r"\\dfrac", r"\\frac"),
    (r"\\(?:d?frac|tfrac)\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)"),
]


def _strip_leading_option(s: str) -> str:
    """'A: 42' / '(b) foo' / 'Option C - bar' -> '42' / 'foo' / 'bar'."""
    s = re.sub(r"^\s*\(?[A-Ha-h]\)?\s*[\.\):：\-]\s+", "", s)
    s = re.sub(r"^\s*\(?[A-Ha-h]\)?\s*[\.\):：]\s*", "", s)
    s = re.sub(r"^\s*(?:option|choice)\s+[A-Ha-h]\s*[\.\):：\-]?\s*", "", s, flags=re.IGNORECASE)
    return s.strip()


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = _strip_leading_option(s)
    for pat, rep in _LATEX_NOISE:
        s = re.sub(pat, rep, s)
    s = s.replace("$", "").replace("\\%", "%")
    s = re.sub(r"[ \t]+", " ", s)
    s = s.strip().strip(".").strip()
    return s.lower()


def parse_number(s: str) -> float | None:
    """Best-effort numeric value: handles 1,234 / 3/4 / 45% / 1.2e3 / -5."""
    if s is None:
        return None
    t = normalize_text(s)
    if not t:
        return None
    t = t.replace(",", "").replace(" ", "")
    pct = t.endswith("%")
    if pct:
        t = t[:-1]
    m = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)", t)
    if m:
        try:
            den = float(m.group(2))
            if den == 0:
                return None
            val = float(m.group(1)) / den
            return val / 100 if pct else val
        except ValueError:
            return None
    m = re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t)
    if m:
        try:
            val = float(t)
            return val / 100 if pct else val
        except ValueError:
            return None
    return None


def numeric_equal(a: str, b: str, rel_tol: float = 1e-3, abs_tol: float = 1e-6) -> bool:
    fa, fb = parse_number(a), parse_number(b)
    if fa is None or fb is None:
        return normalize_text(a) == normalize_text(b)
    return math.isclose(fa, fb, rel_tol=rel_tol, abs_tol=abs_tol)


def mc_letters(s: str) -> set[str]:
    t = normalize_text(s)
    # direct single letter / "option b"
    m = re.fullmatch(r"\(?([a-h])\)?[\.\)]?", t)
    if m:
        return {m.group(1).upper()}
    letters = {x.upper() for x in MC_LETTER_RE.findall(s or "")}
    return letters


def mc_equal(pred: str, gt: str) -> bool:
    g = mc_letters(gt)
    if not g:
        return normalize_text(pred) == normalize_text(gt)
    p = mc_letters(pred)
    return bool(p) and p == g


def _token_f1(a: str, b: str) -> float:
    ta = re.findall(r"[a-z0-9]+", normalize_text(a))
    tb = re.findall(r"[a-z0-9]+", normalize_text(b))
    if not ta or not tb:
        return 0.0
    from collections import Counter

    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


def check_answer(pred: str, gt: str, problem_type: str,
                 freeform_min_f1: float = 0.75) -> tuple[bool, float, str]:
    """Return (is_correct, score, reason)."""
    pt = (problem_type or "").strip().lower()
    gp = extract_answer(gt)
    if not pred or not gp:
        return False, 0.0, "empty_answer"

    if pt in ("multiple choice", "multiple_choice", "mc"):
        ok = mc_equal(pred, gp)
        return ok, 1.0 if ok else 0.0, "mc"

    if pt in ("numerical", "regression", "math"):
        ok = numeric_equal(pred, gp)
        if ok:
            return True, 1.0, "numeric"
        # math answers are often non-numeric (expressions); fall back to text
        ok_text = normalize_text(pred) == normalize_text(gp)
        return ok_text, 1.0 if ok_text else 0.0, "numeric_fallback_text"

    if pt == "ocr":
        f1 = _token_f1(pred, gp)
        return f1 >= 0.9, f1, "ocr_f1"

    if pt == "free-form":
        if normalize_text(pred) == normalize_text(gp):
            return True, 1.0, "exact"
        f1 = _token_f1(pred, gp)
        return f1 >= freeform_min_f1, f1, "freeform_f1"

    # unknown / text / exact
    ok = normalize_text(pred) == normalize_text(gp)
    if ok:
        return True, 1.0, "exact"
    f1 = _token_f1(pred, gp)
    return f1 >= 0.9, f1, "text_f1"


# ── CoT quality filters ───────────────────────────────────────────────────────
REFUSAL_PATTERNS = [
    r"\bi('m| am) (sorry|unable)\b",
    r"\bi cannot (answer|help|assist|determine)\b",
    r"\bas an ai\b",
    r"无法(回答|提供|确定)",
    r"抱歉[，,].{0,10}(无法|不能)",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def repetition_ratio(text: str, n: int = 8) -> float:
    """Fraction of n-grams that are repeats of an already-seen n-gram."""
    words = re.findall(r"\S+", text or "")
    if len(words) < 2 * n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    seen: set[tuple[str, ...]] = set()
    dup = 0
    for g in grams:
        if g in seen:
            dup += 1
        else:
            seen.add(g)
    return dup / len(grams)


def quality_check(
    reasoning: str,
    answer_text: str,
    finish_reason: str,
    *,
    min_reasoning_chars: int = 80,
    max_reasoning_chars: int = 24000,
    max_repeat_ratio: float = 0.35,
    reject_truncated: bool = True,
    reject_refusal: bool = True,
    require_reasoning: bool = True,
) -> tuple[bool, str]:
    if reject_truncated and finish_reason == "length":
        return False, "truncated"
    if not answer_text.strip():
        return False, "empty_answer"
    if require_reasoning and len(reasoning.strip()) < min_reasoning_chars:
        return False, f"reasoning_too_short({len(reasoning.strip())})"
    if len(reasoning) > max_reasoning_chars:
        return False, f"reasoning_too_long({len(reasoning)})"
    if reject_refusal and _REFUSAL_RE.search(reasoning[:2000] + "\n" + answer_text[:500]):
        return False, "refusal"
    rr = repetition_ratio(reasoning)
    if rr > max_repeat_ratio:
        return False, f"degenerate_repetition({rr:.2f})"
    # stray control chars / replacement chars
    if "\ufffd" in reasoning or "\x00" in reasoning:
        return False, "bad_chars"
    return True, "ok"
