from __future__ import annotations

import re
import unicodedata


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (0 if x == y else 1),
                )
            )
        previous = current
    return previous[-1]


def calculate_cer(reference: str, hypothesis: str) -> float | None:
    ref = list(_normalize(reference))
    hyp = list(_normalize(hypothesis))
    if not ref:
        return None
    return round(_levenshtein(ref, hyp) / len(ref), 6)


def calculate_wer(reference: str, hypothesis: str) -> float | None:
    ref = _normalize(reference).split()
    hyp = _normalize(hypothesis).split()
    if not ref:
        return None
    return round(_levenshtein(ref, hyp) / len(ref), 6)


def evaluate_text(reference: str | None, hypothesis: str) -> dict:
    if not reference or not reference.strip():
        return {
            "ground_truth_available": False,
            "cer": None,
            "wer": None,
        }

    return {
        "ground_truth_available": True,
        "cer": calculate_cer(reference, hypothesis),
        "wer": calculate_wer(reference, hypothesis),
    }
