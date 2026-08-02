"""Field-level scoring logic for bill extraction evaluation.

Scoring rules (documented transparently in README):
- vendor_name:    fuzzy match (rapidfuzz token_sort_ratio) >= 85, case-insensitive
- invoice_number: exact match after stripping whitespace/punctuation; null==null is correct
- date:           exact ISO match; skipped if ground truth marks date_ambiguous: true
- amount:         exact match (no tolerance — see README justification)
- currency:       exact match
- gst_details:    presence/absence check + GSTIN substring match if present
"""

import re
import logging

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

VENDOR_FUZZY_THRESHOLD = 85  # token_sort_ratio score to count as correct


def normalize_whitespace(s: str | None) -> str:
    """Collapse whitespace and strip."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def strip_punctuation(s: str | None) -> str:
    """Remove all non-alphanumeric characters for comparison."""
    if s is None:
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", s)


def score_vendor_name(predicted: str | None, expected: str | None) -> dict:
    """Fuzzy match on vendor name using token_sort_ratio."""
    pred = normalize_whitespace(predicted).lower()
    exp = normalize_whitespace(expected).lower()

    if not exp:
        # No ground truth vendor — skip
        return {"correct": None, "score": None, "reason": "no ground truth"}

    ratio = fuzz.token_sort_ratio(pred, exp)
    correct = ratio >= VENDOR_FUZZY_THRESHOLD

    return {
        "correct": correct,
        "score": ratio,
        "reason": f"fuzzy={ratio:.1f} (threshold={VENDOR_FUZZY_THRESHOLD})",
    }


def score_invoice_number(predicted: str | None, expected: str | None) -> dict:
    """Exact match after stripping whitespace and punctuation."""
    pred_clean = strip_punctuation(predicted)
    exp_clean = strip_punctuation(expected)

    # Both null/empty = correct (field genuinely not present)
    if not pred_clean and not exp_clean:
        return {"correct": True, "score": 100, "reason": "both null/empty"}

    correct = pred_clean.lower() == exp_clean.lower()
    return {
        "correct": correct,
        "score": 100 if correct else 0,
        "reason": f"exact match: '{predicted}' vs '{expected}'",
    }


def score_date(
    predicted: str | None, expected: str | None, date_ambiguous: bool = False
) -> dict:
    """Exact ISO date match. Skipped if ground truth is ambiguous."""
    if date_ambiguous:
        return {"correct": None, "score": None, "reason": "date_ambiguous=true, excluded"}

    pred = (predicted or "").strip()
    exp = (expected or "").strip()

    # Both null/empty
    if not pred and not exp:
        return {"correct": True, "score": 100, "reason": "both null/empty"}

    correct = pred == exp
    return {
        "correct": correct,
        "score": 100 if correct else 0,
        "reason": f"ISO match: '{predicted}' vs '{expected}'",
    }


def score_amount(predicted: float | None, expected: float | None) -> dict:
    """Exact match on amount.

    We chose exact match (no ±1 tolerance) because:
    - Even a ₹1 error on a small bill is significant for accounting.
    - Allowing tolerance would inflate accuracy on a small sample.
    - The README documents this choice explicitly.
    """
    # Both null
    if predicted is None and expected is None:
        return {"correct": True, "score": 100, "reason": "both null"}

    # One null, other not
    if predicted is None or expected is None:
        return {
            "correct": False,
            "score": 0,
            "reason": f"one null: predicted={predicted}, expected={expected}",
        }

    correct = abs(predicted - expected) < 0.01  # float comparison epsilon
    return {
        "correct": correct,
        "score": 100 if correct else 0,
        "reason": f"exact: {predicted} vs {expected}",
    }


def score_currency(predicted: str | None, expected: str | None) -> dict:
    """Exact match on currency code."""
    pred = (predicted or "INR").strip().upper()
    exp = (expected or "INR").strip().upper()
    correct = pred == exp
    return {
        "correct": correct,
        "score": 100 if correct else 0,
        "reason": f"'{pred}' vs '{exp}'",
    }


def score_gst_details(predicted: str | None, expected: str | None) -> dict:
    """Two-part scoring: presence/absence + GSTIN substring check.

    Returns a dict with:
    - presence_correct: did the model correctly identify whether GST info exists?
    - gstin_correct: if GST info is present in ground truth, did model get the GSTIN?
    """
    pred_present = bool(predicted and predicted.strip())
    exp_present = bool(expected and expected.strip())

    presence_correct = pred_present == exp_present

    # Check GSTIN substring (15-char alphanumeric Indian GSTIN)
    gstin_pattern = re.compile(r"\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z\d][A-Z]")

    exp_gstin_match = gstin_pattern.search(expected or "")
    pred_gstin_match = gstin_pattern.search(predicted or "")

    gstin_correct = None  # not applicable if no GSTIN in ground truth
    if exp_gstin_match:
        if pred_gstin_match:
            gstin_correct = exp_gstin_match.group() == pred_gstin_match.group()
        else:
            gstin_correct = False

    return {
        "presence_correct": presence_correct,
        "gstin_correct": gstin_correct,
        "reason": f"presence: pred={pred_present} exp={exp_present}"
        + (f", GSTIN: {gstin_correct}" if gstin_correct is not None else ""),
    }


def score_extraction(extraction_dict: dict, ground_truth: dict) -> dict:
    """Score all fields of an extraction against ground truth.

    Args:
        extraction_dict: dict from BillExtraction.model_dump()
        ground_truth: dict from ground_truth.json entry

    Returns:
        dict with per-field scoring results
    """
    return {
        "vendor_name": score_vendor_name(
            extraction_dict.get("vendor_name"),
            ground_truth.get("vendor_name"),
        ),
        "invoice_number": score_invoice_number(
            extraction_dict.get("invoice_number"),
            ground_truth.get("invoice_number"),
        ),
        "date": score_date(
            extraction_dict.get("date"),
            ground_truth.get("date"),
            ground_truth.get("date_ambiguous", False),
        ),
        "amount": score_amount(
            extraction_dict.get("amount"),
            ground_truth.get("amount"),
        ),
        "currency": score_currency(
            extraction_dict.get("currency"),
            ground_truth.get("currency"),
        ),
        "gst_details": score_gst_details(
            extraction_dict.get("gst_details"),
            ground_truth.get("gst_details"),
        ),
    }
