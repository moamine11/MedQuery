"""Scoped extraction diagnostics and review evidence."""

from collections import Counter
from difflib import SequenceMatcher
import re

from experiments.extraction.engines import area, overlap

from .layout import comparable_geometry
from .routing import ODD_ARABIC, broken_chars, script_counts

CRITICAL = re.compile(
    r"[+−–<>≤≥±-]?\d+(?:[.,٫٬:/–−-]\d+)*(?:[²³⁰¹⁴⁵⁶⁷⁸⁹%٪])?"
    r"|(?<!\w)(?:μg|µg|mcg|mg|kg|mL|mmHg|mmol|m²|m³|IU|UI|"
    r"hours?|days?|weeks?|jours?|semaines?|heures?|fois|not|never|without|"
    r"sans|jamais|pas|contre-indiqu\w*|لا|لم|لن|ليس|عدم|دون|غير|يجب|ضرورة)(?!\w)",
    re.IGNORECASE,
)
DEPENDENT = re.compile(r"(?:\b(?:see|shown in|table|figure|fig\.|algorithm|voir|tableau|ci-dessus|ci-dessous)\b|الجدول|الشكل|الخوارزمية)", re.I)


def critical_tokens(text):
    return Counter(match.group() for match in CRITICAL.finditer(text))


def comparison_text(text):
    # Only collapse whitespace for sensitive comparisons. Keep superscripts,
    # digits, punctuation, bidi runs, negation, and spelling as observed.
    return " ".join(text.split())


def relative_disagreement(a, b):
    return 1 - SequenceMatcher(None, comparison_text(a), comparison_text(b), autojunk=False).ratio()


def scoped_text(lines, box):
    return " ".join(line["text"] for line in lines if area(line["bbox"]) and
                    overlap(line["bbox"], box) / area(line["bbox"]) >= .48)


def arabic_quality(text):
    ar, latin = script_counts(text)
    return (broken_chars(text) + 2 * len(ODD_ARABIC.findall(text)) +
            max(0, latin - ar / 3)) / max(ar + latin, 1)


def scope_issue(issues, code, bbox=None, *, severity="review", **details):
    issues.append({"code": code, "bbox": bbox or [0, 0, 1, 1], "severity": severity, **details})


def assess_block(block, *, alternatives, arabic, ambiguous, page_ocr, confidence, structures, profile):
    issues = []
    bbox, text = block["bbox"], block["text"]
    linked = [item for item in structures if comparable_geometry(bbox, item["bbox"]) >= .38]
    note_owners = [table for table in structures if table["kind"] == "table" and any(
        comparable_geometry(bbox, note["bbox"]) >= .5 for note in table["footnote_candidates"]
    )]
    if note_owners:
        block["kind"] = "table_footnote"
        block["structure_keys"] = [table["key"] for table in note_owners]
        scope_issue(issues, "unresolved_structure", bbox)
    elif linked:
        categories = {item["kind"] for item in linked}
        block["kind"] = "structural_text" if len(categories) > 1 else next(iter(categories)) + "_text"
        block["structure_keys"] = [item["key"] for item in linked]
        scope_issue(issues, "unresolved_structure", bbox)
    elif DEPENDENT.search(text) and structures:
        scope_issue(issues, "potential_structure_dependency", bbox)
    if any(line["vertical"] for line in block["lines"]):
        scope_issue(issues, "vertical_text_order_unverified", bbox)
    if ambiguous:
        scope_issue(issues, "unverified_reading_order", bbox)
    if broken_chars(text) or (arabic and ODD_ARABIC.search(text)):
        scope_issue(issues, "suspicious_glyphs", bbox)
    tokens = critical_tokens(text)
    for name, lines in alternatives.items():
        observed = scoped_text(lines, bbox)
        if not observed:
            continue
        other = critical_tokens(observed)
        if (tokens or other) and tokens != other:
            scope_issue(issues, "critical_token_disagreement", bbox, competing_candidate=name,
                        selected_tokens=list(tokens.elements()), candidate_tokens=list(other.elements()))
        elif arabic and relative_disagreement(text, observed) > profile["arabic_line_disagreement"]:
            scope_issue(issues, "arabic_candidate_disagreement", bbox, competing_candidate=name)
    if page_ocr:
        if confidence is None or confidence < profile["minimum_ocr_confidence"]:
            scope_issue(issues, "low_ocr_confidence", bbox)
        if tokens:
            scope_issue(issues, "ocr_critical_text_requires_source_check", bbox)
    block["diagnostics"] = issues
    block["status"] = "needs_review" if issues or block["kind"] in {"caption", "footnote"} else "extracted"
