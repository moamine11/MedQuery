"""Compare local output with independently transcribed, page-bound visual references."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from statistics import median
import unicodedata

from rapidfuzz.distance import Levenshtein

from benchmark import file_hash
from engines import area, inside, overlap, rect_union
from prepare import CONFIG, REPO, local_output, write_json

DIRECTIONAL = "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u00ad"


def normalize(text, *, arabic_marks=False):
    """Preserve quantities, visible symbols and punctuation; collapse whitespace."""
    chars = []
    for char in text:
        code = ord(char)
        if char in DIRECTIONAL:
            continue
        if arabic_marks and (0x0610 <= code <= 0x061A or 0x064B <= code <= 0x065F or
                             code == 0x0670 or 0x06D6 <= code <= 0x06ED):
            continue
        if 0xFB50 <= code <= 0xFDFF or 0xFE70 <= code <= 0xFEFC or 0xFB00 <= code <= 0xFB06:
            char = unicodedata.normalize("NFKC", char)
        chars.append(char)
    return " ".join(unicodedata.normalize("NFC", "".join(chars)).split())


def critical_present(text, token):
    # Boundary checks prevent matching a quantity inside a larger number.
    token = normalize(token)
    return re.search(r"(?<![\w])" + re.escape(token) + r"(?![\w])", text) is not None


def iou(a, b):
    shared = overlap(a, b)
    union = area(a) + area(b) - shared
    return shared / union if union else 0


def local_words(lines, bbox):
    return [[word for word in row if inside(word["bbox"], bbox)] for row in lines]


def excerpt_scores(reference, lines, minimum_iou):
    rows = [[word for word in row if word["text"]] for row in local_words(lines, reference["bbox"])]
    actual = normalize(" ".join(word["text"] for row in rows for word in row))
    expected = normalize(reference["text"])
    words = [word for row in rows for word in row]
    expected_tokens, actual_tokens = expected.split(), actual.split()
    ops = Levenshtein.editops(expected_tokens, actual_tokens)
    edits = Counter(edit.tag for edit in ops)
    bbox = rect_union([word["bbox"] for word in words]) if words else None
    localization = iou(reference["bbox"], bbox) if bbox else 0
    expected_unmarked, actual_unmarked = normalize(expected, arabic_marks=True), normalize(actual, arabic_marks=True)
    return {
        "id": reference["id"], "kind": reference["kind"],
        "ref_bbox": reference["bbox"], "observed_bbox": bbox, "bbox_iou": localization,
        "bbox_hit": localization >= minimum_iou, "text_and_bbox_hit": localization >= minimum_iou and actual == expected,
        "ref_chars": len(expected), "char_errors": Levenshtein.distance(expected, actual),
        "unmarked_ref_chars": len(expected_unmarked),
        "unmarked_char_errors": Levenshtein.distance(expected_unmarked, actual_unmarked),
        "ref_words": len(expected_tokens), "word_errors": len(ops),
        "correct_words": len(expected_tokens) - edits["delete"] - edits["replace"],
        "missing_words": edits["delete"], "inserted_words": edits["insert"], "exact_text": actual == expected,
        "critical": [dict(token, correct=critical_present(actual, token["text"])) for token in reference["critical"]],
        "reference": expected, "observed": actual,
    }


def order_scores(order, excerpts, output, *, geometric=False):
    source = normalize(output["source_text"])
    expected = {item["id"]: normalize(item["text"]) for item in excerpts}
    indexes = {key: source.find(expected[key]) for key in order}
    if geometric:
        words = [word for line in output["lines"] for word in line if word["text"]]
        boxes = {item["id"]: item["bbox"] for item in excerpts}
        indexes = {key: next((i for i, word in enumerate(words) if inside(word["bbox"], boxes[key])), -1)
                   for key in order}
    pairs = []
    for i, first in enumerate(order):
        for second in order[i + 1:]:
            a, b = indexes[first], indexes[second]
            state = "missing_anchor" if min(a, b) < 0 else "correct" if a < b else "inverted"
            pairs.append({"first": first, "second": second, "status": state})
    return pairs


def located_cell(cells, reference):
    bbox = reference["bbox"]
    matches = [cell for cell in cells if cell.get("bbox") and inside(bbox, cell["bbox"])
               and overlap(bbox, cell["bbox"]) / area(bbox) >= 0.5]
    return min(matches, key=lambda cell: area(cell["bbox"]), default=None)


def cell_ok(cell, reference):
    if cell is None:
        return False
    return critical_present(normalize(cell["text"]), reference["text"])


def table_scores(table, excerpts, output, minimum_iou):
    prediction = max(output["tables"], key=lambda item: iou(table["bbox"], item["bbox"]), default=None)
    localization = iou(table["bbox"], prediction["bbox"]) if prediction else 0
    cells = prediction["cells"] if localization >= minimum_iou else []
    references = {item["id"]: item for item in excerpts}
    row_scores, merge_scores, footnote_scores = [], [], []
    used_ids = set()
    for pair in table["pairs"]:
        a_id, b_id = pair["first"], pair["second"]
        a, b = references[a_id], references[b_id]
        ca, cb = located_cell(cells, a), located_cell(cells, b)
        present = cell_ok(ca, a) and cell_ok(cb, b)
        correct = False
        if present and ca is not cb:
            if pair["kind"] == "same_row":
                correct = ca["row"] == cb["row"] and ca["col"] != cb["col"]
            else:
                correct = (ca["col"] <= cb["col"] < ca["col"] + ca.get("col_span", 1) and ca["row"] < cb["row"]
                           and (ca.get("column_header") or ca["row"] == 0 or (
                               ca["col"] < len(prediction.get("header_names") or [])
                               and normalize(str(prediction["header_names"][ca["col"]])) == normalize(a["text"])
                           )))
        row_scores.append({**pair, "both_cell_texts_preserved": bool(present), "correct": bool(correct)})
        used_ids.update((a_id, b_id))
    for merge in table["merged_cells"]:
        anchor = references[merge["excerpt"]]
        cell = located_cell(cells, anchor)
        span = "col_span" if merge["axis"] == "column" else "row_span"
        span_ok = cell is not None and cell.get(span) in (None, merge["span"])
        correct = cell_ok(cell, anchor) and span_ok and (
            cell.get(span) == merge["span"] or iou(cell["bbox"], merge["bbox"]) >= minimum_iou
        )
        merge_scores.append({**merge, "correct": bool(correct), "explicit_span": bool(cell and cell.get(span))})
        used_ids.add(merge["excerpt"])
    for pair in table["footnotes"]:
        marker, note = references[pair["source"]], references[pair["note"]]
        marker_present = critical_present(normalize(output["source_text"]), marker["text"])
        note_present = critical_present(normalize(output["source_text"]), note["text"])
        links = prediction.get("footnote_links", []) if prediction else []
        relation = any(link == [pair["source"], pair["note"]] for link in links)
        attached = bool(prediction and localization >= minimum_iou and any(
            note["bbox"] and overlap(note["bbox"], n["bbox"]) / area(note["bbox"]) >= 0.4
            and critical_present(normalize(n["text"]), note["text"])
            for n in prediction.get("linked_footnotes", []) if n.get("bbox")
        ))
        footnote_scores.append({
            **pair, "both_texts_preserved": marker_present and note_present,
            "table_attachment_correct": attached, "correct": relation,
        })
    cell_scores = [{"id": key, "correct": cell_ok(located_cell(cells, references[key]), references[key])}
                   for key in sorted(used_ids)]
    return {
        "id": table["id"], "table_bbox_hit": localization >= minimum_iou, "bbox_iou": localization,
        "pairs": row_scores, "cells": cell_scores, "merged_cells": merge_scores, "footnotes": footnote_scores,
        "continuation": ({"reference": table["continuation"], "linked": bool(prediction and prediction.get("continuation_links"))}
                         if table.get("continuation") else None),
    }


def heading_hit(reference, output, minimum_iou):
    bbox = reference["bbox"]
    inside_heads = [h for h in output["headings"] if h.get("bbox") and inside(h["bbox"], bbox)]
    candidate = normalize(" ".join(h["text"] for h in inside_heads))
    return bool(inside_heads and iou(bbox, rect_union([h["bbox"] for h in inside_heads])) >= minimum_iou
                and candidate == normalize(reference["text"]))


def validate_reference(reference, manifest, manifest_path):
    if reference["manifest_sha256"] != file_hash(manifest_path):
        raise ValueError("The visual reference set belongs to a different frozen page manifest")
    old = {page["key"]: page for page in manifest["pages"]}
    if set(old) != set(reference["layouts"]):
        raise ValueError("Layout strata must cover the frozen manifest")
    seen = set()
    for ref in reference["pages"]:
        source = old[ref["key"]]
        if ref["key"] in seen or ref["verification"] != "visual_transcription" or (
            ref["sha256"], ref["physical_page"], ref["render_sha256"]
        ) != (source["sha256"], source["physical_page"], source["review_render_sha256"]):
            raise ValueError(f"Duplicate, unverified, or wrong source annotation: {ref['key']}")
        if file_hash(REPO / source["review_render"]) != ref["render_sha256"]:
            raise ValueError(f"Render bytes changed: {ref['key']}")
        seen.add(ref["key"])
        identifiers = set()
        for excerpt in ref["excerpts"]:
            box = excerpt["bbox"]
            if excerpt["id"] in identifiers or not all(0 <= dim <= 1 for dim in box) or not area(box):
                raise ValueError(f"Invalid excerpt: {ref['key']} / {excerpt['id']}")
            for token in excerpt["critical"]:
                if not critical_present(normalize(excerpt["text"]), token["text"]):
                    raise ValueError(f"Critical token is outside its annotated text: {ref['key']} / {excerpt['id']}")
            identifiers.add(excerpt["id"])


def make_row(measurement, ref, layout, threshold, region_only=False):
    page = measurement["page"]
    raw = measurement.get("output", {})
    output = {"source_text": "", "lines": [], "tables": [], "headings": [], "flowchart_edges": [], **raw}
    excerpt_refs = [] if ref is None else ref["excerpts"]
    if region_only:
        excerpt_refs = [item for item in excerpt_refs
                        if any(inside(item["bbox"], crop) for crop in ref["oracle_regions"])]
    excerpts = [dict(excerpt_scores(item, output["lines"], threshold),
                     language=item.get("language", page["language"])) for item in excerpt_refs]
    for i, item in enumerate(excerpt_refs):
        if item["kind"] == "heading":
            excerpts[i]["heading_hit"] = heading_hit(item, output, threshold)
    tables, flow_edges, order, geometry_order = [], [], [], []
    control = None
    if ref and not region_only:
        order = order_scores(ref.get("order", []), excerpt_refs, output)
        geometry_order = order_scores(ref.get("order", []), excerpt_refs, output, geometric=True)
        tables = [table_scores(table, excerpt_refs, output, threshold) for table in ref.get("tables", [])]
        flow_edges = [{"edge": edge, "correct": edge in output["flowchart_edges"]} for edge in ref.get("flowchart_edges", [])]
        if ref.get("control"):
            control = {
                "kind": ref["control"], "alphabetic_words": len(re.findall(r"[^\W\d_]+", normalize(output["source_text"]))),
                "should_have_no_alphabetic_words": ref["control"] != "branding_only",
            }
    diagnostics = raw.get("diagnostics", {})
    return {
        "mode": measurement["mode"], "key": page["key"], "asset_sha256": page["sha256"],
        "physical_page": page["physical_page"], "split": page["split"],
        "language": page["language"], "layout": layout, "status": measurement["status"],
        "elapsed_seconds": measurement["elapsed_seconds"], "peak_rss_kib": measurement.get("peak_rss_kib"),
        "visual_reference": bool(ref), "excerpts": excerpts, "order": order, "geometry_order": geometry_order,
        "tables": tables, "flow_edges": flow_edges,
        "target": ref.get("target") if ref else None, "control": control,
        "page_has_exact_located_anchor": any(item["text_and_bbox_hit"] for item in excerpts) if excerpts else None,
        "diagnostics": diagnostics,
    }


def ratio(a, b):
    return round(a / b, 6) if b else None


def text_metrics(excerpts):
    totals = Counter()
    by_critical = defaultdict(Counter)
    for excerpt in excerpts:
        for key in ("ref_chars", "char_errors", "ref_words", "word_errors", "correct_words", "missing_words",
                    "inserted_words", "unmarked_ref_chars", "unmarked_char_errors"):
            totals[key] += excerpt[key]
        for token in excerpt["critical"]:
            category = by_critical[token["kind"]]
            category["n"] += 1
            category["correct"] += token["correct"]
    headings = [item for item in excerpts if "heading_hit" in item]
    flow_labels = [item for item in excerpts if item["kind"] in {"flow_label", "flow_condition"}]
    return {
        "excerpts": len(excerpts), **totals,
        "CER": ratio(totals["char_errors"], totals["ref_chars"]),
        "WER": ratio(totals["word_errors"], totals["ref_words"]),
        "CER_without_arabic_marks": ratio(totals["unmarked_char_errors"], totals["unmarked_ref_chars"]),
        "aligned_word_recall": ratio(totals["correct_words"], totals["ref_words"]),
        "exact_excerpts": sum(item["exact_text"] for item in excerpts),
        "bbox_hits": sum(item["bbox_hit"] for item in excerpts),
        "exact_text_and_bbox_hits": sum(item["text_and_bbox_hit"] for item in excerpts),
        "median_bbox_iou": round(median(item["bbox_iou"] for item in excerpts), 6) if excerpts else None,
        "heading_hits": sum(item["heading_hit"] for item in headings), "heading_total": len(headings),
        "flow_label_hits": sum(item["exact_text"] for item in flow_labels), "flow_label_total": len(flow_labels),
        "critical_hits": sum(c["correct"] for c in by_critical.values()),
        "critical_total": sum(c["n"] for c in by_critical.values()), "critical_by_kind": by_critical,
    }


def performance(rows):
    times = sorted(row["elapsed_seconds"] for row in rows if row["status"] == "ok")
    all_peak = [row["peak_rss_kib"] for row in rows if row["peak_rss_kib"] is not None]
    return {
        "attempted": len(rows), "statuses": Counter(row["status"] for row in rows),
        "failure_rate": ratio(sum(row["status"] != "ok" for row in rows), len(rows)),
        "total_elapsed_seconds": round(sum(row["elapsed_seconds"] for row in rows), 5),
        "median_seconds": round(median(times), 5) if times else None,
        "p95_seconds": round(times[math.ceil(len(times) * .95) - 1], 5) if times else None,
        "max_rss_mib": round(max(all_peak) / 1024, 3) if all_peak else None,
    }


def stats(rows):
    excerpts = [item for row in rows for item in row["excerpts"]]
    located = [row["page_has_exact_located_anchor"] for row in rows if row["page_has_exact_located_anchor"] is not None]
    tables = [table for row in rows for table in row["tables"]]
    structures = {}
    for name in ("pairs", "cells", "merged_cells", "footnotes"):
        items = [item for table in tables for item in table[name]]
        structures[name] = {"total": len(items), "correct": sum(item["correct"] for item in items)}
    continuity = [tab["continuation"] for tab in tables if tab["continuation"]]
    edges = [item for row in rows for item in row["flow_edges"]]
    controls = [row for row in rows if row["control"]]
    return {
        "performance": performance(rows), "text": text_metrics(excerpts),
        "exact_anchor_pages": {"correct": sum(located), "total": len(located)},
        "reading_order_pairs": Counter(item["status"] for row in rows for item in row["order"]),
        "geometry_order_pairs": Counter(item["status"] for row in rows for item in row["geometry_order"]),
        "table_localization": {"correct": sum(row["table_bbox_hit"] for row in tables), "total": len(tables)},
        "table_structure": structures,
        "table_footnote_attachments": sum(item["table_attachment_correct"] for table in tables for item in table["footnotes"]),
        "continuations": {"linked": sum(row["linked"] for row in continuity), "total": len(continuity)},
        "flow_edges": {"correct": sum(edge["correct"] for edge in edges), "total": len(edges)},
        "route_actions": Counter(row["diagnostics"].get("action", "not_applicable") for row in rows),
        "route_reasons": Counter(row["diagnostics"].get("reason", "not_applicable") for row in rows),
        "control_ocr": sum(row["diagnostics"].get("action") in {"page_ocr", "region_ocr"} or
                           row["mode"] == "tesseract_full" for row in controls),
        "control_false_text": sum(row["control"]["should_have_no_alphabetic_words"] and
                                  row["control"]["alphabetic_words"] > 0 for row in controls),
        "controls": len(controls),
    }



def review_items(rows, refs):
    """Collect localized review cases. This is an experiment artifact."""
    items = []
    for row in rows:
        ref = refs.get(row["key"])
        if row["mode"] not in {"selective_refined", "pymupdf_tables", "docling"}:
            continue
        location = {key: row[key] for key in ("key", "physical_page", "asset_sha256", "mode")}

        def add(reason, bbox, **details):
            items.append({**location, "reason": reason, "status": "needs_review", "bbox": bbox, **details})

        if row["status"] != "ok":
            add("engine_failure_or_partial_output", [0, 0, 1, 1], outcome=row["status"])
        for excerpt in row["excerpts"]:
            missing = [token["kind"] for token in excerpt["critical"] if not token["correct"]]
            if missing:
                add("critical_token_disagreement", excerpt["ref_bbox"], excerpt_id=excerpt["id"], kinds=missing)
        if not ref:
            continue
        if row["mode"] in {"docling", "pymupdf_tables"}:
            for table, expected in zip(row["tables"], ref.get("tables", []), strict=True):
                unresolved = (not table["table_bbox_hit"] or
                              any(not relation["correct"] for kind in ("pairs", "merged_cells", "footnotes")
                                  for relation in table[kind]) or
                              table["continuation"] and not table["continuation"]["linked"])
                if unresolved:
                    add("unresolved_table_relation", expected["bbox"], table_id=expected["id"])
        if row["mode"] == "selective_refined":
            if any(edge["correct"] is False for edge in row["flow_edges"]):
                nodes = {node for edge in ref["flowchart_edges"] for node in edge[:2]}
                bbox = rect_union([excerpt["bbox"] for excerpt in ref["excerpts"] if excerpt["id"] in nodes])
                add("unresolved_figure_edge", bbox, scope="annotated_nodes")
            if row["layout"] in {"rtl_three_column", "rtl_single_column", "rtl_diagram", "journal_table"} and (
                any(pair["status"] != "correct" for pair in row["order"])
            ):
                add("unverified_reading_order", [0, 0, 1, 1])
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="phase3-final")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    home = local_output(REPO / config["working_dir"])
    run = local_output(home / "runs" / args.run)
    manifest_path, reference_path = home / "manifest.json", home / "references/verified.json"
    manifest, reference = (json.loads(path.read_text()) for path in (manifest_path, reference_path))
    validate_reference(reference, manifest, manifest_path)
    refs = {row["key"]: row for row in reference["pages"]}
    pages = {row["key"]: row for row in manifest["pages"]}
    groups, region_groups, environments = {}, {}, {}
    all_rows = []
    for directory in sorted(run.iterdir()):
        if not directory.is_dir():
            continue
        measurements = [json.loads(path.read_text()) for path in sorted(directory.glob("*__*.json"))]
        if not measurements:
            continue
        contexts = json.loads((directory / "run.json").read_text())
        if contexts["input_sha256"]["manifest.json"] != file_hash(manifest_path):
            raise ValueError(f"Run {directory.name} does not match the scored manifest")
        environments[directory.name] = contexts["environment"]
        rows, region_rows = [], []
        for result in measurements:
            page = result["page"]
            key = page["key"]
            if any(page[field] != pages[key][field] for field in ("sha256", "physical_page", "split", "language")):
                raise ValueError(f"Mismatched engine page provenance: {key}")
            ref, layout = refs.get(key), reference["layouts"][key]
            if directory.name != "reference_regions":
                rows.append(make_row(result, ref, layout, config["reference_iou"]))
            if ref and ref.get("oracle_regions"):
                region_rows.append(make_row(result, ref, layout, config["reference_iou"], region_only=True))
        if rows:
            groups[directory.name] = {
                "all": stats(rows),
                "by_split": {split: stats([row for row in rows if row["split"] == split])
                             for split in ("development", "held_out")},
                "by_language": {language: {
                    "performance": performance([row for row in rows if row["language"] == language]),
                    "text": text_metrics([item for row in rows for item in row["excerpts"]
                                          if item["language"] == language]),
                } for language in ("ar", "en", "fr")},
                "by_layout": {label: stats([row for row in rows if row["layout"] == label])
                              for label in sorted(set(reference["layouts"].values()))},
            }
            all_rows.extend(rows)
        if region_rows:
            region_groups[directory.name] = {"pooled": stats(region_rows), "by_page": {
                r["key"]: stats([r]) for r in region_rows
            }}
            if directory.name == "reference_regions":
                all_rows.extend(region_rows)
    arabic_keys = {page["key"] for page in pages.values() if page["language"] == "ar"}
    rendered_labels = [{"key": key, "visual": ref["printed_label"], "pdf_page_label": pages[key]["pdf_page_label"]}
                       for key, ref in refs.items() if "printed_label" in ref]
    output = {
        "reference_sha256": file_hash(reference_path), "scorer_sha256": file_hash(Path(__file__)),
        "manifest_sha256": file_hash(manifest_path), "reference_pages": len(refs),
        "modes": groups, "region_comparison": region_groups, "environments": environments,
        "arabic_page_comparison": {
            mode: stats([row for row in all_rows if row["key"] in arabic_keys and row["mode"] == mode])
            for mode in groups
        },
        "printed_label_checks": rendered_labels,
    }
    write_json(run / "summary.json", output)
    write_json(run / "error_analysis.json", all_rows)
    write_json(run / "review_queue.json", review_items(all_rows, refs))
    print(f"Scored {len(all_rows)} page-mode measurements with {len(refs)} visual references.")


if __name__ == "__main__":
    main()
