"""Evaluate the production artifacts with frozen visual references and mapping invariants."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import pymupdf

from prepare import REPO, local_output, write_json
from score import excerpt_scores, iou, order_scores, text_metrics
from engines import inside

from ingestion.extraction.config import file_hash
from ingestion.extraction.engines import benchmark
from ingestion.extraction.quality import critical_tokens
from ingestion.extraction.layout import valid_box

HOME = REPO / "data/ingestion/phase4"
PHASE3 = REPO / "data/ingestion/phase3"


def rows(path):
    return [json.loads(row) for row in path.read_text().splitlines()]


def mapping_check(page, counts):
    source, normalized = page["source_text"], page["normalized_text"]
    offset = 0
    by_id = {b["block_key"]: b for b in page["blocks"]}
    for item in page["normalization_map"]:
        start, end = item["normalized_start"], item["normalized_end"]
        a, b = item["source_start"], item["source_end"]
        assert start == offset and start < end and 0 <= a < b <= len(source)
        if item["mode"] == "identity":
            assert normalized[start:end] == source[a:b]
        else:
            assert item["mode"] == "range"
        if item.get("block_key"):
            block = by_id[item["block_key"]]
            assert block["char_start"] <= a < b <= block["char_end"]
            assert block["status"] == "extracted"
        counts["segments"] += 1
        offset = end
    assert offset == len(normalized)
    counts["normalized_codepoints"] += offset
    counts["normalized_pages"] += bool(normalized)
    for block in page["blocks"]:
        source_block = source[block["char_start"]:block["char_end"]]
        assert valid_box(block["bbox"]) and all(valid_box(line["bbox"]) for line in block["lines"])
        assert source_block == block["text"]
        counts["valid_source_block_offsets"] += 1
        if block["normalized_range"]:
            a, b = block["normalized_range"]
            assert block["status"] == "extracted" and block["kind"] in {"paragraph", "heading", "list_item"}
            counts["normalized_blocks"] += 1
            counts["normalized_critical_disagreements"] += critical_tokens(normalized[a:b]) != critical_tokens(source_block)
        else:
            counts["excluded_review_blocks"] += block["status"] == "needs_review"
    assert not page["retrieval_eligible"]
    for figure in page["figures"]:
        assert figure["relationship_status"] == "unverified" and figure["edges"] == []
    for table in page["tables"]:
        assert table["relationship_status"] == "unverified" and table["continuation"]["linked_fragments"] == []


def phase3_quality():
    roots = [PHASE3 / "runs" / run / "summary.json" for run in ("phase3-final", "phase4-regression")]
    prior, current = [json.loads(path.read_text()) for path in roots]
    fields = ("text", "exact_anchor_pages", "reading_order_pairs", "geometry_order_pairs", "table_structure",
              "table_localization", "flow_edges", "route_actions")
    compare = {}
    for mode, results in prior["modes"].items():
        previous, measured = results["all"], current["modes"][mode]["all"]
        compare[mode] = {"identical_quality": all(previous[k] == measured[k] for k in fields),
                         "attempts": measured["performance"]["attempted"], "outcomes": measured["performance"]["statuses"]}
    crops = current["region_comparison"]["reference_regions"]["pooled"]
    crop_pass = prior["region_comparison"]["reference_regions"]["pooled"]["text"] == crops["text"]
    assert all(row["identical_quality"] for row in compare.values()) and crop_pass
    return {"modes": compare, "crop_quality_identical": crop_pass,
            "comparison_attempts": sum(row["attempts"] for row in compare.values()) + crops["performance"]["attempted"]}


def evaluate(page, reference, output, *, geometry_lines=None):
    excerpts = [excerpt_scores(item, output["lines"], .50) for item in reference["excerpts"]]
    order = reference.get("order", [])
    exact = order_scores(order, reference["excerpts"], output)
    geom = order_scores(order, reference["excerpts"],
                        {**output, "lines": geometry_lines if geometry_lines is not None else output["lines"]},
                        geometric=True)
    return {"asset_key": page["asset_key"], "physical_page": page["physical_page"], "language": page["language"],
            "layout": reference.get("layout"), "excerpts": excerpts, "exact_order": exact, "geometry_order": geom}


def aggregate(items):
    return {"pages": len(items), "text": text_metrics([e for item in items for e in item["excerpts"]]),
            **{key: Counter(pair["status"] for item in items for pair in item[key])
               for key in ("exact_order", "geometry_order")}}


def analyze(run, reference_rows, baseline=None):
    page_records = rows(run / "pages.jsonl")
    cases, comparisons, details, counts, label_checks = [], defaultdict(list), [], Counter(), []
    for page in page_records:
        mapping_check(page, counts)
        key = page["asset_key"], page["physical_page"]
        if ref := reference_rows.get(key):
            digest = ref.get("asset_sha256", ref.get("sha256"))
            assert digest == page["asset_sha256"]
            selected = page["candidates"].get(page.get("selected_candidate"), {"lines": []})
            layout = [[{"text": line["text"], "bbox": line["bbox"]}]
                      for block in page["blocks"] for line in block["lines"]]
            production = evaluate(page, ref, {**selected, "source_text": page["source_text"]}, geometry_lines=layout)
            comparisons["production"].append(production)
            for name in ("native", "tesseract", "paddle"):
                if name in page["candidates"]:
                    comparisons[name].append(evaluate(page, ref, page["candidates"][name]))
            if baseline is not None:
                comparisons["phase3_refined"].append(evaluate(page, ref, baseline[key]))
            cases.append({
                "asset_key": key[0], "page": key[1], "layout": ref.get("layout"), "route": page["routing"]["route"],
                "rotation": page.get("rotation"), "candidate": page.get("selected_candidate"),
                "printed_label": page.get("printed_page_label"), "candidates": list(page["candidates"]),
                "status": page["status"], "normalized_blocks": sum(bool(x["normalized_range"]) for x in page["blocks"]),
                "table_iou": max((iou(item["bbox"], ref["table_bbox"]) for item in page["tables"]), default=0)
                if "table_bbox" in ref else None,
                "figure_iou": max((iou(item["bbox"], ref["figure_bbox"]) for item in page["figures"]), default=0)
                if "figure_bbox" in ref else None,
            })
            if ref.get("figure_bbox"):
                label_lines = [[{"text": part["text"], "bbox": part["bbox"]}]
                               for figure in page["figures"] for part in figure["labels"]
                               if part["candidate"].startswith("tesseract_region_")]
                check = [excerpt_scores(item, label_lines, .50) for item in ref["excerpts"]
                         if inside(item["bbox"], ref["figure_bbox"])]
                label_checks.append({"asset_key": key[0], "page": key[1],
                                     "automatic_crop": bool(label_lines), "text": text_metrics(check)})
            details.append(production)
    results = {
        name: {"all": aggregate(items), "arabic_documents": aggregate([q for q in items if q["language"] == "ar"])}
        for name, items in comparisons.items()
    }
    # Only compare alternative candidates on pages with the same evidence
    # available from both engines and the production selection.
    paired_keys = {(item["asset_key"], item["physical_page"]) for item in comparisons.get("paddle", [])}
    results["paddle_paired"] = {
        name: aggregate([q for q in items if (q["asset_key"], q["physical_page"]) in paired_keys])
        for name, items in comparisons.items() if name in {"native", "tesseract", "paddle", "production"}
    }
    cold = run / "summary.cold.json"
    return {
        "quality": results, "normalized_mapping": counts, "cases": cases, "source_bound_errors": details,
        "automatic_crop_label_checks": label_checks,
        "routes": Counter(page["routing"]["route"] for page in page_records),
        "candidate_attempts": Counter(name.split("_region_", 1)[0] for page in page_records for name in page["candidates"]),
        "engine_failures": Counter(d.get("engine", "?") + "/" + d.get("failure", "?")
                                   for page in page_records for d in page["diagnostics"]
                                   if d["code"] == "engine_unavailable_or_failed"),
        "runtime": json.loads((cold if cold.exists() else run / "summary.json").read_text()),
        "review_issue_counts": Counter(row["code"] for row in rows(run / "review_queue.jsonl")),
    }


def read_fresh(root, directory):
    manifest = directory / "manifest.json"
    annotations = json.loads((directory / "verified.json").read_text())
    assert file_hash(manifest) == annotations["manifest_sha256"]
    expected = {(r["asset_id"], r["physical_page"]): r for r in json.loads(manifest.read_text())}
    config = json.loads((REPO / "experiments/extraction/config.json").read_text())
    old = json.loads((PHASE3 / "manifest.json").read_text())["pages"]
    assert not {(r["asset_id"], r["physical_page"]) for r in old} & expected.keys()
    by_key = {r["asset_id"]: r for r in old}
    setup = PHASE3 / "setup/tesseract/usr"
    refs, baseline = {}, {}
    for reference in annotations["references"]:
        k = reference["asset_id"], reference["physical_page"]
        row = expected[k]
        assert all(reference[n] == v for n, v in row.items())
        assert file_hash(directory / "renders" / row["render"]) == row["sha256"]
        path = root / by_key[k[0]]["path"]
        assert file_hash(path) == row["asset_sha256"]
        with pymupdf.open(path) as pdf:
            baseline[k] = benchmark.hybrid(
                pdf[k[1] - 1], config, setup / "bin/tesseract", setup / "share/tesseract-ocr/5/tessdata",
                by_key[k[0]]["language"], allow_regions=True, refined=True,
            )
        refs[k] = reference
    assert set(refs) == set(expected)
    return refs, baseline



def fixture_checks(run):
    pages = {page["physical_page"]: page for page in rows(run / "pages.jsonl")}
    image, overlay, rotated, mixed = (pages[n] for n in (3, 4, 5, 6))
    checked = {
        "blank": pages[2]["status"] == "blank" and "tesseract" not in pages[2]["candidates"],
        "image_only_ocr": image["selected_candidate"] == "tesseract" and "42" in image["source_text"]
                          and not image["candidates"]["native"]["source_text"].strip(),
        "hidden_overlay_conflict_held": overlay["routing"]["route"] == "suspect_overlay"
                                        and "42" in overlay["source_text"] and "74" in overlay["candidates"]["native"]["source_text"]
                                        and "42" not in overlay["normalized_text"],
        "rotated_cropbox": rotated["rotation"] == 90 and rotated["cropbox_pt"] != rotated["mediabox_pt"]
                           and "ROTATED CROPBOX TEST" in rotated["source_text"],
        "automatic_region_with_independent_prose": any(key.startswith("tesseract_region_") for key in mixed["candidates"])
                           and "P-71" in " ".join(label["text"] for figure in mixed["figures"] for label in figure["labels"])
                           and "Independent source paragraph" in mixed["normalized_text"] and "P-71" not in mixed["normalized_text"],
        "decorative": pages[7]["status"] == "decorative" and "tesseract" not in pages[7]["candidates"],
        "table_exclusion": bool(pages[8]["tables"]) and "Cell 42" not in pages[8]["normalized_text"]
                           and "Separate paragraph" in pages[8]["normalized_text"],
    }
    return checked

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-run", type=Path, default=HOME / "pilot-verified")
    parser.add_argument("--confirmation-run", type=Path, default=HOME / "confirmation/run")
    parser.add_argument("--fresh-run", type=Path, default=HOME / "fresh/retested")
    parser.add_argument("--fixtures-run", type=Path, default=HOME / "fixtures/verified")
    args = parser.parse_args()
    root = REPO / json.loads((REPO / "experiments/extraction/config.json").read_text())["collection_root"]
    fresh_refs, baseline = read_fresh(root, HOME / "fresh")
    confirmation_refs, confirmation_baseline = read_fresh(root, HOME / "confirmation")
    assert not set(fresh_refs) & set(confirmation_refs)
    pilot = json.loads((PHASE3 / "references/verified.json").read_text())
    pilot_refs = {(item["key"].rsplit("__", 1)[0], item["physical_page"]): item for item in pilot["pages"]}
    results = {
        "phase3_regression": phase3_quality(),
        "pilot": analyze(args.pilot_run, pilot_refs),
        "first_fresh_initial": analyze(HOME / "fresh/final", fresh_refs, baseline),
        "fresh_retested": analyze(args.fresh_run, fresh_refs, baseline),
        "confirmation": analyze(args.confirmation_run, confirmation_refs, confirmation_baseline),
        "fixtures": analyze(args.fixtures_run, {}),
    }
    results["fixtures"]["checks"] = fixture_checks(args.fixtures_run)
    write_json(local_output(HOME / "evaluation.json"), results)
    for name in ("pilot", "first_fresh_initial", "fresh_retested", "confirmation", "fixtures"):
        s = results[name]
        print(name, s["runtime"]["page_count"], s["runtime"]["page_statuses"], s["normalized_mapping"])


if __name__ == "__main__":
    main()
