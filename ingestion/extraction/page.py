"""Extract a single verified physical page and retain source evidence."""

import math
import subprocess
from xml.etree.ElementTree import ParseError

from experiments.extraction.engines import area, overlap, rect_union

from . import CONTRACT_VERSION
from .layout import (comparable_geometry, dominant_direction, engine_lines, gutters, line_kind,
                     native_lines, order_lines, printed_label)
from .quality import arabic_quality, assess_block, relative_disagreement, scope_issue
from .routing import choose_page, inspect_page, layout_needed, region_proposals, script_counts
from .structure import build_figures, build_tables


def paragraph_blocks(lines):
    blocks = []
    for line in lines:
        kind = line_kind(line)
        previous = blocks[-1]["lines"][-1] if blocks else None
        x0, y0, x1, y1 = line["bbox"]
        if previous and kind == blocks[-1]["kind"] == "paragraph":
            ax, ay, bx, by = previous["bbox"]
            adjacent_group = line["group"] is not None and line["group"] == previous["group"] and (
                line["line_index"] == previous["line_index"] + 1
            )
            inferred = line["group"] is None and previous["group"] is None
            gap = y0 - by
            if (adjacent_group or inferred) and not (line["vertical"] or previous["vertical"]) and (
                -0.007 <= gap <= (.013 if adjacent_group else .007) and
                min(x1, bx) - max(x0, ax) >= .3 * min(x1 - x0, bx - ax)
            ):
                blocks[-1]["lines"].append(line)
                continue
        blocks.append({"kind": kind, "lines": [line]})
    return blocks


def confidence(raw):
    features = raw["diagnostics"]
    if "ocr_mean_word_confidence" in features:
        return features["ocr_mean_word_confidence"]
    if (mean := features.get("ocr_mean_line_confidence")) is not None:
        return mean * 100
    return None


def select_candidate(candidates, route, features, profile):
    if "native_error" in candidates and candidates.get("poppler", {}).get("lines"):
        return "poppler"
    if route in {"blank", "decorative", "low_detail"}:
        return "native"
    ocr = candidates.get("tesseract", {})
    if ocr.get("lines"):
        if route in {"suspect_overlay", "weak_native", "garbled_native"} or (
            route == "arabic_comparison" and (
                any(reason in features["arabic_reasons"] for reason in ("rtl_suspicious_glyphs", "rtl_script_mismatch_or_mixed"))
                or arabic_quality(ocr["source_text"]) < arabic_quality(candidates["native"]["source_text"])
            )
        ):
            paddle = candidates.get("paddle")
            if (paddle and paddle["lines"] and confidence(paddle) is not None and
                confidence(paddle) >= profile["minimum_ocr_confidence"] and
                (confidence(ocr) is None or confidence(ocr) < profile["minimum_ocr_confidence"] or
                 arabic_quality(ocr["source_text"]) > 1.8 * arabic_quality(paddle["source_text"]) + .03)):
                return "paddle"
            return "tesseract"
    if candidates.get("poppler", {}).get("lines") and (
        route in {"weak_native", "garbled_native", "suspect_overlay"} or not candidates["native"]["lines"]
    ):
        return "poppler"
    if not ocr.get("lines") and candidates.get("paddle", {}).get("lines"):
        return "paddle"
    return "native"


def base_record(page, context):
    return {
        "contract_version": CONTRACT_VERSION, **context,
        "width_pt": page.rect.width, "height_pt": page.rect.height, "rotation": page.rotation,
        "cropbox_pt": list(page.cropbox), "mediabox_pt": list(page.mediabox),
        "native_to_display_matrix": list(page.rotation_matrix),
        "coordinate_system": "displayed-cropbox-normalized-top-left",
        "pdf_page_label": page.get_label() or None, "retrieval_eligible": False,
    }


def process_page(engine, context):
    page, profile = engine.page, engine.profile
    record = base_record(page, context)
    if math.ceil(page.rect.width * profile["dpi"] / 72) * math.ceil(page.rect.height * profile["dpi"] / 72) > profile["max_pixels"]:
        raise ValueError("Page exceeds the configured raster pixel limit")
    page_id = context["asset_key"] + ":" + str(context["physical_page"])
    candidates, issues = {}, []

    def attempt(name, lang, *, box=None, key=None):
        key = key or name
        try:
            value = engine.extract(name, lang, box)
            candidates[key] = value
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, MemoryError, ParseError) as error:
            scope_issue(issues, "engine_unavailable_or_failed", box, severity="diagnostic", engine=name,
                        failure=type(error).__name__)
            return None
        if value["diagnostics"].get("status") == "partial_success":
            scope_issue(issues, "partial_engine_output", box, severity="diagnostic", engine=name)
        return value

    native = attempt("native", context["language"])
    if native is None:
        candidates["native_error"] = {}
        native = {"source_text": "", "lines": [], "headings": [], "diagnostics": {}}
        if engine.tools.get("poppler"):
            attempt("poppler", context["language"])
    lines = native_lines(page, native["headings"]) if "native_error" not in candidates else []
    probe = candidates.get("poppler") if "native_error" in candidates else None
    features = inspect_page(page, probe or native, engine_lines(probe) if probe else lines,
                            context["language"], profile)
    route = choose_page(features, profile)
    lang = features["language"]
    rtl = features["right_to_left"]

    if route in {"weak_native", "garbled_native", "suspect_overlay", "arabic_comparison"} and (
        not candidates.get("poppler", {}).get("lines")
    ):
        attempt("tesseract", lang)
        tess = candidates.get("tesseract", {})
        tess_arabic = script_counts(tess.get("source_text", ""))[0] >= profile["arabic_min_letters"]
        # Arabic native availability is deliberately insufficient. Compare a
        # second OCR engine only where the first comparison leaves uncertainty.
        if (rtl or (lang == "mixed" and tess_arabic)) and engine.tools.get("paddle") and (
            not tess.get("lines") or confidence(tess) is None or confidence(tess) < profile["minimum_ocr_confidence"]
            or (rtl and relative_disagreement(tess["source_text"], native["source_text"]) > profile["arabic_line_disagreement"])
        ):
            attempt("paddle", lang)
        if route != "arabic_comparison" and not tess.get("lines") and engine.tools.get("poppler") and "poppler" not in candidates:
            attempt("poppler", lang)

    if features["grid"] and route not in {"blank", "decorative", "low_detail", "suspect_overlay"}:
        attempt("native_tables", lang)

    regions = [] if route in {"blank", "decorative", "low_detail", "suspect_overlay"} else region_proposals(page, features, lines, profile)
    need_layout = layout_needed(features, candidates.get("native_tables", {}).get("tables"), route)
    # An image-only scan with convincing text-like detail gets OCR first. A
    # low-panel ordinary scan does not need a layout model just for being a scan.
    need_layout &= bool(native["lines"] or regions or features["grid"] or features["vector_panels"] >= 9)
    if need_layout and route not in {"blank", "decorative", "low_detail"} and profile["docling_enabled"]:
        if engine.tools.get("docling"):
            attempt("docling", lang)
        else:
            scope_issue(issues, "complex_layout_engine_unavailable", severity="diagnostic", engine="docling")

    primary = select_candidate(candidates, route, features, profile)
    picked = candidates.get(primary, native)
    selected_lines = lines if primary == "native" else engine_lines(picked)
    for line in selected_lines:
        if not line["style"] and any(
            comparable_geometry(h["bbox"], line["bbox"]) >= .8 and
            relative_disagreement(h["text"], line["text"]) < .2
            for h in candidates.get("docling", {}).get("headings", [])
        ):
            line["style"] = "heading"
    selected_boxes = gutters(selected_lines, profile["minimum_column_lines"])
    cuts = features["gutters"] if len(features["gutters"]) > len(selected_boxes) else selected_boxes
    selected_lines, ambiguous = order_lines(selected_lines, cuts, rtl)

    tables = build_tables(candidates, features, lines, page_id)
    figures = build_figures(candidates, features, lines, regions, page_id)
    for structure in tables + figures:
        structure["source_anchor"] = {"physical_page": record["physical_page"], "bbox": structure["bbox"],
                                      "asset_sha256": record["asset_sha256"]}
    if route == "low_detail":
        scope_issue(issues, "low_visual_detail_unclassified")
    if route == "suspect_overlay":
        scope_issue(issues, "suspect_text_layer")
    if need_layout and not candidates.get("docling") and (features["rotated_text"] or ambiguous):
        scope_issue(issues, "unverified_reading_order")
    if ambiguous:
        scope_issue(issues, "unverified_reading_order", candidate=primary)

    # Use native words around a crop. Figure labels remain attached to their
    # source region even if they are not part of the page's selected text.
    for i, figure in enumerate(figures, 1):
        box = figure["bbox"]
        covered = [line for line in selected_lines if comparable_geometry(line["bbox"], box) >= .45]
        region_key = f"tesseract_region_{i}"
        proposed = any(comparable_geometry(box, region) >= .6 for region in regions)
        if proposed and primary not in {"tesseract", "paddle"}:
            crop = attempt("tesseract", lang, box=box, key=region_key)
            if crop and crop["lines"]:
                figure["labels"] = [{"text": line["text"], "bbox": line["bbox"], "candidate": region_key}
                                    for line in engine_lines(crop)]
        if not figure["labels"]:
            figure["labels"] = [{"text": line["text"], "bbox": line["bbox"], "candidate": primary} for line in covered]
    for table in tables:
        table["fragments"] = [{"physical_page": record["physical_page"], "bbox": table["bbox"]}]

    label, label_evidence = printed_label(selected_lines, record["pdf_page_label"])
    if label_evidence:
        label_evidence["candidate"] = primary
    alternatives = {}
    for name in ("native", "tesseract", "paddle", "poppler"):
        if name == primary or name not in candidates:
            continue
        alternatives[name] = lines if name == "native" else engine_lines(candidates[name])

    blocks, offset, text_parts = [], 0, []
    structures = tables + figures
    # A caption or heuristic region can be insufficient to locate the whole
    # structure. Until bounded, adjacent prose may depend on omitted context.
    uncertain = [item for item in structures if item["extent_uncertain"]]
    for ordinal, block in enumerate(paragraph_blocks(selected_lines), 1):
        entries = block["lines"]
        exact = "\n".join(line["text"] for line in entries)
        bbox = rect_union([line["bbox"] for line in entries])
        block.update({"block_key": f"{page_id}:block:{ordinal}", "ordinal": ordinal, "bbox": bbox,
                      "text": exact, "candidate": primary, "char_start": offset, "char_end": offset + len(exact)})
        block["direction"] = dominant_direction(exact, "rtl" if rtl else "ltr")
        ar, latin = script_counts(exact)
        block["language"] = "ar" if ar > latin else context["language"]
        assess_block(block, alternatives=alternatives, arabic=rtl, ambiguous=ambiguous,
                     page_ocr=primary in {"tesseract", "paddle"}, confidence=confidence(picked),
                     structures=structures, profile=profile)
        if label_evidence and comparable_geometry(bbox, label_evidence["bbox"]) >= .85 and exact.strip() == label:
            block.update(kind="printed_label", status="decorative")
        if route == "low_detail":
            block["status"] = "needs_review"
        if block["kind"] in {"paragraph", "list_item"} and any(
            comparable_geometry(bbox, item["search_bbox"]) >= .38 for item in uncertain
        ):
            scope_issue(block["diagnostics"], "structure_extent_unverified", bbox)
            block["status"] = "needs_review"
        if route in {"suspect_overlay", "garbled_native"} and primary in {"native", "poppler"}:
            scope_issue(block["diagnostics"], "suspect_text_layer_without_pixel_candidate", bbox)
            block["status"] = "needs_review"
        blocks.append(block)
        text_parts.append(exact)
        offset += len(exact) + 2

    if route in {"blank", "decorative"} and not blocks:
        status = route
    elif not blocks:
        scope_issue(issues, "missing_text")
        status = "needs_review" if route == "low_detail" else "unreadable"
    else:
        status = "needs_review" if any(i["severity"] == "review" for i in issues) or structures or any(
            block["status"] == "needs_review" for block in blocks
        ) else "extracted"
    if route in {"weak_native", "garbled_native", "suspect_overlay", "arabic_comparison"} and not (
        candidates.get("tesseract", {}).get("lines") or candidates.get("paddle", {}).get("lines")
    ):
        scope_issue(issues, "expected_pixel_text_unavailable")
        if status == "extracted":
            status = "needs_review"
        for block in blocks:
            if block["status"] == "extracted":
                block["status"] = "needs_review"
    record.update({
        "source_text": "\n\n".join(text_parts), "blocks": blocks, "status": status,
        "printed_page_label": label, "printed_label_evidence": label_evidence,
        "tables": tables, "figures": figures, "selected_candidate": primary, "candidates": candidates,
        "routing": {"route": route, "regions_proposed": regions, "observations": features,
                    "column_boundaries": cuts, "layout_requested": bool(need_layout)},
        "diagnostics": issues,
    })
    return record
