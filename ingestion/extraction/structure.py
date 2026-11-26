"""Source-bound table and figure candidates with review-gated relationships."""

from experiments.extraction.engines import area, rect_union

import re

from .layout import TABLE_CAPTION, comparable_geometry, engine_lines, valid_box
from .quality import scope_issue

CONTINUED = ("continued", "continuation", "suite", "تابع", "تتمة")


def candidate_groups(entries):
    groups = []
    for candidate in entries:
        if not valid_box(candidate["bbox"]):
            continue
        group = next((group for group in groups if any(
            comparable_geometry(item["bbox"], candidate["bbox"]) > .65 for item in group
        )), None)
        if group is None:
            groups.append([candidate])
        else:
            group.append(candidate)
    return groups


def add_caption_fragments(entries, captions, lines, native_panels, kind):
    for caption in captions:
        if any(comparable_geometry(item["bbox"], caption) >= .25 or (
            item["bbox"][0] <= caption[2] and item["bbox"][2] >= caption[0] and
            min(abs(item["bbox"][1] - caption[3]), abs(item["bbox"][3] - caption[1])) <= .09
        ) for item in entries):
            continue
        nearby = [panel for panel in native_panels if panel[0] < caption[2] and panel[2] > caption[0] and
                  (0 <= panel[1] - caption[3] < .2 or 0 <= caption[1] - panel[3] < .12)]
        # A located caption without bounded content is held separately. The
        # page's normalization gate also applies if it cannot bound the structure.
        bbox = rect_union([caption, *nearby])
        # Without visual figure evidence, the text may be a caption deferred
        # from another page. Hold the caption and linked prose, not both columns.
        if kind == "figure" and not nearby:
            search = bbox
        else:
            next_heading = min((line["bbox"][1] for line in lines if line.get("style") == "heading"
                                and line["bbox"][1] > bbox[3] + .035), default=.935)
            search = [max(.0, bbox[0] - .05), bbox[1], min(1., max(.93, bbox[2] + .03)),
                      min(1., max(bbox[3], next_heading))]
        entries.append({"engine": "caption", "bbox": bbox, "search_bbox": search, "extent_uncertain": True})



def with_candidate_relationships(engine, table):
    cells = [{"cell_key": f"{cell['row']}:{cell['col']}", **cell} for cell in table.get("cells", [])]
    headers = [cell for cell in cells if cell.get("column_header")]
    if engine == "native_tables" and table.get("header_names") and not table.get("header_external"):
        headers = [cell for cell in cells if cell["row"] == 0 and
                   cell["col"] < len(table["header_names"]) and cell["text"] == table["header_names"][cell["col"]]]
    positioned = [cell for cell in cells if valid_box(cell.get("bbox"))]
    for cell in cells:
        cell["header_candidate_keys"] = [header["cell_key"] for header in headers if
                                         header["row"] < cell["row"] and
                                         header["col"] <= cell["col"] < header["col"] + header.get("col_span", 1)]
        if engine != "native_tables" or not valid_box(cell.get("bbox")):
            continue
        # Native bbox evidence can indicate merges where an interior edge
        # in another row or column falls strictly inside this cell.
        for dimension, index in (("row", 1), ("col", 0)):
            start, end = cell["bbox"][index], cell["bbox"][index + 2]
            axes = sorted({round(other["bbox"][index], 3) for other in positioned
                           if start + .003 < other["bbox"][index] < end - .003})
            if axes:
                cell[f"{dimension}_span_candidate"] = len(axes) + 1
    return {"engine": engine, **table, "cells": cells}


def continued_fragment(lines):
    """Bound explicit source continuation cues independently of table models."""
    for caption in lines:
        if not TABLE_CAPTION.match(caption["text"]):
            continue
        top = caption["bbox"][1]
        same_row = [item for item in lines if abs(item["bbox"][1] - top) < .012]
        header = " ".join(line["text"] for line in same_row)
        if not any(cue in header.casefold() for cue in CONTINUED):
            continue
        markers = [item for item in lines if item["bbox"][1] > top + .08 and
                   item["text"].strip(" .()").casefold() in CONTINUED]
        # A terminal cue defines the visible extent. Without it keep only the
        # caption-derived guidance and let the uncertainty gate scope the page.
        if markers:
            end = min(marker["bbox"][3] for marker in markers)
            body = [item["bbox"] for item in lines if top - .004 <= item["bbox"][1] and item["bbox"][3] <= end + .004]
            if body:
                return {"engine": "source_continuation_cues", "bbox": rect_union(body),
                        "extent_uncertain": False, "continuation_cue": header}
    return None

def build_tables(candidates, features, lines, page_id):
    entries = [with_candidate_relationships(name, table) for name in ("native_tables", "docling")
               for table in candidates.get(name, {}).get("tables", []) if valid_box(table["bbox"]) and
               not (area(table["bbox"]) < .05 and (table["bbox"][3] < .08 or table["bbox"][1] > .92))]
    if fragment := continued_fragment(lines):
        entries.insert(0, fragment)
    add_caption_fragments(entries, features["table_captions"], lines, features["panels"], "table")
    if not entries and features["grid"]:
        panels = features["panels"]
        entries.append({"engine": "vector_grid", "bbox": rect_union(panels) if panels else [0, 0, 1, 1],
                        "extent_uncertain": True})
    tables = []
    for i, group in enumerate(candidate_groups(entries), 1):
        bbox = rect_union([x["bbox"] for x in group])
        matching = [line for line in lines if (line["bbox"] in features["table_captions"]
                    and abs(line["bbox"][1] - bbox[1]) < .13)]
        caption = " ".join(line["text"] for line in matching) or None
        note_lines = [*lines, *engine_lines(candidates.get("tesseract", {"lines": []}))]
        notes = [line for line in note_lines if re.search(r"^[*†‡§]|\*{1,2}\s*:", line["text"].lstrip())
                 and -0.015 <= line["bbox"][1] - bbox[3] <= .10]
        table = {
            "key": f"{page_id}:table:{i}", "kind": "table", "bbox": bbox, "caption": caption,
            "status": "needs_review", "relationship_status": "unverified",
            "extent_uncertain": not any(item.get("cells") or item.get("continuation_cue") for item in group),
            "search_bbox": rect_union([item.get("search_bbox", item["bbox"]) for item in group]),
            "grid": {"candidates": group},
            "footnote_candidates": [{"text": n["text"], "bbox": n["bbox"]} for n in notes],
            "continuation": {"cue": next((item["continuation_cue"] for item in group if item.get("continuation_cue")),
                                         caption if caption and any(x in caption.casefold() for x in CONTINUED) else None),
                             "linked_fragments": []},
            "diagnostics": [],
        }
        scope_issue(table["diagnostics"], "unresolved_table_relationship", bbox,
                    candidate_engines=[x["engine"] for x in group])
        tables.append(table)
    return tables


def build_figures(candidates, features, lines, regions, page_id):
    entries = [{"engine": "docling", **f} for f in candidates.get("docling", {}).get("diagnostics", {}).get("figures", [])
               if valid_box(f["bbox"]) and area(f["bbox"]) >= .03]
    entries.extend({"engine": "raster", "bbox": box} for box in regions)
    add_caption_fragments(entries, features["figure_captions"], lines, features["panels"], "figure")
    if not entries and features["vector_panels"] >= 9 and not features["grid"]:
        panels = features["panels"]
        bounds = rect_union(panels)
        entries.append({"engine": "vector_panels", "bbox": bounds, "extent_uncertain": True})
    figures = []
    for i, group in enumerate(candidate_groups(entries), 1):
        bbox = rect_union([x["bbox"] for x in group])
        figures.append({
            "key": f"{page_id}:figure:{i}", "kind": "figure", "bbox": bbox,
            "status": "needs_review", "candidate_regions": group,
            "extent_uncertain": all(x.get("extent_uncertain", False) for x in group),
            "search_bbox": rect_union([item.get("search_bbox", item["bbox"]) for item in group]),
            "caption_candidates": [{"text": line["text"], "bbox": line["bbox"]} for line in lines
                                   if line["bbox"] in features["figure_captions"] and
                                   min(abs(line["bbox"][1] - bbox[3]), abs(line["bbox"][3] - bbox[1])) <= .13],
            "labels": [], "edges": [], "relationship_status": "unverified",
            "diagnostics": [{"code": "unresolved_figure_relationship", "severity": "review", "bbox": bbox}],
        })
    return figures
