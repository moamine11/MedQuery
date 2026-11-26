"""Deterministic route candidates built from observed text and page graphics."""

import re
import unicodedata

from experiments.extraction.engines import area, image_stats, overlap, pdf_box

from .layout import FIGURE_CAPTION, TABLE_CAPTION, gutters, valid_box

ODD_ARABIC = re.compile(r"(?<!\w)ال(?!\w)|اال|[ا-ي][A-Za-z](?![A-Za-z])|(?<![A-Za-z])[A-Za-z][ا-ي]")


def broken_chars(text):
    return sum(char in "\ufffd\ufffe\uffff" or
               (unicodedata.category(char) in {"Cc", "Co", "Cs"} and char not in "\r\n\t")
               for char in text) + text.count("(cid:")


def script_counts(text):
    arabic = sum(unicodedata.bidirectional(c) == "AL" and c.isalpha() for c in text)
    latin = sum(c.isalpha() and unicodedata.name(c, "").startswith("LATIN ") for c in text)
    return arabic, latin


def graphics(page):
    images = [box for item in page.get_image_info() if valid_box(box := pdf_box(page, item["bbox"]))]
    horizontal = vertical = panels = 0
    boxes = []
    for path in page.get_drawings():
        panel = pdf_box(page, path["rect"])
        if valid_box(panel) and .006 <= area(panel) <= .55:
            boxes.append(panel)
            panels += 1
        for item in path["items"]:
            if item[0] == "re":
                box = item[1] * page.rotation_matrix
                if .03 * page.rect.width <= box.width <= .95 * page.rect.width and .005 * page.rect.height <= box.height:
                    horizontal += 2
                    vertical += 2
            elif item[0] == "l":
                a, b = [p * page.rotation_matrix for p in item[1:]]
                dx, dy = abs(a.x - b.x), abs(a.y - b.y)
                horizontal += dx >= .08 * page.rect.width and dy < 1
                vertical += dy >= .05 * page.rect.height and dx < 1
    return images, {"grid": horizontal >= 5 and vertical >= 3, "lines": [horizontal, vertical],
                    "vector_panels": panels, "panels": boxes}


def inspect_page(page, raw, lines, language, profile):
    text = raw["source_text"]
    letters = sum(c.isalpha() for c in text)
    broken = broken_chars(text)
    den = max(len(text.strip()), 1)
    a, l = script_counts(text)
    lang = language.split("-")[0]
    rtl = lang == "ar" or (a >= profile["arabic_min_letters"] and a >= l)
    lang = "mixed" if (lang == "mul" or (a >= profile["arabic_min_letters"] and l >= 35 and (not rtl or l / max(a + l, 1) > .3))) else (
        "ar" if rtl else lang if lang in ("fr", "en") else "und"
    )
    images, drawing = graphics(page)
    traces = page.get_texttrace()
    hidden = sum(len(x["chars"]) for x in traces if x["type"] == 3 or x["opacity"] == 0)
    visible = sum(len(x["chars"]) for x in traces) - hidden
    ink, edge = image_stats(page)
    largest = max((area(box) for box in images), default=0)
    cuts = gutters(lines, profile["minimum_column_lines"])
    rotated_text = any(line["vertical"] for line in lines)
    table_caption = [line["bbox"] for line in lines if TABLE_CAPTION.match(line["text"])]
    figure_caption = [line["bbox"] for line in lines if FIGURE_CAPTION.match(line["text"])]
    arabic_reasons = []
    if rtl and letters >= profile["arabic_min_letters"]:
        if cuts:
            arabic_reasons.append("rtl_columns")
        if figure_caption or drawing["vector_panels"] >= 9:
            arabic_reasons.append("rtl_complex_graphics")
        if broken or ODD_ARABIC.search(text):
            arabic_reasons.append("rtl_suspicious_glyphs")
        if a < profile["arabic_min_letters"] or l / max(a + l, 1) > profile["arabic_latin_fraction"]:
            arabic_reasons.append("rtl_script_mismatch_or_mixed")
    return {
        "language": lang, "right_to_left": rtl, "letters": letters, "broken_chars": broken,
        "garbled": broken / den > profile["garble_fraction"], "arabic_letters": a, "latin_letters": l,
        "arabic_reasons": arabic_reasons, "ink": ink, "edge_density": edge, "largest_raster": largest,
        "raster_boxes": images, "hidden_chars": hidden, "visible_chars": visible, "gutters": cuts,
        "table_captions": table_caption, "figure_captions": figure_caption, "rotated_text": rotated_text,
        **drawing,
    }


def choose_page(features, profile):
    weak = features["letters"] < profile["minimum_letters"]
    if not features["letters"] and features["ink"] < .00015 and features["edge_density"] < .00015:
        return "blank"
    if not features["letters"] and not features["hidden_chars"] and features["ink"] >= .98 and features["edge_density"] == 0:
        return "decorative"
    if weak and features["edge_density"] < profile["blank_edge"]:
        # Sparse dark marks may be a physical label, a diagram or faint clinical text.
        return "low_detail"
    if features["largest_raster"] >= profile["scan_area"] and features["hidden_chars"] > features["visible_chars"]:
        return "suspect_overlay"
    if weak:
        return "weak_native"
    if features["garbled"]:
        return "garbled_native"
    if features["arabic_reasons"]:
        return "arabic_comparison"
    return "native"


def region_proposals(page, features, lines, profile):
    proposals = []
    for box in sorted(features["raster_boxes"], key=area, reverse=True):
        size = area(box)
        if not profile["region_min_area"] <= size < profile["scan_area"]:
            continue
        if any(overlap(box, b) / size >= .5 for b in proposals):
            continue
        coverage = sum(overlap(line["bbox"], box) for line in lines) / size
        if coverage > profile["native_region_max_text_density"]:
            continue
        ink, detail = image_stats(page, box)
        if detail < profile["region_min_detail"] or not .007 <= ink <= .82:
            continue
        d = profile["region_padding"]
        proposals.append([max(0, box[0] - d), max(0, box[1] - d), min(1, box[2] + d), min(1, box[3] + d)])
    return proposals[:profile["region_max_count"]]


def layout_needed(features, native_tables, route):
    return bool(features["table_captions"] or features["figure_captions"] or features["rotated_text"]
                or features["vector_panels"] >= 9 or features["gutters"]
                or route in {"suspect_overlay", "weak_native"}
                or (features["grid"] and not native_tables))
