"""Local engine adapters and routing rules for the benchmark."""

import csv
import io
from collections import Counter
import re
import subprocess
import unicodedata
import xml.etree.ElementTree as ET

from PIL import Image, ImageChops
import pymupdf


def rect_union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def overlap(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def inside(box, region):
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return region[0] <= x <= region[2] and region[1] <= y <= region[3]


def normalize(box, width, height):
    return [max(0, min(1, value / dimension))
            for value, dimension in zip(box, (width, height, width, height))]


def pdf_box(page, coords):
    rectangle = pymupdf.Rect(coords) * page.rotation_matrix
    return normalize(rectangle, page.rect.width, page.rect.height)


def result(lines, *, text=None, tables=None, headings=None, diagnostics=None):
    return {
        "source_text": "\n".join(" ".join(word["text"] for word in line) for line in lines) if text is None else text,
        "lines": lines, "tables": tables or [], "headings": headings or [],
        "diagnostics": diagnostics or {}, "flowchart_edges": [],
    }


def word(text, bbox):
    return {"text": text, "bbox": bbox}


def native(page, config, *, sorted_text=False, structure=False):
    words_by_line = {}
    for x0, y0, x1, y1, text, block, line, _ in page.get_text("words", sort=sorted_text):
        words_by_line.setdefault((block, line), []).append(word(text, pdf_box(page, (x0, y0, x1, y1))))
    compact = dict(flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)
    layout = page.get_text("dict", **compact)["blocks"]
    spans = [span for block in layout for line in block["lines"] for span in line["spans"]
             if span["text"].strip()]
    sizes = Counter()
    for span in spans:
        sizes[round(span["size"], 1)] += sum(c.isalpha() for c in span["text"])
    body = sizes.most_common(1)[0][0] if sizes else None
    headings = []
    if body:
        for block in layout:
            for line in block["lines"]:
                text = "".join(s["text"] for s in line["spans"]).strip()
                if len(text) >= 5 and max((s["size"] for s in line["spans"]), default=0) >= body * config["heading_font_ratio"]:
                    headings.append({"text": text, "bbox": pdf_box(page, line["bbox"])})
    tables = []
    if structure:
        for table in page.find_tables().tables:
            cells = []
            for ri, (row, values) in enumerate(zip(table.rows, table.extract())):
                for ci, (box, value) in enumerate(zip(row.cells, values)):
                    if box is not None:
                        cells.append({"row": ri, "col": ci, "text": value or "", "bbox": pdf_box(page, box)})
            tables.append({
                "bbox": pdf_box(page, table.bbox), "row_count": table.row_count, "col_count": table.col_count,
                "cells": cells, "header_names": table.header.names, "header_external": table.header.external,
                "header_bbox": pdf_box(page, table.header.bbox) if table.header.bbox else None,
                "continuation_links": [], "footnote_links": [],
            })
    return result(list(words_by_line.values()), text=page.get_text("text", sort=sorted_text),
                  headings=headings, tables=tables)


def poppler(path, number, mode, timeout):
    common = ["pdftotext", "-f", str(number), "-l", str(number), "-enc", "UTF-8"]
    flags = ["-layout"] if mode == "poppler_layout" else []
    text = subprocess.run([*common, *flags, path, "-"], capture_output=True,
                          timeout=timeout, check=True).stdout.decode().rstrip("\f")
    xml = subprocess.run([*common, "-bbox-layout", "-cropbox", path, "-"], capture_output=True,
                         timeout=timeout, check=True).stdout
    # Poppler 24.02 writes PDF BEL glyphs literally, which XML 1.0 forbids.
    xml, removed = re.subn(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]", b"", xml)
    page = ET.fromstring(xml).find(".//{*}page")
    if page is None:
        raise ValueError("Poppler bbox stream has no page")
    width, height = float(page.attrib["width"]), float(page.attrib["height"])
    lines = []
    for line in page.findall(".//{*}line"):
        lines.append([word(element.text or "", normalize(
            [float(element.attrib[k]) for k in ("xMin", "yMin", "xMax", "yMax")], width, height
        )) for element in line.findall("{*}word")])
    return result(lines, text=text, diagnostics={
        "word_geometry_stream": "bbox-layout", "bbox_stream_width_pt": width, "bbox_stream_height_pt": height,
        "xml_invalid_chars_removed": removed,
    })


def tess(page, config, binary, models, language, region=None):
    dpi = config["dpi"]
    clip = None
    if region:
        width, height = page.rect.width, page.rect.height
        x0, y0, x1, y1 = region
        # get_pixmap clips in displayed page coordinates, including rotation.
        clip = pymupdf.Rect(x0 * width, y0 * height, x1 * width, y1 * height)
    pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False, clip=clip)
    command = [binary, "stdin", "stdout", "--tessdata-dir", models, "-l", config["ocr_languages"][language],
               "--dpi", str(dpi), "--oem", "1", "--psm", "11" if region else "3", "-c", "tessedit_create_tsv=1"]
    output = subprocess.run(command, input=pixmap.tobytes("png"), capture_output=True, check=True,
                            timeout=config["timeout_seconds"]).stdout.decode()
    groups, confidence = {}, []
    zoom = dpi / 72
    rows = csv.DictReader(io.StringIO(output), delimiter="\t", quoting=csv.QUOTE_NONE)
    if not rows.fieldnames or "level" not in rows.fieldnames or "text" not in rows.fieldnames:
        raise ValueError("Tesseract did not return a TSV word stream")
    for row in rows:
        if row["level"] != "5" or not row["text"] or not row["text"].strip():
            continue
        x, y, w, h = (int(row[k]) for k in ("left", "top", "width", "height"))
        coords = (x + pixmap.x, y + pixmap.y, x + w + pixmap.x, y + h + pixmap.y)
        box = normalize(coords, page.rect.width * zoom, page.rect.height * zoom)
        key = tuple(row[k] for k in ("block_num", "par_num", "line_num"))
        groups.setdefault(key, []).append(word(row["text"], box))
        confidence.append(float(row["conf"]))
    return result(list(groups.values()), diagnostics={
        "ocr_mean_word_confidence": sum(confidence) / len(confidence) if confidence else None,
        "ocr_words": len(confidence), "ocr_psm": 11 if region else 3,
    })


def image_stats(page, box=None):
    clip = None
    if box:
        clip = pymupdf.Rect(*(v * d for v, d in zip(
            box, (page.rect.width, page.rect.height, page.rect.width, page.rect.height),
        )))
    pix = page.get_pixmap(dpi=72, clip=clip, colorspace=pymupdf.csGRAY, alpha=False)
    image = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    histogram = image.histogram()
    ink = sum(histogram[:210]) / (image.width * image.height)
    edges = []
    for dx, dy in ((1, 0), (0, 1)):
        w, h = image.width - dx, image.height - dy
        if w and h:
            difference = ImageChops.difference(image.crop((0, 0, w, h)), image.crop((dx, dy, w + dx, h + dy)))
            edges.append(sum(difference.histogram()[42:]) / (w * h))
    return ink, max(edges, default=0)


def column_order_warning(lines, language):
    if len(lines) < 20:
        return False
    line_boxes = [rect_union([w["bbox"] for w in line]) for line in lines if line]
    for gutter in (0.34, 0.50, 0.66):
        left = [i for i, b in enumerate(line_boxes) if b[2] < gutter - 0.008 and b[2] - b[0] > 0.11]
        right = [i for i, b in enumerate(line_boxes) if b[0] > gutter + 0.008 and b[2] - b[0] > 0.11]
        crossing = sum(b[0] < gutter < b[2] for b in line_boxes)
        if min(len(left), len(right)) >= 8 and crossing < len(line_boxes) * 0.18:
            first, second = (right, left) if language == "ar" else (left, right)
            inverted = sum(a > b for a in first for b in second) / (len(first) * len(second))
            if inverted >= 0.20:
                return True
    return False


def route(page, raw, language, config, *, allow_regions, refined=False):
    text = raw["source_text"]
    letters = sum(c.isalpha() for c in text)
    garbled = sum(c == "\uFFFD" or unicodedata.category(c) == "Co" for c in text)
    garbled += text.count("(cid:")
    traces = page.get_texttrace()
    hidden = sum(len(span["chars"]) for span in traces if span["type"] == 3 or span["opacity"] == 0)
    visible = sum(len(span["chars"]) for span in traces if span["type"] != 3 and span["opacity"] > 0)
    images = [pdf_box(page, image["bbox"]) for image in page.get_image_info()]
    largest = max(map(area, images), default=0)
    ink, edge = image_stats(page)
    diagnostics = {
        "letters": letters, "garbled": garbled, "largest_image_area": largest, "ink": ink, "edge_density": edge,
        "hidden_text_chars": hidden, "visible_text_chars": visible,
        "order_review": column_order_warning(raw["lines"], language),
    }
    maximum_edge = config["refined_blank_edge"] if refined else config["blank_max_ink"]
    if letters < config["minimum_letters"] and edge < maximum_edge:
        return "retain", [], {**diagnostics, "reason": "low_visual_detail"}
    weak_text = letters < config["minimum_letters"]
    if weak_text and (not refined or not allow_regions):
        return "page_ocr", [], {**diagnostics, "reason": "weak_text"}
    if garbled / max(len(text.strip()), 1) > config["garble_fraction"]:
        return "page_ocr", [], {**diagnostics, "reason": "garbled_text"}
    if largest > config["scan_area"] and hidden > visible:
        return "page_ocr", [], {**diagnostics, "reason": "scan_with_hidden_overlay"}
    candidates = []
    if allow_regions:
        for image in sorted(images, key=area, reverse=True):
            if area(image) < config["region_min_area"] or area(image) >= config["scan_area"]:
                continue
            if any(inside(w["bbox"], image) for line in raw["lines"] for w in line):
                continue
            if any(overlap(image, candidate) / area(image) > 0.5 for candidate in candidates):
                continue
            region_ink, region_edge = image_stats(page, image)
            if 0.008 <= region_ink <= config["region_max_ink"] and region_edge >= 0.005:
                if refined:
                    pad = config["refined_region_padding"]
                    image = [max(0, image[0] - pad), max(0, image[1] - pad),
                             min(1, image[2] + pad), min(1, image[3] + pad)]
                candidates.append(image)
            if len(candidates) >= config["region_max_count"]:
                break
    if candidates:
        return "region_ocr", candidates, {**diagnostics, "reason": "uncovered_raster_regions"}
    if weak_text:
        return "page_ocr", [], {**diagnostics, "reason": "weak_text"}
    return "retain", [], {**diagnostics, "reason": "native_text_available"}


def hybrid(page, config, binary, models, language, *, allow_regions, refined=False):
    baseline = native(page, config)
    action, regions, diagnostics = route(page, baseline, language, config, allow_regions=allow_regions, refined=refined)
    diagnostics.update({"action": action, "regions": regions})
    if action == "page_ocr":
        output = tess(page, config, binary, models, language)
        output["diagnostics"].update({**diagnostics, "requires_review": True})
        return output
    if action == "region_ocr":
        items = [tess(page, config, binary, models, language, region) for region in regions]
        extra_lines = [line for item in items for line in item["lines"]]
        diagnostics["ocr_words"] = sum(item["diagnostics"]["ocr_words"] for item in items)
        diagnostics["requires_review"] = True
        return result(baseline["lines"] + extra_lines, headings=baseline["headings"], diagnostics=diagnostics,
                      text=baseline["source_text"] + "\n" + result(extra_lines)["source_text"])
    baseline["diagnostics"].update(diagnostics)
    return baseline
