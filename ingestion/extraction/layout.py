"""Displayed page geometry, native line fidelity and conservative column order."""

from collections import Counter, defaultdict
import re
import unicodedata

import pymupdf

from experiments.extraction.engines import area, overlap, pdf_box, rect_union

HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:[.)])?\s+\D")
LIST = re.compile(r"^\s*(?:[•▪◦●*]|[-–]\s|\(?[\d٠-٩]+[.)]\s|[a-zA-Z][.)]\s)")
TABLE_CAPTION = re.compile(r"^\s*(?:table(?:au)?\s+(?:\d+(?:\.\d+)*|[IVX]+)(?=[\s.:—–-]|$)|الجدول(?=[\s\d]))", re.I)
FIGURE_CAPTION = re.compile(r"^\s*(?:fig(?:ure|ura)?\.?\s*\d+|(?:algorithme|flowchart)\s*[:\d]|الشكل(?=[\s\d])|خوارزمية\s*[:\d])", re.I)
FOOTNOTE = re.compile(r"^\s*(?:[*†‡§]|[a-z]\)\s|[⁎¹²³⁴⁵⁶⁷⁸⁹])")
PRINT_LABEL = re.compile(r"(?:[A-Z]{1,4}[-–])?\d{1,4}|[ivxlcdmIVXLCDM]{1,8}")


def valid_box(box):
    return bool(box and len(box) == 4 and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)


def dominant_direction(text, fallback="ltr"):
    strong = Counter(unicodedata.bidirectional(char) for char in text)
    rtl = strong["AL"] + strong["R"]
    return "rtl" if rtl > strong["L"] else "ltr" if strong["L"] else fallback


def native_lines(page, headings):
    raw = page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)
    lines = []
    for block in raw["blocks"]:
        for i, line in enumerate(block.get("lines", [])):
            text = "".join(span["text"] for span in line["spans"])
            bbox = pdf_box(page, line["bbox"])
            if not text.strip() or not valid_box(bbox):
                continue
            style = "heading" if any(overlap(bbox, h["bbox"]) / area(bbox) > .8 for h in headings) else None
            vector = pymupdf.Point(*line["dir"])
            displayed = vector * page.rotation_matrix - pymupdf.Point(0, 0) * page.rotation_matrix
            lines.append({"text": text, "bbox": bbox, "style": style, "group": block["number"],
                          "vertical": abs(displayed.y) > abs(displayed.x), "line_index": i,
                          "writing_size": max(span["size"] for span in line["spans"])})
    return lines


def engine_lines(raw):
    return [{"text": " ".join(word["text"] for word in words), "bbox": rect_union([word["bbox"] for word in words]),
             "group": None, "style": None, "vertical": False}
            for line in raw["lines"] if (words := [word for word in line
                                                   if word["text"].strip() and valid_box(word["bbox"])])]


def gutters(lines, minimum):
    boxes = [line["bbox"] for line in lines if .075 <= line["bbox"][2] - line["bbox"][0] <= .7
             and .055 < line["bbox"][1] < .945 and not line["vertical"]]
    if len(boxes) < 2 * minimum:
        return []
    candidates = []
    for i in range(20, 81):
        x = i / 100
        left = [b for b in boxes if b[2] <= x - .006]
        right = [b for b in boxes if b[0] >= x + .006]
        if min(len(left), len(right)) < minimum or len(boxes) - len(left) - len(right) > .15 * len(boxes):
            continue
        l0, l1 = min(b[1] for b in left), max(b[3] for b in left)
        r0, r1 = min(b[1] for b in right), max(b[3] for b in right)
        if min(l1, r1) - max(l0, r0) < .08:
            continue
        candidates.append(x)
    groups = []
    for value in candidates:
        if groups and value - groups[-1][-1] < .011:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [round(sum(group) / len(group), 4) for group in groups if len(group) >= 2][:3]


def order_lines(lines, boundaries, rtl):
    if not boundaries:
        # A y-sort interleaves dense native columns when the gutter is
        # inconclusive. Keep the parser sequence in that case.
        return lines, False
    # Full-width material partitions the page into bands. A spanning line that
    # overlaps column lines is flagged as ambiguous and kept in source order.
    spanning, columns = [], []
    for line in lines:
        a, top, b, bottom = line["bbox"]
        if any(a + .004 < cut < b - .004 for cut in boundaries):
            spanning.append(line)
        else:
            column = sum((a + b) / 2 > cut for cut in boundaries)
            columns.append((line, top, bottom, column))
    if any(not s.get("style") and min(s["bbox"][3], bottom) - max(s["bbox"][1], top) > .004
           for s in spanning for _, top, bottom, _ in columns):
        return lines, True
    spanning.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))
    bands = defaultdict(list)
    for line, top, bottom, column in columns:
        band = sum((top + bottom) / 2 >= (s["bbox"][1] + s["bbox"][3]) / 2 for s in spanning)
        bands[band].append((line, column))
    ordered = []
    for i in range(len(spanning) + 1):
        ordered.extend(line for line, _ in sorted(bands[i], key=lambda item: (
            -item[1] if rtl else item[1], item[0]["bbox"][1], item[0]["bbox"][0],
        )))
        if i < len(spanning):
            ordered.append(spanning[i])
    return ordered, False


def line_kind(line):
    text = line["text"]
    if TABLE_CAPTION.match(text) or FIGURE_CAPTION.match(text):
        return "caption"
    if FOOTNOTE.match(text) and line["bbox"][1] > .65:
        return "footnote"
    if line.get("style") == "heading":
        return "heading"
    if LIST.match(text):
        return "list_item"
    return "paragraph"


def printed_label(lines, pdf_label):
    options = [line for line in lines if (line["bbox"][1] >= .945 or line["bbox"][3] <= .048)
               and PRINT_LABEL.fullmatch(line["text"].strip())]
    if len(options) != 1 and pdf_label:
        options = [line for line in options if line["text"].strip() == pdf_label]
    if len(options) != 1:
        return None, None
    line = options[0]
    return line["text"].strip(), {"bbox": line["bbox"], "method": "marginal_text_candidate",
                                  "visually_verified": False}


def comparable_geometry(a, b):
    return overlap(a, b) / min(area(a), area(b)) if valid_box(a) and valid_box(b) else 0
