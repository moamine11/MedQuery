"""Finalize document-level structure and derive traceable normalized text."""

from collections import Counter
import re

from .layout import HEADING
from .normalize import assemble_normalized
from .quality import critical_tokens

MARGIN_TOKEN = re.compile(r"\d|[%<>=≤≥±]")
MARGIN_DIRECTIVE = re.compile(r"\b(?:should|must|avoid|recommend\w*|contraindicat\w*|"
                              r"doit|éviter|recommand\w*)\b|توصية|ينبغي|يُوصى|يوصى", re.I)


def margin_signature(block, page):
    bbox = block["bbox"]
    if bbox[3] <= .065:
        zone = "header"
    elif bbox[1] >= .935:
        zone = "footer"
    else:
        return None
    text = " ".join(block["text"].split())
    if (not text or len(text) > 105 or len(text.split()) > 14 or MARGIN_TOKEN.search(text)
        or critical_tokens(text) or MARGIN_DIRECTIVE.search(text) or block["kind"] not in {"heading", "paragraph"} or page["status"] in {"failed", "unreadable"}
    ):
        return None
    return (zone, text)


def finalize_asset(pages):
    repetitions = Counter()
    for page in pages:
        repetitions.update({margin_signature(block, page) for block in page["blocks"]} - {None})
    sections, parents, active = [], {}, None
    previous = None
    last_page = None
    for page in pages:
        if last_page is None or page["physical_page"] != last_page + 1 or page["status"] in {"failed", "unreadable"}:
            active = None
            parents = {}
        last_page = page["physical_page"]
        if previous and page["physical_page"] == previous["physical_page"] + 1:
            mark_possible_continuations(previous, page)
        previous = page
        for block in page["blocks"]:
            repeat = margin_signature(block, page)
            if repeat and repetitions[repeat] >= 3 and block["status"] == "extracted":
                block.update(kind=f"repeated_{repeat[0]}", status="decorative")
            if block["kind"] == "heading":
                match = HEADING.match(block["text"])
                label = match[1] if match else None
                parent = label.rpartition(".")[0] if label else None
                ordinal = len(sections) + 1
                active = {
                    "section_key": f"{page['asset_key']}:section:{ordinal}", "ordinal": ordinal,
                    "heading": block["text"], "section_label": label,
                    "parent_section_key": parents.get(parent), "start_page": page["physical_page"],
                    "end_page": page["physical_page"], "anchors": [], "status": block["status"],
                }
                sections.append(active)
                if label:
                    parents[label] = active["section_key"]
            if active:
                active["anchors"].append({"physical_page": page["physical_page"], "bbox": block["bbox"],
                                         "char_start": block["char_start"], "char_end": block["char_end"],
                                         "block_key": block["block_key"]})
                active["end_page"] = page["physical_page"]
                block["section_key"] = active["section_key"]
        assemble_normalized(page)
    return sections



def mark_possible_continuations(previous, page):
    def column_counts(table):
        return {c["col_count"] for c in table["grid"]["candidates"] if c.get("col_count")}
    for old in previous.get("tables", []):
        for new in page.get("tables", []):
            if (old["bbox"][3] < .76 or new["bbox"][1] > .22 or
                    max(abs(old["bbox"][i] - new["bbox"][i]) for i in (0, 2)) > .04 or
                    not column_counts(old) & column_counts(new)):
                continue
            new["continuation"].setdefault("possible_predecessors", []).append(
                {"physical_page": previous["physical_page"], "table_key": old["key"], "bbox": old["bbox"]}
            )
            new["diagnostics"].append({"code": "possible_unverified_table_continuation", "bbox": new["bbox"],
                                       "severity": "review", "preceding_page": previous["physical_page"],
                                       "preceding_table_key": old["key"]})

def review_entries(page, identity):
    components = [("page", None, page["diagnostics"])]
    components.extend(("block", b["block_key"], b["diagnostics"]) for b in page["blocks"])
    for kind, key in (("table", "tables"), ("figure", "figures")):
        components.extend((kind, item["key"], item["diagnostics"]) for item in page[key])
    for scope, component, diagnostics in components:
        for issue in diagnostics:
            if issue["severity"] != "review":
                continue
            if scope == "block" and issue["code"] in {"unresolved_structure", "unverified_reading_order"}:
                # The owning table, figure or page already has a review entry.
                continue
            row = {name: page[name] for name in ("asset_key", "asset_sha256", "physical_page", "source_path")}
            row.update(scope=scope, component_key=component, status="needs_review",
                       printed_page_label=page.get("printed_page_label"), **issue)
            yield {**row, "review_key": identity(row)}
