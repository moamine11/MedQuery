"""Conservative plain-text normalization with source code-point ranges."""

import re
import unicodedata

PIECES = re.compile(r"[^\S\r\n]+|\r\n?|\n|[^\s]+")
DISCRETIONARY = re.compile(r"(?<=[^\W\d_])\u00ad[ \t]*\r?\n[ \t]*(?=[^\W\d_])")


def normalize(text, source_start=0, *, repair_wrap=True):
    """Return normalized text and ranges. This never folds symbols with NFKC."""
    parts, spans = [], []
    end = 0

    def emit(value, start, stop, mode):
        nonlocal end
        if not value:
            return
        start += source_start
        stop += source_start
        if mode == "identity" and spans and spans[-1]["mode"] == mode and spans[-1]["source_end"] == start:
            spans[-1].update(source_end=stop, normalized_end=end + len(value))
        else:
            spans.append({"normalized_start": end, "normalized_end": end + len(value),
                          "source_start": start, "source_end": stop, "mode": mode})
        parts.append(value)
        end += len(value)

    breaks = list(DISCRETIONARY.finditer(text)) if repair_wrap else []
    cuts = [(0, None), *((match.start(), match.end()) for match in breaks), (len(text), None)]
    pending, pending_lines = None, 0
    for (left, skip_until), (right, _) in zip(cuts, cuts[1:]):
        for match in PIECES.finditer(text, skip_until if skip_until is not None else left, right):
            value = match.group()
            if value.isspace():
                pending = (pending[0] if pending else match.start(), match.end())
                pending_lines += value in {"\r\n", "\n", "\r"}
                continue
            if pending and parts:
                whitespace = "\n\n" if pending_lines > 1 else (
                    "\n" if pending_lines and not repair_wrap else " "
                )
                emit(whitespace, *pending, "identity" if whitespace == text[slice(*pending)] else "range")
            pending, pending_lines = None, 0
            composed = unicodedata.normalize("NFC", value)
            emit(composed, match.start(), match.end(), "identity" if value == composed else "range")
    return "".join(parts), spans


def assemble_normalized(page):
    parts, segments, offset = [], [], 0
    group, barrier = 0, True
    for block in page["blocks"]:
        block["normalized_range"] = None
        block["continuity_group"] = None
        if block["status"] != "extracted" or block["kind"] not in {"paragraph", "heading", "list_item"}:
            barrier |= block["status"] == "needs_review"
            continue
        a, b = block["char_start"], block["char_end"]
        value, mapping = normalize(page["source_text"][a:b], a, repair_wrap=block["kind"] != "heading")
        if not value:
            continue
        if parts:
            # Page assembly already stores a two-newline separator directly
            # before every noninitial block.
            segments.append({"normalized_start": offset, "normalized_end": offset + 2,
                             "source_start": a - 2, "source_end": a, "mode": "identity"})
            parts.append("\n\n")
            offset += 2
        for span in mapping:
            segments.append({**span, "normalized_start": span["normalized_start"] + offset,
                             "normalized_end": span["normalized_end"] + offset, "block_key": block["block_key"]})
        if barrier:
            group += 1
            barrier = False
        block["continuity_group"] = group
        block["normalized_range"] = [offset, offset + len(value)]
        parts.append(value)
        offset += len(value)
    page["normalized_text"] = "".join(parts)
    page["normalization_map"] = segments
