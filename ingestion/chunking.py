"""Citation-aware, review-gated text chunking for extracted MedQuery documents."""

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

from ingestion.extraction.config import ROOT, canonical, fingerprint
from ingestion.extraction.layout import valid_box

DEFAULT_TOKENIZER_PATH = ROOT / "data/ingestion/phase5/tokenizer/tokenizer.json"
ACCEPTED_KINDS = frozenset({"paragraph", "heading", "list_item"})
IGNORED_PAGE_STATUSES = frozenset({"failed", "unreadable", "blank", "decorative"})

# Abbreviations that should not trigger sentence splits before a space/number.
_ABBREV_RE = re.compile(
    r"(?:\b(?:Art|Fig|Tab|Dr|Pr|Prof|No|N°|Vol|Ed|Ref|vs|approx|env|cf|al|etc|i\.e|e\.g)|"
    r"\b[A-ZÀ-ÖØ-Ýa-zà-öø-ÿ])\.$",
    re.UNICODE,
)
# Number/comparator tokens that should stay attached to the following unit/value.
_QUANTITY_LEFT_RE = re.compile(r"(?:[≤≥<>±~≈]|\d+(?:[.,٫/:-]\d+)*)$")
_UNIT_RIGHT_RE = re.compile(
    r"^(?:%|mg|µg|ug|mcg|g|kg|ml|mL|L|dL|mmol|µmol|mol|UI|IU|mIU|kPa|mmHg|cm|mm|m²|kg/m²|"
    r"j|jour|jours|sem|mois|an|ans|h|min|s|ملغ|غ|كغ|مل|لتر|وحدة)\b",
    re.IGNORECASE | re.UNICODE,
)
_NEGATION_LEFT_RE = re.compile(
    r"(?:\b(?:ne|n['’]|pas|non|sans|aucun|aucune|jamais|ni|not|no|never|without)|"
    r"(?:لا|غير|عدم|دون|ليس|ليست))$",
    re.IGNORECASE | re.UNICODE,
)


def read_jsonl(path):
    """Read UTF-8 JSONL strictly by physical newline boundaries (never str.splitlines)."""
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            stripped = line.rstrip("\r\n")
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path, rows):
    """Write rows as deterministic UTF-8 JSONL."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class TokenCounter:
    """Actual tokenizer counter backed by HuggingFace tokenizers (with optional test fallback)."""

    def __init__(self, tokenizer_path=None, *, fallback=False):
        self.path = Path(tokenizer_path) if tokenizer_path else DEFAULT_TOKENIZER_PATH
        self._tokenizer = None
        if self.path.is_file():
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(str(self.path))
            self.name = f"hf_tokenizers:{self.path.name}"
            self.fingerprint = sha256(self.path.read_bytes()).hexdigest()[:16]
        elif fallback:
            self.name = "regex_fallback"
            self.fingerprint = "regex_fallback_v1"
        else:
            raise FileNotFoundError(
                f"Tokenizer file not found: {self.path}. Provide --tokenizer or ensure the Phase 5 tokenizer is cached."
            )

    def count(self, text):
        if not text:
            return 0
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
        return len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE))

    def encode_offsets(self, text):
        """Return list of (start_char, end_char) offsets for each token in text."""
        if not text:
            return []
        if self._tokenizer is not None:
            enc = self._tokenizer.encode(text, add_special_tokens=False)
            return [(a, b) for a, b in enc.offsets if b > a]
        return [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text, re.UNICODE)]


@dataclass(frozen=True)
class ProseUnit:
    asset_key: str
    record_id: str
    asset_sha256: str
    source_path: str
    revision_key: str | None
    language: str
    physical_page: int
    printed_page_label: str | None
    block_key: str
    kind: str
    section_key: str | None
    section_label: str | None
    heading: str | None
    parent_section_key: str | None
    continuity_stream: int
    text: str
    tokens: int
    char_start: int
    char_end: int
    bbox: tuple[float, float, float, float]


def _block_segments(page, block):
    bkey = block["block_key"]
    segs = [s for s in page.get("normalization_map", []) if s.get("block_key") == bkey]
    if segs:
        return segs
    nr = block.get("normalized_range")
    if nr:
        return [{
            "block_key": bkey,
            "mode": "range",
            "normalized_start": nr[0],
            "normalized_end": nr[1],
            "source_start": block["char_start"],
            "source_end": block["char_end"],
        }]
    return []


def _map_norm_to_source(segments, page_norm_start, sub_start, sub_end, fallback_start, fallback_end):
    """Map [sub_start, sub_end) relative to block normalized text back to source_text offsets."""
    abs_a = page_norm_start + sub_start
    abs_b = page_norm_start + sub_end
    starts, ends = [], []
    for seg in segments:
        sa, sb = seg["normalized_start"], seg["normalized_end"]
        if sb <= abs_a or sa >= abs_b:
            continue
        if seg.get("mode") == "identity":
            starts.append(seg["source_start"] + max(0, abs_a - sa))
            ends.append(seg["source_start"] + min(sb - sa, abs_b - sa))
        else:
            starts.append(seg["source_start"])
            ends.append(seg["source_end"])
    if not starts or not ends:
        return fallback_start, fallback_end
    a, b = min(starts), max(ends)
    return (a, b) if a < b else (fallback_start, fallback_end)


def _sub_bbox(block, source_text, char_start, char_end):
    """Compute tight union bbox from block lines overlapping [char_start, char_end), falling back to block bbox."""
    lines = block.get("lines") or []
    if not lines or (char_start == block["char_start"] and char_end == block["char_end"]):
        return tuple(float(v) for v in block["bbox"])
    cursor = block["char_start"]
    matched = []
    for line in lines:
        ltext = line.get("text", "")
        idx = source_text.find(ltext, cursor, block["char_end"])
        if idx == -1:
            continue
        lend = idx + len(ltext)
        cursor = lend
        if lend > char_start and idx < char_end and valid_box(line.get("bbox")):
            matched.append(line["bbox"])
    if not matched:
        return tuple(float(v) for v in block["bbox"])
    return (
        min(b[0] for b in matched),
        min(b[1] for b in matched),
        max(b[2] for b in matched),
        max(b[3] for b in matched),
    )


def _split_sentences_safe(text):
    """Split text into sentence slices (start, end) without breaking abbreviations or decimals."""
    slices = []
    start = 0
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch == "\n":
            j = i
            while j < n and text[j] == "\n":
                j += 1
            if start < i:
                slices.append((start, i))
            start = j
            i = j
            continue
        if ch in ".!?؟" and i + 1 < n and text[i + 1].isspace():
            if ch == "." and i > 0 and text[i - 1].isdigit() and i + 2 < n and text[i + 2].isdigit():
                i += 1
                continue
            prefix = text[max(0, i - 16):i + 1]
            if ch == "." and _ABBREV_RE.search(prefix):
                i += 1
                continue
            end = i + 1
            while end < n and text[end] in "”»\"')]}":
                end += 1
            slices.append((start, end))
            while end < n and text[end].isspace():
                end += 1
            start = end
            i = end
            continue
        i += 1
    if start < n:
        tail = text[start:n].strip()
        if tail:
            left_pad = len(text[start:n]) - len(text[start:n].lstrip())
            right_pad = len(text[start:n]) - len(text[start:n].rstrip())
            slices.append((start + left_pad, n - right_pad))
    return slices or [(0, n)]


def _split_words_safe(text, counter, max_tokens, overlap_tokens=0):
    """Split text into <=max_tokens slices at safe word boundaries preserving quantity-unit and negation links."""
    words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
    if not words:
        return [(0, len(text))]

    # Merge tokens that must stay together (quantity+unit, negation+next word).
    groups = []
    idx = 0
    while idx < len(words):
        a, b, tok = words[idx]
        while idx + 1 < len(words):
            _, nb, ntok = words[idx + 1]
            if (_QUANTITY_LEFT_RE.search(tok) and _UNIT_RIGHT_RE.match(ntok)) or _NEGATION_LEFT_RE.search(tok):
                idx += 1
                b = nb
                tok = text[a:b]
            else:
                break
        groups.append((a, b))
        idx += 1

    pieces = []
    gi = 0
    while gi < len(groups):
        start_char = groups[gi][0]
        best_gj = gi
        gj = gi
        while gj < len(groups):
            cand = text[start_char:groups[gj][1]]
            if counter.count(cand) <= max_tokens:
                best_gj = gj
                gj += 1
            else:
                break
        if best_gj == gi and counter.count(text[start_char:groups[gi][1]]) > max_tokens:
            # Single oversized group: split using actual tokenizer offsets.
            ga, gb = groups[gi]
            tok_offs = counter.encode_offsets(text[ga:gb])
            step = max(1, max_tokens - overlap_tokens) if overlap_tokens < max_tokens else max_tokens
            pos = 0
            while pos < len(tok_offs):
                end_pos = min(len(tok_offs), pos + max_tokens)
                pieces.append((ga + tok_offs[pos][0], ga + tok_offs[end_pos - 1][1]))
                if end_pos == len(tok_offs):
                    break
                pos += step
            gi += 1
            continue

        pieces.append((start_char, groups[best_gj][1]))
        if best_gj == len(groups) - 1:
            break
        if overlap_tokens > 0:
            back = best_gj
            while back > gi and counter.count(text[groups[back][0]:groups[best_gj][1]]) < overlap_tokens:
                back -= 1
            gi = max(gi + 1, back + (1 if back == gi else 0))
        else:
            gi = best_gj + 1
    return pieces


_TRAILING_SENTENCE_RE = re.compile(r"[.;,!?]\s*$")


def _is_structural_heading(block, next_block=None):
    """Distinguish true section headings from inline bold/emphasized prose fragments."""
    if block.get("kind") != "heading":
        return False
    text = " ".join(block.get("text", "").split())
    if not text:
        return False
    if len(text) > 140 or len(text.split()) > 18:
        return False
    if text[0].islower() or text[0] in "([{\"'«":
        return False
    if _TRAILING_SENTENCE_RE.search(text):
        return False
    if next_block and next_block.get("kind") == "heading":
        ntext = " ".join(next_block.get("text", "").split())
        if ntext and (ntext[0].islower() or ntext[0] in "([{\"'«"):
            return False
    return True


def _extract_prose_units(pages, sections_by_key, counter, *, max_unit_tokens, overlap_tokens=0):
    """Extract ordered, continuity-tracked ProseUnits from accepted blocks only."""
    units = []
    stream_id = 0
    prev_page_num = None
    prev_ended_clean = False
    active_section_key = None

    for page in sorted(pages, key=lambda p: p["physical_page"]):
        phys = page["physical_page"]
        if prev_page_num is None or phys != prev_page_num + 1 or not prev_ended_clean:
            stream_id += 1
            active_section_key = None
        prev_page_num = phys

        if page.get("status") in IGNORED_PAGE_STATUSES:
            prev_ended_clean = False
            active_section_key = None
            continue

        blocks = page.get("blocks", [])
        norm_text = page.get("normalized_text", "")
        source_text = page.get("source_text", "")

        accepted_indices = []
        for idx, block in enumerate(blocks):
            nr = block.get("normalized_range")
            if (
                block.get("status") == "extracted"
                and block.get("kind") in ACCEPTED_KINDS
                and nr is not None
                and block.get("continuity_group") is not None
                and 0 <= nr[0] < nr[1] <= len(norm_text)
                and 0 <= block.get("char_start", -1) < block.get("char_end", -1) <= len(source_text)
                and valid_box(block.get("bbox"))
            ):
                accepted_indices.append(idx)

        if not accepted_indices:
            prev_ended_clean = False
            active_section_key = None
            continue

        # Check leading or trailing review barriers on the page.
        first_acc = accepted_indices[0]
        last_acc = accepted_indices[-1]
        has_leading_review = any(blocks[i].get("status") == "needs_review" for i in range(first_acc))
        has_trailing_review = any(blocks[i].get("status") == "needs_review" for i in range(last_acc + 1, len(blocks)))
        if has_leading_review or blocks[first_acc]["continuity_group"] > 1:
            stream_id += 1
            active_section_key = None

        prev_cg = blocks[first_acc]["continuity_group"]
        for pos_idx, idx in enumerate(accepted_indices):
            block = blocks[idx]
            next_acc = blocks[accepted_indices[pos_idx + 1]] if pos_idx + 1 < len(accepted_indices) else None
            if block["continuity_group"] != prev_cg:
                stream_id += 1
                prev_cg = block["continuity_group"]
                active_section_key = None

            nr_start, nr_end = block["normalized_range"]
            b_norm = norm_text[nr_start:nr_end]
            if not b_norm.strip():
                continue

            if block["kind"] == "heading":
                if _is_structural_heading(block, next_acc):
                    eff_kind = "heading"
                    active_section_key = block.get("section_key") or active_section_key
                else:
                    eff_kind = "paragraph"
                    if active_section_key is None:
                        active_section_key = block.get("section_key")
            else:
                eff_kind = block["kind"]
                if block.get("section_key") and active_section_key is None:
                    active_section_key = block.get("section_key")

            sec_key = active_section_key
            sec_meta = sections_by_key.get(sec_key, {}) if sec_key else {}
            segs = _block_segments(page, block)
            btok = counter.count(b_norm)
            if btok <= max_unit_tokens:
                sub_ranges = [(0, len(b_norm))]
            else:
                sub_ranges = []
                # Pack sentences up to max_unit_tokens; split oversized sentences safely.
                sent_slices = _split_sentences_safe(b_norm)
                cur_a, cur_b = None, None
                for sa, sb in sent_slices:
                    stext = b_norm[sa:sb]
                    if counter.count(stext) > max_unit_tokens:
                        if cur_a is not None:
                            sub_ranges.append((cur_a, cur_b))
                            cur_a, cur_b = None, None
                        for wa, wb in _split_words_safe(stext, counter, max_unit_tokens, overlap_tokens):
                            sub_ranges.append((sa + wa, sa + wb))
                        continue
                    if cur_a is None:
                        cur_a, cur_b = sa, sb
                    elif counter.count(b_norm[cur_a:sb]) <= max_unit_tokens:
                        cur_b = sb
                    else:
                        sub_ranges.append((cur_a, cur_b))
                        cur_a, cur_b = sa, sb
                if cur_a is not None:
                    sub_ranges.append((cur_a, cur_b))

            for sub_a, sub_b in sub_ranges:
                piece = b_norm[sub_a:sub_b].strip()
                if not piece:
                    continue
                cs, ce = _map_norm_to_source(segs, nr_start, sub_a, sub_b, block["char_start"], block["char_end"])
                bbox = _sub_bbox(block, source_text, cs, ce)
                units.append(ProseUnit(
                    asset_key=page["asset_key"],
                    record_id=page["record_id"],
                    asset_sha256=page["asset_sha256"],
                    source_path=page["source_path"],
                    revision_key=page.get("revision_key"),
                    language=(page.get("language") or "und").lower(),
                    physical_page=phys,
                    printed_page_label=page.get("printed_page_label"),
                    block_key=block["block_key"],
                    kind=eff_kind,
                    section_key=sec_key,
                    section_label=sec_meta.get("section_label"),
                    heading=sec_meta.get("heading"),
                    parent_section_key=sec_meta.get("parent_section_key"),
                    continuity_stream=stream_id,
                    text=piece,
                    tokens=counter.count(piece),
                    char_start=cs,
                    char_end=ce,
                    bbox=bbox,
                ))

        prev_ended_clean = not has_trailing_review
    return units


def _build_chunk_record(units, ordinal, strategy, config_fp, counter):
    """Construct a PostgreSQL-compatible chunk + citation spans record from ProseUnits."""
    retrieval_text = "\n\n".join(u.text for u in units)
    token_count = counter.count(retrieval_text)
    first = units[0]
    spans = []
    for span_no, u in enumerate(units, 1):
        spans.append({
            "span_no": span_no,
            "physical_page": u.physical_page,
            "printed_page_label": u.printed_page_label,
            "char_start": u.char_start,
            "char_end": u.char_end,
            "source_anchor": u.block_key,
            "bbox": [round(v, 6) for v in u.bbox],
            "table_cells": None,
            "section_key": u.section_key,
        })
    section_keys = [u.section_key for u in units if u.section_key]
    primary_sec = next((u for u in units if u.section_key), first)
    chunk_uid = fingerprint({
        "asset_key": first.asset_key,
        "asset_sha256": first.asset_sha256,
        "strategy": strategy,
        "config_fingerprint": config_fp,
        "ordinal": ordinal,
        "retrieval_text": retrieval_text,
        "spans": [(s["physical_page"], s["char_start"], s["char_end"], s["source_anchor"]) for s in spans],
    })
    return {
        "chunk_id": chunk_uid,
        "ordinal": ordinal,
        "asset_key": first.asset_key,
        "record_id": first.record_id,
        "asset_sha256": first.asset_sha256,
        "source_path": first.source_path,
        "revision_key": first.revision_key,
        "language_tag": first.language,
        "strategy": strategy,
        "token_count": token_count,
        "char_count": len(retrieval_text),
        "section_key": primary_sec.section_key,
        "section_label": primary_sec.section_label,
        "heading": primary_sec.heading,
        "parent_section_key": primary_sec.parent_section_key,
        "section_keys": list(dict.fromkeys(section_keys)),
        "table_id": None,
        "applicability_note": None,
        "continuity_stream": first.continuity_stream,
        "physical_pages": sorted({u.physical_page for u in units}),
        "retrieval_text": retrieval_text,
        "retrieval_eligible": False,
        "spans": spans,
    }


def _chunk_fixed(units, counter, *, max_tokens, overlap_tokens, strategy, config_fp, start_ordinal=0):
    """Fixed-size token windowing with overlap inside each continuity stream."""
    if not units:
        return []
    chunks = []
    by_stream = defaultdict(list)
    for u in units:
        by_stream[u.continuity_stream].append(u)

    ordinal = start_ordinal
    for stream_id in sorted(by_stream):
        stream_units = by_stream[stream_id]
        i = 0
        while i < len(stream_units):
            j = i
            picked = []
            while j < len(stream_units):
                cand_text = "\n\n".join([*(p.text for p in picked), stream_units[j].text])
                if counter.count(cand_text) <= max_tokens:
                    picked.append(stream_units[j])
                    j += 1
                else:
                    break
            if not picked:
                picked = [stream_units[i]]
                j = i + 1
            ordinal += 1
            chunks.append(_build_chunk_record(picked, ordinal, strategy, config_fp, counter))
            if j >= len(stream_units):
                break
            if overlap_tokens > 0:
                back = j - 1
                while back > i and counter.count("\n\n".join(u.text for u in stream_units[back:j])) < overlap_tokens:
                    back -= 1
                i = max(i + 1, back + (1 if back == i else 0))
            else:
                i = j
    return chunks


def _chunk_section_aware(units, counter, *, min_tokens=300, target_tokens=380, max_tokens=450, config_fp, start_ordinal=0):
    """Section- and paragraph-aware chunking respecting continuity streams and clinical context."""
    if not units:
        return []
    groups = []
    cur_key = None
    cur_group = []
    for u in units:
        gkey = (u.continuity_stream, u.section_key)
        if gkey != cur_key:
            if cur_group:
                groups.append(cur_group)
            cur_key = gkey
            cur_group = [u]
        else:
            cur_group.append(u)
    if cur_group:
        groups.append(cur_group)

    chunk_buffers = []
    pending_headings = []
    for idx, group in enumerate(groups):
        if all(u.kind == "heading" for u in group):
            if idx + 1 < len(groups) and groups[idx + 1][0].continuity_stream == group[0].continuity_stream:
                pending_headings.extend(group)
            continue
        if pending_headings and pending_headings[0].continuity_stream == group[0].continuity_stream:
            combined_head = [*pending_headings, *group]
            if counter.count("\n\n".join(u.text for u in combined_head[:2])) <= max_tokens:
                group = combined_head
        pending_headings = []
        section_buffers = []
        buf = []
        for u in group:
            if not buf:
                buf.append(u)
                continue
            cur_text = "\n\n".join(item.text for item in buf)
            cur_tok = counter.count(cur_text)
            cand_tok = counter.count(cur_text + "\n\n" + u.text)
            starts_with_heading = len(buf) == 1 and buf[0].kind == "heading"
            if cand_tok <= max_tokens and (starts_with_heading or cur_tok < min_tokens or cand_tok <= target_tokens):
                buf.append(u)
            else:
                if not all(item.kind == "heading" for item in buf):
                    section_buffers.append(buf)
                buf = [u]
        if buf:
            if section_buffers:
                prev_text = "\n\n".join(item.text for item in section_buffers[-1])
                tail_text = "\n\n".join(item.text for item in buf)
                if counter.count(tail_text) < 80 and counter.count(prev_text + "\n\n" + tail_text) <= max_tokens:
                    section_buffers[-1].extend(buf)
                else:
                    section_buffers.append(buf)
            else:
                section_buffers.append(buf)
        chunk_buffers.extend(section_buffers)

    return [
        _build_chunk_record(buf, start_ordinal + idx, "section_aware", config_fp, counter)
        for idx, buf in enumerate(chunk_buffers, 1)
    ]


STRATEGIES = {
    "fixed_256_32": {"max_tokens": 256, "overlap_tokens": 32},
    "fixed_512_64": {"max_tokens": 512, "overlap_tokens": 64},
    "section_aware": {"min_tokens": 300, "target_tokens": 380, "max_tokens": 450, "overlap_tokens": 0},
}


def chunk_pages(
    pages,
    *,
    sections=None,
    documents=None,
    strategy="section_aware",
    tokenizer_path=None,
    counter=None,
    min_tokens=300,
    target_tokens=380,
    max_tokens=450,
    overlap_tokens=0,
):
    """Chunk extracted pages into deterministic citation-backed chunks and report unchunkable documents."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unsupported chunking strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
    tok = counter or TokenCounter(tokenizer_path)
    params = dict(STRATEGIES[strategy])
    if strategy == "section_aware":
        params.update(min_tokens=min_tokens, target_tokens=target_tokens, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    cfg_fp = fingerprint({"strategy": strategy, "params": params, "tokenizer": tok.fingerprint})

    sections_by_key = {s["section_key"]: s for s in (sections or []) if s.get("section_key")}
    pages_by_asset = defaultdict(list)
    for page in pages:
        pages_by_asset[page["asset_key"]].append(page)

    known_assets = {}
    for doc in (documents or []):
        known_assets[doc["asset_key"]] = doc
    for asset_key, asset_pages in pages_by_asset.items():
        if asset_key not in known_assets:
            known_assets[asset_key] = {
                "asset_key": asset_key,
                "record_id": asset_pages[0]["record_id"],
                "asset_sha256": asset_pages[0]["asset_sha256"],
                "path": asset_pages[0]["source_path"],
            }

    all_chunks = []
    empty_documents = []
    unchunkable_documents = []
    asset_summaries = []

    for asset_key in sorted(known_assets):
        doc_info = known_assets[asset_key]
        asset_pages = sorted(pages_by_asset.get(asset_key, []), key=lambda p: p["physical_page"])
        if not asset_pages:
            empty_documents.append({
                "asset_key": asset_key,
                "record_id": doc_info.get("record_id"),
                "reason": "no_extracted_pages",
            })
            continue

        units = _extract_prose_units(
            asset_pages,
            sections_by_key,
            tok,
            max_unit_tokens=params["max_tokens"],
            overlap_tokens=params["overlap_tokens"],
        )
        if strategy == "section_aware":
            chunks = _chunk_section_aware(
                units,
                tok,
                min_tokens=params["min_tokens"],
                target_tokens=params["target_tokens"],
                max_tokens=params["max_tokens"],
                config_fp=cfg_fp,
                start_ordinal=len(all_chunks),
            )
        else:
            chunks = _chunk_fixed(
                units,
                tok,
                max_tokens=params["max_tokens"],
                overlap_tokens=params["overlap_tokens"],
                strategy=strategy,
                config_fp=cfg_fp,
                start_ordinal=len(all_chunks),
            )

        if not chunks:
            statuses = {p.get("status") for p in asset_pages}
            total_norm = sum(len(p.get("normalized_text", "")) for p in asset_pages)
            reason = "no_normalized_prose_all_pages_held_or_blank" if total_norm == 0 else "no_accepted_prose_units"
            unchunkable_documents.append({
                "asset_key": asset_key,
                "record_id": doc_info.get("record_id"),
                "pages": len(asset_pages),
                "page_statuses": sorted(statuses),
                "reason": reason,
            })
        else:
            all_chunks.extend(chunks)

        asset_summaries.append({
            "asset_key": asset_key,
            "record_id": doc_info.get("record_id"),
            "pages": len(asset_pages),
            "prose_units": len(units),
            "chunks": len(chunks),
            "total_tokens": sum(c["token_count"] for c in chunks),
        })

    summary = {
        "strategy": strategy,
        "params": params,
        "tokenizer": tok.name,
        "tokenizer_fingerprint": tok.fingerprint,
        "config_fingerprint": cfg_fp,
        "documents_total": len(known_assets),
        "documents_chunked": sum(1 for a in asset_summaries if a["chunks"] > 0),
        "empty_documents_count": len(empty_documents),
        "unchunkable_documents_count": len(unchunkable_documents),
        "empty_documents": empty_documents,
        "unchunkable_documents": unchunkable_documents,
        "total_chunks": len(all_chunks),
        "total_spans": sum(len(c["spans"]) for c in all_chunks),
        "total_tokens": sum(c["token_count"] for c in all_chunks),
        "assets": asset_summaries,
    }
    return all_chunks, summary


def validate_chunk_citations(chunks, pages):
    """Verify every chunk has valid, non-empty PostgreSQL-compatible offsets and bounding boxes."""
    page_map = {(p["asset_key"], p["physical_page"]): p for p in pages}
    issues = []
    for chunk in chunks:
        if chunk["ordinal"] <= 0:
            issues.append((chunk["chunk_id"], "invalid_ordinal"))
        if not chunk["retrieval_text"].strip():
            issues.append((chunk["chunk_id"], "empty_retrieval_text"))
        if chunk["language_tag"] != chunk["language_tag"].lower():
            issues.append((chunk["chunk_id"], "non_lowercase_language"))
        if not chunk["spans"]:
            issues.append((chunk["chunk_id"], "missing_spans"))
        for expected_no, span in enumerate(chunk["spans"], 1):
            if span["span_no"] != expected_no:
                issues.append((chunk["chunk_id"], "non_sequential_span_no"))
            page = page_map.get((chunk["asset_key"], span["physical_page"]))
            if not page:
                issues.append((chunk["chunk_id"], "missing_source_page"))
                continue
            src = page["source_text"]
            a, b = span["char_start"], span["char_end"]
            if a is None or b is None or a < 0 or b <= a or b > len(src):
                issues.append((chunk["chunk_id"], "invalid_source_offsets"))
            elif not src[a:b].strip():
                issues.append((chunk["chunk_id"], "whitespace_only_cited_slice"))
            if not valid_box(span["bbox"]):
                issues.append((chunk["chunk_id"], "invalid_bbox"))
            if span["table_cells"] is not None and chunk["table_id"] is None:
                issues.append((chunk["chunk_id"], "table_cells_without_table_id"))
    return issues


def add_command(commands):
    command = commands.add_parser("chunk", help="Chunk accepted extracted prose with citation spans.")
    command.add_argument("--input", type=Path, required=True,
                         help="Extraction run directory (containing pages.jsonl) or explicit pages.jsonl file.")
    command.add_argument("--sections", type=Path, help="Optional sections.jsonl (defaults to <input>/sections.jsonl).")
    command.add_argument("--documents", type=Path, help="Optional documents.json (defaults to <input>/documents.json).")
    command.add_argument("--output", type=Path, required=True, help="Output directory for chunks.jsonl and summary.")
    command.add_argument("--strategy", choices=sorted(STRATEGIES), default="section_aware")
    command.add_argument("--tokenizer", type=Path, help="Path to HuggingFace tokenizer.json.")
    command.add_argument("--min-tokens", type=int, default=300)
    command.add_argument("--target-tokens", type=int, default=380)
    command.add_argument("--max-tokens", type=int, default=450)
    return command


def main(args):
    input_path = args.input.resolve()
    pages_path = input_path / "pages.jsonl" if input_path.is_dir() else input_path
    base_dir = input_path if input_path.is_dir() else input_path.parent
    sections_path = args.sections or (base_dir / "sections.jsonl")
    docs_path = args.documents or (base_dir / "documents.json")

    pages = read_jsonl(pages_path)
    sections = read_jsonl(sections_path) if sections_path.is_file() else []
    documents = json.loads(docs_path.read_text(encoding="utf-8")).get("assets", []) if docs_path.is_file() else []

    chunks, summary = chunk_pages(
        pages,
        sections=sections,
        documents=documents,
        strategy=args.strategy,
        tokenizer_path=args.tokenizer,
        min_tokens=args.min_tokens,
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
    )
    citation_issues = validate_chunk_citations(chunks, pages)
    if citation_issues:
        raise ValueError(f"Citation validation failed with {len(citation_issues)} issues: {citation_issues[:3]}")

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "chunks.jsonl", chunks)
    # Also emit flattened chunk_spans.jsonl for direct inspection / future PostgreSQL staging.
    span_rows = [
        {"chunk_id": c["chunk_id"], "asset_key": c["asset_key"], **s}
        for c in chunks for s in c["spans"]
    ]
    write_jsonl(out_dir / "chunk_spans.jsonl", span_rows)
    summary["citation_validation_errors"] = 0
    (out_dir / "chunk_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in (
        "strategy", "documents_total", "documents_chunked",
        "empty_documents_count", "unchunkable_documents_count",
        "total_chunks", "total_spans", "total_tokens", "citation_validation_errors",
    )}, ensure_ascii=False, sort_keys=True))
    return 0
