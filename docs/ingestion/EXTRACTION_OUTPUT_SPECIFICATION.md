# MedQuery extraction output specification

**Contract v1** · Implemented by the Phase 4 local `extract` command. This is the handoff for review and later processing. It uses the existing SQL schema as a future persistence target. [CLI documentation](../../README.md#local-pdf-extraction) · [PostgreSQL schema](../../database/SCHEMA.md)

## Files and identity

| File | Fields and invariants |
| --- | --- |
| `identity.json` | Contract and pipeline versions plus input and configuration SHA-256 fingerprints. Input includes asset hashes, roles, selected physical pages and exclusions. Configuration includes parser code and versions, model checksums, and content-affecting profile settings and limits. |
| `pages.jsonl` | Selected source text, normalized text and map, block order, source geometry, tables, figures, retained engine candidates and decisions. Each row has an `asset_key`, `asset_sha256`, `record_id`, `physical_page` and source path. |
| `sections.jsonl` | Section key, ordinal, optional numeric label and parent, start/end physical page and ordered block anchors. Sections open on heading candidates. A gap in the selected physical pages resets inferred continuity. |
| `documents.json` | Source asset key, record ID, SHA-256, path, observed physical page count, processed subset, status counts, diagnostics and exclusions. |
| `review_queue.jsonl` | Stable review key, affected scope and key, diagnostic code, available evidence and original asset, physical page and region. Repeated block-level structure and order flags remain in their blocks when the owning structure or page is already queued. |
| `summary.json`, `attempts.json` | Run counts, tool identities and checksums, profile, per-page timings, failure and reuse outcomes and worker RSS. Timing is specific to each invocation. |
| `checkpoints/` | Per-page extraction keys and checksummed pre-assembly payloads. Successful pages commit atomically and can be reused on matching source hashes and extraction settings. |

Every page belongs to the exact validated source asset. The row carries acquisition and revision fields when they exist in the validated inventory, plus document and asset reuse states. The selected subset may omit other pages or companion assets. Final extraction manifests for a future persistence step must account for required corrections and supplements and for every selected page that carries source evidence.

## Source text and geometry

`physical_page` is 1-based in the original PDF and independent of numbering printed on the page. `printed_page_label` is a nullable marginal-text candidate with a bounding box and evidence status. `pdf_page_label` is the separately recorded PDF PageLabels value. A visually unconfirmed candidate carries `visually_verified: false`. Verify it against the source before treating it as a printed label.

`source_text` records the chosen parser's page evidence in the selected order, with two newline code points between blocks. It can include candidate text under review. Each block holds its exact source slice, line boxes, kind, candidate engine, direction, language, reading ordinal, diagnostic status and offset range. Raw independent engine results remain in `candidates`. Block offsets count Unicode code points from zero and exclude the end. The selected source slice must match `source_text[char_start:char_end]`. Context, dose and spelling errors in a candidate stay available for source review. PostgreSQL-incompatible NULs must be handled separately in a future persistence phase.

`bbox` is `[x0,y0,x1,y1]` on the displayed PDF CropBox after PDF rotation. The top left is the origin, x increases rightward and y increases downward. Geometry-bearing blocks and review regions require finite coordinates and `0 <= x0 < x1 <= 1`, `0 <= y0 < y1 <= 1`. Page evidence carries the CropBox and MediaBox in points, displayed dimensions, PDF rotation and native-to-display matrix. OCR boxes are translated from the actual crop origin and pixel scale. PaddleOCR candidates provide line polygons reduced to boxes. Poppler fallback checks displayed dimensions before its geometry can be selected. A failed page can have a page-wide review anchor without verified page dimensions.

## Normalized text and review

`normalized_text` is assembled from technically extracted paragraph, heading and list-item blocks. Source text stays available for audit, including excluded items. Every output code point belongs to a `normalization_map` entry. An `identity` entry maps equal substrings one-to-one. A `range` entry maps the normalized span to the full original range after NFC composition or whitespace collapse. A discretionary soft-hyphen repair preserves the surrounding source ranges separately. Page-block offsets and bounding boxes enable cross-checking either representation. Newline separators map to the original block separators.

A block has a `normalized_range` and a `continuity_group` only when included. Changes in continuity group mark gaps across unresolved content. Keep these boundaries when reading normalized text. Caption, footnote, table, figure, uncertain and confirmed repeated margin blocks retain source evidence outside normalized prose. Repeated margin detection uses exact short text across at least three selected pages and preserves numeric text, detected clinical tokens and recommendation language. NFC leaves Arabic base letters, marks, digits, Latin runs, visible hyphens, superscripts and dosage signs intact.

Page and block diagnostics distinguish blank or uniform graphical pages, ambiguous low detail, missing text, extraction errors, suspicious text layers, ambiguous order and critical disagreement. Review is scoped where the original page provides bounded evidence. A page can simultaneously have components in review and independently extracted regions. All emitted pages and assets have `retrieval_eligible: false` pending the existing metadata, clinical and publication process.

## Tables and figures

Each table keeps a page-fragment anchor, caption candidate, `grid.candidates` from the relevant engines, and footnote candidates. Candidate grids preserve row and column indexes, detected cell boxes, roles and reported spans. Candidate PyMuPDF spans can also carry geometric evidence and candidate header keys. An unknown relationship remains explicitly unverified. The optional possible predecessor field records nearby same-asset evidence from adjacent selected pages. It leaves `linked_fragments` empty pending verification.

Each figure keeps the original page-region anchor, candidate source bounds, captions and readable labels with candidate provenance. Labels can come from a regional OCR pass or selected page text. Visible arrows and conditions require independent source verification. The Phase 4 figure edge arrays are empty and trigger review. Render either the displayed page or source crop from the exact PDF bytes and stored geometry.

## Future SQL mapping

| Contract data | Existing destination |
| --- | --- |
| Versioned input and configuration | `extractions` and `extraction_assets` once a complete, authorized input manifest is staged. |
| Immutable page source | `extracted_pages` records the exact asset, physical page index, verified printed label and selected source text. |
| Sections and heading hierarchy | `sections` records verified hierarchy, parent-before-child ordinal, heading, label, derived text and source page IDs in the same asset and extraction. |
| Structured tables | `extracted_tables` records verified captions and relationships in the grid JSON object with physical start/end page IDs. |
| Block geometry, normalized maps, candidate evidence and figure graphs | Versioned local artifacts linked to the run, asset and physical page. |
| Future citations | `chunk_spans` can use the physical page FK, code-point offsets, source label or box and optional table-cell coordinates. |

The database completion trigger also requires chunks and source spans. See the [core migration](../../database/migrations/001_core.sql) and [publication migration](../../database/migrations/002_publication.sql).
