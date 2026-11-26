# Proposed ingestion architecture

Run a local Python CLI with PostgreSQL and a Git-ignored workspace per machine. Put category-specific logic in adapters and use one validated data model, one extraction interface and one persistence layer. The following interfaces and CLI are a design for review.

| Stage | Inputs and outputs |
| --- | --- |
| Discover and harmonize | Resolve an explicit collection root and manifest/adapter. Enumerate selected local assets, hash and probe read-only files, validate paths and joins, emit canonical candidates and issues. Support symlink containment checks and bounded file sizes, page counts, workers and command timeouts. |
| Validate and reconcile | Check JSON Schema, registered codes, calendar/range values, one primary asset, unique stable keys, explicit source evidence, version and translation relations, and asset-manifest permissions. Compare canonical issuer/identifier, language and revision before using hashes, titles and page similarity to propose merges. Unresolved conflicts enter a review queue. |
| Extract by page or region | Produce immutable source-page text, layout blocks, language hints, labels and coordinates. Detect substantial text, bad glyphs, scan overlays, blank pages and mixed content. Compare native text, page OCR and region OCR candidates. Use parsers behind interchangeable interfaces with pinned versions and capture diagnostics. |
| Reconstruct layout | Preserve column and RTL reading order, headings, numbered clauses, table headers, merged cells, continuations, units, footnotes and source geometry. Keep diagrams as located figures with original labels and reviewed branch/edge evidence in local structured artifacts. Unresolved arrows or table values hold that material from retrieval. |
| Normalize and chunk | Retain immutable `source_text` and keep a traceable mapping from normalized text to source offsets or geometric anchors. Preserve code points for offsets, Arabic meaning, decimal separators, negation, dosage units and special characters. Compare heading, article, procedure and table-aware chunks against a baseline. Keep clinical qualifiers, table header/value links and flowchart conditions with their local context. |
| Assess quality | Emit per-page and document diagnostics, flag missing/garbled sections and assess language, reading order, citations, table relationships and clinically sensitive tokens. Block advancement when a high-risk value or reference cannot be verified. Retain reviewed clinical scope/evidence with diagnostic status. |
| Embed and store | Use the existing `database.connection.connect()` and qualified `medical_rag.` names. Persist the extraction graph and citations, complete it transactionally, then batch embeddings by a registered model revision and input profile. Reuse matching chunk/model embeddings with matching input hash. Test local multilingual models and add external candidates only with permission for the version and every input asset. |
| Review and publish | Review follows completion against the current clinical context revision. Publish through the existing SQL function with a complete serving-model embedding set and verified permissions and dates. Retrieval uses the eligible views plus request-specific language, jurisdiction, population and authority policy. |

## PostgreSQL and local boundaries

`001_core.sql` supplies issuer/document/version identity, asset manifests, citation, embedding and attempt constraints. `002_publication.sql` enforces immutable asset identity and completed extraction data, review revision checks and transactional publication. The six existing integration tests cover connectivity and pgvector, foreign keys, duplicate identity and current version uniqueness, acquisition-backed citation, embeddings, and retrieval gating. The connection utility uses PG environment overrides and the Compose defaults. Phase 1 uses local analysis only.

Create the input manifest from explicitly selected assets of one version, including mandatory correction material. Each stored page belongs to a manifest asset. Keep physical pages 1-based and preserve printed labels independently. Where useful, preserve legal article or diagram labels as source anchors. Use 0-based, end-exclusive offsets into exact stored page text and normalized top-left-origin bounding boxes. Store table grids as JSON objects and cell coordinates as JSON arrays. Use an anchor, valid character range or box for every citation span. At least one chunk and a valid span for every chunk are required to complete an extraction. Flowchart structure and parser diagnostics live in the local extraction artifacts. Citations can point to original page regions using existing fields.

Keep normalized artifacts and versioned policy snapshots locally under an ignored run root. For SQL citations retain acquisition URLs separately from official landing URLs, actual asset hash, pages and labels. Keep both publisher attribution and distribution route when evidence supports them. Strip credentials from URLs and diagnostics. Do not send external embedding inputs when permission is unknown; consult `extraction_permissions` immediately before dispatch and record the decision.

The application must enforce the published scope. Use `retrievable_chunks`, `retrievable_embeddings` and `retrievable_citations` for the current view and add filters for the requested language, geographic and population scope, and reviewed authority/type. Keep educational material outside clinical retrieval and offer separately authorized use if needed. Keep legal or administrative text in the appropriate evidence lane and display foreign jurisdiction clearly. Superseded, expired, withdrawn or revoked material stays available for audit without implicit fallback. The SQL effective-date checks use the session timezone and treat the end date as exclusive.

## Repeatability and recovery

- Use deterministic work keys per stage and scope and canonical, ordered fingerprints for the input manifest and parser configuration. Include pipeline version and configuration. Record page routing, OCR languages, normalization options and renderer/configuration versions in stage artifacts. Keep runs against the same inputs comparable.
- Hash while reading, keep raw files read-only, and recheck bytes just before commit or use a verified immutable local copy. Preserve multiple acquisition attempts without duplicating document identities or adding the same hash twice within a version. Route ambiguous cross-version or cross-collection matches to review.
- Write stage artifacts under a run key with atomic rename and checksums. Checkpoint long extraction by asset/page and retain independent results when one PDF fails. Record attempts as running, succeeded, failed, skipped or needs review with stable reason codes. On restart, reconcile durable outputs, expire abandoned attempts under a per-work-key lock and allocate the next attempt number.
- Use short transactions and per-identity or extraction locks when reconciling, retrying and selecting work. Load and validate the extraction graph atomically before setting completed; do expensive parsing outside the transaction. Retries reuse the same extraction business key if inputs/configuration are unchanged. Completed output is immutable. New pipeline settings or input changes create another extraction and require their own review. Only the publication action changes the selected/current state.

## Proposed CLI

```sh
python -m ingestion inspect --collection therapeutic_guidelines --root /local/therapeutic_guidelines --adapter therapeutic-guidelines
python -m ingestion validate --collection treatment_protocols --root /local/treatment_protocols --manifest manifest.jsonl
python -m ingestion ingest --collection treatment_protocols --root /local/treatment_protocols --manifest manifest.jsonl --dry-run
python -m ingestion ingest --collection treatment_protocols --root /local/treatment_protocols --manifest manifest.jsonl --resume
python -m ingestion status --run-id <run-id>
```

Use `--manifest` for canonical JSONL and `--adapter` for supported legacy inputs. Keep inspect, validate and dry-run read-only with respect to PostgreSQL. Ingest stages reviewed inputs. Publication is a separate authorized action. Support a local pilot subset by record/asset ID and a bounded worker option. Emit machine-readable outcomes and use a nonzero exit when the requested run still has blockers or failures.
