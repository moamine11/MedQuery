# Storage Input/Output Mapping (Phase 7)

This document specifies how validated ingestion, extraction, chunking, and embedding artifacts map into the existing `medical_rag` PostgreSQL + pgvector schema (`001_core.sql`, `002_publication.sql`) via [storage.py](../../ingestion/storage.py) (`python -m ingestion load`).

## Insertion Dependency Order & Field Mapping

| Step | Source Artifact & Fields | Target Table | Mapping & Compatibility Rules |
| :--- | :--- | :--- | :--- |
| 1 | `inventory.jsonl`: `metadata.issuer.{key,name,kind,country,homepage}` | `sources` | Uses explicit `issuer.key` when present or deterministic `src:<sha256[:16]>` over normalized `(name, kind, country)`. Upserted on `source_key`. |
| 2 | `inventory.jsonl`: `metadata.{title,canonical_title,language}`, `identity_key` | `documents` | Uses `identity_key` when present or deterministic `rec:<collection>:<record_id>`; lowercases BCP 47 `language_tag`. Sets `translation_of_id` only for `state: "verified"` relations. |
| 3 | `inventory.jsonl`: `metadata.identifiers[]` | `document_identifiers` | Inserts only `verified: true`, `scope: "document"` identifiers (excluding `as_recorded`). |
| 4 | `inventory.jsonl` + `extraction_selection_manifest.json`: `categories`, `target_categories` | `document_categories` | Unions collection and cross-collection `target_categories` (preserving multi-collection membership for deduplicated HAS guidelines). |
| 5 | `inventory.jsonl`: `revision_key`, `edition`, `title`, `official_url`, `publication_date`, `effective_*`, `applicability`, `reuse` | `document_versions` | Uses `revision_key` or `rev:<sha256([record_id, primary_sha256])[:16]>`. Maps only full `YYYY-MM-DD` dates to SQL `date` (`YYYY` / `YYYY-MM` preserved in `edition_label` and `source_metadata`). Always starts `status='staged'` with `use_status='unknown'` unless explicitly cleared. |
| 6 | `metadata.applicability.{countries,rules}`, `metadata.topics[]` | `version_countries`, `medical_topics`, `version_topics`, `version_applicability` | Maps `country_list` ISO alpha-2 codes, half-open `int4range` (`[min, max)`), and topic vocabulary/code/label pairs. |
| 7 | `manifest.eligible_assets[]` + `metadata.assets[]` | `document_assets` | Converts 64-hex `sha256` to 32-byte `sha256_digest`. Stores portable `medquery://<collection>/<relative_path>` URIs (no host-specific absolute paths). Keeps `primary` and `annex` (`correction`/`supplement`) assets together under the parent `document_version`. |
| 8 | `embedding_model.json` | `embedding_models` | Registers local model metadata (`intfloat/multilingual-e5-small`, rev `614241f...`, `384` dims, `cosine`, `is_external=false`). Rejects external models. |
| 9 | Per-version grouped `pages.jsonl`, `sections.jsonl`, `chunks.jsonl`, `chunk_embeddings.jsonl` | `extractions`, `extraction_assets`, `extracted_pages`, `sections`, `extracted_tables`, `chunks`, `chunk_spans`, `chunk_embeddings` | Creates one `extraction` per `(version_id, pipeline_name, pipeline_version, input_fingerprint, config_fingerprint)`. Rejects NUL (`\x00`) bytes without altering citation offsets. Inserts sections parent-before-child with derived `text_content`. Stores held tables in `extracted_tables` (`chunks.table_id` stays `NULL`). Re-indexes `chunks.ordinal` `1..N` per extraction and validates offsets, `bbox`, dimensions, and `input_sha256`. Transitions `extractions.status` to `'completed'` only when chunks and spans exist. |
| 10 | Extended metadata & `review_queue.jsonl` | `ingestion_attempts` | Records `document_type`, `authority`, `phase2_classification`, `content_languages`, raw partial dates, candidate `relations`, review counts/codes, and metadata digest in `source_metadata`. |

## Idempotency, Recovery & Safety Separation

- **Idempotency:** Matching completed `extractions` rows are reused without violating `protect_extraction_input` triggers; `chunk_embeddings` uses `ON CONFLICT (chunk_id, model_id) DO NOTHING`.
- **Crash recovery:** Incomplete `extractions` stuck in `status='staged'` are cleaned via `_clear_staged_extraction()` inside the load transaction before re-inserting and completing.
- **Clinical vs. research separation:** Loaded records remain `status='staged'` with `use_status='unknown'` and zero `clinical_reviews`, so `retrievable_chunks`, `retrievable_embeddings`, and `retrievable_citations` return 0 rows. `research_retrieve(..., allow_research_only=True)` provides an explicit, clearly labeled research-only exact cosine query that excludes `educational` records by default.
