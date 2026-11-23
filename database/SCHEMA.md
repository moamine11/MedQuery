# Medical RAG schema

The `medical_rag` PostgreSQL schema uses pgvector from `public`. It contains 22 application tables and a `schema_migrations` ledger. Apply the numbered migrations in order to the existing Compose service.

```sh
cat database/migrations/[0-9]*.sql | docker compose exec -T postgres psql -X -U medical_user -d medical_rag -v ON_ERROR_STOP=1
```

Each migration holds an advisory lock, runs in a transaction, and records its version on success. Replaying the applied files skips their changes.

## Publication and provenance

- `sources`, `documents`, and `document_identifiers` establish issuer and stable publication identity. `categories` and `document_categories` tag documents with one or more of the five seeded categories. Independently released languages and jurisdictional adaptations have separate document identities. Use lowercase BCP 47 language tags and uppercase ISO 3166-1 alpha-2 country codes. The database checks their basic shape, and callers validate registry membership and canonicalize input.
- `document_versions` records publisher revisions and replacement links. `version_countries`, `medical_topics`, `version_topics`, and `version_applicability` record geographic and clinical scope. Structured conditions within a population rule apply together. A verified exclusion overrides an inclusion. Unknown scope supplies no positive match. Use the accompanying source wording where structured fields do not capture the criteria.
- `document_assets` records the storage URI and SHA-256 of each representation, plus its own license, retrieval permission, and external embedding permission. `extraction_assets` lists every input asset for an extraction. The applicable permission must be allowed on the version and on every manifest asset. An alternate excluded from the manifest has no effect. Review `extraction_permissions` before transmitting content to an external embedding provider. `ingestion_attempts` preserves acquisition URLs and outcomes, even when identity resolution fails.
- Resolve a verified source identifier before creating a revision. Reuse an identical asset within a version. Use the hash as evidence when reconciling possible matches across documents. Persist a new issued revision for verified content changes and route ambiguous matches to review. A retry uses the same work key with a new attempt number.

## Extraction and citation

`extractions` identifies a run by its version, input manifest fingerprint, pipeline version, and configuration fingerprint. `extracted_pages`, `sections`, `extracted_tables`, `chunks`, and `chunk_spans` hold its output. A completed extraction is immutable and every chunk must have a citation span. Physical pages are 1-based. Printed labels retain source numbering. Unpaginated sources use one synthetic page and can carry an original anchor. Text offsets are 0-based, end-exclusive character positions in stored page text. Bounding boxes are normalized `[x0,y0,x1,y1]` values from the top-left of the page. Table grids use JSON objects and span cell coordinates use JSON arrays.

`embedding_models` identifies provider, model revision, input profile, dimensions, distance, and external processing. `chunk_embeddings` has one vector per chunk and model. Begin with exact search and a single serving model per query. Add a model-specific approximate index after measuring corpus size and recall with eligibility filters.

## Review and retrieval

`clinical_reviews` stores append-only decisions against a completed extraction and its version's clinical context revision. Changes to clinical metadata invalidate earlier approvals. Use `publish_extraction(extraction_id, model_id)` to check approval, effective dates, source permissions, and embedding coverage before switching the current version and selected extraction in one transaction.

Use `retrievable_chunks`, `retrievable_embeddings`, and `retrievable_citations` for current content. These views check the effective window, latest review, and permissions at read time. Effective end dates are exclusive and date checks use the session time zone. Apply request-specific language, country, and population filtering in retrieval queries. Use separate access for historical review. Superseded, withdrawn, expired, and revoked guidance remains available for audit and leaves current retrieval without fallback.
