# MedQuery — Teammate Setup Guide for Database V1

This guide explains how NLP and Web teammates set up and query the shared **Database V1 (`medical_rag`)** research baseline locally on **Linux/macOS** or **Windows**.

---

## 1. Sources & Governance Boundaries (Read Before Sharing)

Keep these four artifact sources strictly separate:

1. **Code & schema (from GitHub):** Pull from `nlp-dev` or `web-dev` (see [CONTRIBUTING.md](../../CONTRIBUTING.md)). Contains `ingestion/`, `database/`, `tests/`, `compose.yaml`, `requirements*.txt`, and `.env.example`. Contains zero PDFs, embeddings, or credentials.
2. **Research database backup (obtained privately from the team):** `backups/medical_rag_v1.dump` (`45.91 MiB` / `48,143,605` bytes; SHA-256 `6c002b37c4cab9fcf1b0e60c5792659a2d03a03937c8cf1254b219fb95e502b6`). Share **strictly within the authorized MedQuery academic research team**; never commit or publish externally. Database derivatives inherit source rights restrictions and are not automatically authorized for redistribution.
3. **Original PDFs (obtained only through authorized sources):** Database V1 indexes `98` eligible PDFs (`203` held/excluded PDFs are intentionally absent). Only `40` explicitly open/academic-licensed PDFs (`open_or_public_reuse`, `CC BY-NC-SA 3.0 IGO`, and unmodified `CC BY-NC-ND 4.0`) are cleared for unmodified team sharing; `58` `use_status = 'unknown'` PDFs require publisher acquisition or rights clearance, and `203` excluded/held PDFs must never be distributed.
4. **Research retrieval vs. clinical serving:** All `96` document versions in Database V1 remain `status = 'staged'` with `use_status = 'unknown'` and `clinical_reviews = 0`. Authoritative clinical views (`retrievable_chunks`, `retrievable_embeddings`, `retrievable_citations`) intentionally return **`0` rows**. For NLP experiments, use `research_retrieve(..., allow_research_only=True)` in [ingestion/storage.py](../../ingestion/storage.py).

---

## 2. Required Software & Pinned Model

- **Docker + Docker Compose** running image `pgvector/pgvector:pg17` (verified on **PostgreSQL `17.12`** and **pgvector `0.8.7`**; see [compose.yaml](../../compose.yaml))
- **Python `3.11+`** with packages from [requirements.txt](../../requirements.txt)
- **Pinned embedding model:** `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3` (`384` dims, `cosine`, `e5_passage_v1`, `query: ` / `passage: ` prefixes, mean pooling, L2-normalized, `max_seq_length=512`)

---

## 3. Configure `.env` (Never Commit `.env`)

Copy [`.env.example`](../../.env.example) to `.env` and set a local password:

- **Linux/macOS (`bash`/`zsh`):**
  ```sh
  cp .env.example .env
  # Edit .env and set POSTGRES_PASSWORD=<your_local_password>
  ```
- **Windows (PowerShell):**
  ```powershell
  Copy-Item .env.example .env
  # Edit .env and set POSTGRES_PASSWORD=<your_local_password>
  ```

Connection defaults used by [database/connection.py](../../database/connection.py): `PGHOST=127.0.0.1`, `PGPORT=5433`, `PGDATABASE=medical_rag`, `PGUSER=medical_user`, `POSTGRES_PASSWORD`.

---

## 4. Restore Database V1 into a Fresh Docker Environment

> [!IMPORTANT]
> **Do NOT run `database/migrations/[0-9]*.sql` before `pg_restore`.**
> `backups/medical_rag_v1.dump` is a full schema-and-data `pg_dump -Fc` archive that already includes `CREATE EXTENSION vector`, the `medical_rag` schema, `schema_migrations`, triggers, views, and all rows. On a **fresh** `docker compose up -d --wait`, `medical_rag` is empty and restores cleanly with zero conflicts.

### Linux / macOS (`bash` / `zsh`)

```sh
# 1. Verify backup checksum
(cd backups && sha256sum -c medical_rag_v1.dump.sha256)

# 2. Start fresh PostgreSQL 17 + pgvector container
docker compose up -d --wait

# 3. Restore directly into the empty medical_rag database
docker compose cp backups/medical_rag_v1.dump postgres:/tmp/medical_rag_v1.dump
docker compose exec postgres pg_restore -U medical_user -d medical_rag --no-owner --no-privileges /tmp/medical_rag_v1.dump
docker compose exec postgres rm -f /tmp/medical_rag_v1.dump
```

### Windows (PowerShell)

```powershell
# 1. Verify backup checksum
(Get-FileHash backups\medical_rag_v1.dump -Algorithm SHA256).Hash.ToLower()
# Expected: 6c002b37c4cab9fcf1b0e60c5792659a2d03a03937c8cf1254b219fb95e502b6

# 2. Start fresh PostgreSQL 17 + pgvector container
docker compose up -d --wait

# 3. Copy dump into container and restore (avoids PowerShell binary stdin issues)
docker compose cp backups/medical_rag_v1.dump postgres:/tmp/medical_rag_v1.dump
docker compose exec postgres pg_restore -U medical_user -d medical_rag --no-owner --no-privileges /tmp/medical_rag_v1.dump
docker compose exec postgres rm -f /tmp/medical_rag_v1.dump
```

*(If your local `medical_rag` database was previously initialized with migrations or old data, reset it cleanly before step 3 by running:*
`docker compose exec postgres psql -X -U medical_user -d postgres -v ON_ERROR_STOP=1 -c "DROP DATABASE IF EXISTS medical_rag; CREATE DATABASE medical_rag OWNER medical_user;"` *)*

### Verify Restored Row Counts (Linux & Windows)

```sh
docker compose exec postgres psql -X -U medical_user -d medical_rag -c "
  SELECT 'documents' AS tbl, count(*) FROM medical_rag.documents
  UNION ALL SELECT 'document_assets', count(*) FROM medical_rag.document_assets
  UNION ALL SELECT 'extracted_pages', count(*) FROM medical_rag.extracted_pages
  UNION ALL SELECT 'chunks', count(*) FROM medical_rag.chunks
  UNION ALL SELECT 'chunk_embeddings', count(*) FROM medical_rag.chunk_embeddings
  UNION ALL SELECT 'retrievable_chunks', count(*) FROM medical_rag.retrievable_chunks;
"
```

Expected counts: `documents = 96`, `document_assets = 98`, `extracted_pages = 6,884`, `chunks = 14,610`, `chunk_embeddings = 14,610`, `retrievable_chunks = 0`.

---

## 5. Portable PDF Paths & Automated Tests

Asset URIs use `medquery://<collection>/<asset_path>`. Place authorized PDFs under Git-ignored `data/raw/` or point `MEDQUERY_COLLECTION_ROOTS` (JSON) to your local collection directories:

- **Linux/macOS:**
  ```sh
  export MEDQUERY_COLLECTION_ROOTS='{"clinical_protocols_ministry_directives":"/path/to/col1","treatment_protocols":"/path/to/col2","therapeutic_guidelines":"/path/to/col3"}'
  set -a && . ./.env && set +a
  .venv/bin/python -m unittest discover -s tests -v
  ```
- **Windows (PowerShell):**
  ```powershell
  $env:MEDQUERY_COLLECTION_ROOTS = '{"clinical_protocols_ministry_directives":"C:/path/to/col1","treatment_protocols":"C:/path/to/col2","therapeutic_guidelines":"C:/path/to/col3"}'
  Get-Content .env | ForEach-Object { if ($_ -match '^([^#=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2]) } }
  .\.venv\Scripts\python -m unittest discover -s tests -v
  ```
