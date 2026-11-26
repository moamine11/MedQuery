# MedQuery — Medical RAG Ingestion & Storage Foundation

MedQuery is a local-first multilingual (`fr`, `en`, `ar`) ingestion, extraction, chunking, embedding, and PostgreSQL/pgvector storage pipeline for clinical guidelines, treatment protocols, and regulatory directives. Every retrieved passage preserves immutable source text, code-point citation spans, normalized bounding boxes, and explicit staging/publication safety gates.

> **Database status:** Database V1 (`medical_rag`) is populated and verified (`96` documents, `98` PDF assets, `6,884` physical pages, `14,610` chunks, `104,385` citation spans, `14,610` `384`-dim embeddings; `medical_rag_pilot` remains isolated at `8` documents / `104` chunks). All `96` versions remain `status = 'staged'` (`use_status = 'unknown'`, `clinical_reviews = 0`), so authoritative `retrievable_*` views remain gated at `0` rows; research queries use `research_retrieve(..., allow_research_only=True)`. See [Teammate Database Setup](docs/ingestion/TEAMMATE_DATABASE_SETUP.md).

---

## Architecture

1. **Intake & Cross-Collection Reconciliation (`ingestion/intake.py`, `adapters.py`, `metadata.py`, `reconciliation.py`):** Confines PDF reads, validates canonical JSONL / CSV / collection metadata against [docs/ingestion/metadata.schema.json](docs/ingestion/metadata.schema.json), computes SHA-256 digests and physical page counts, enforces holds/exclusions, and reconciles cross-collection duplicates.
2. **Review-Gated Extraction & Normalization (`ingestion/extraction/`):** Uses PyMuPDF, Poppler, Tesseract (`ara`/`fra`/`eng`), Docling (Heron + TableFormer Fast), and PaddleOCR (Arabic mobile models) inside isolated per-page subprocesses with atomic page checkpoints (`--resume`). Tables, figures, and uncertain OCR blocks are quarantined in `needs_review` outside normalized prose.
3. **Citation-Aware Chunking (`ingestion/chunking.py`):** Applies `section_aware` chunking (`min=300`, `target=380`, `max=450`, `overlap=0` tokens using the `intfloat/multilingual-e5-small` tokenizer), preserving continuity streams, review barriers, and `[char_start, char_end)` + `[0, 1]` bounding-box citation spans.
4. **Local CPU/CUDA Embeddings (`ingestion/embeddings.py`):** Encodes passages locally with `intfloat/multilingual-e5-small` (`rev 614241f622f53c4eeff9890bdc4f31cfecc418b3`, `384` dims, cosine, mean pooling, L2-normalized, `passage: ` / `query: ` prefixes, non-truncating length validation, and SHA-256 checkpoint resume). Supports `--device auto|cpu|cuda` with automatic CUDA OOM batch backoff and CPU fallback.
5. **PostgreSQL + pgvector Storage (`database/`, `ingestion/storage.py`):** Persists document identities, versions, portable `medquery://` asset URIs, extractions, pages, sections, held tables, chunks, citation spans, and embeddings transactionally inside `medical_rag` (`001_core.sql`, `002_publication.sql`). Authoritative `retrievable_*` views remain gated until clinical approval (`clinical_reviews`) and `publish_extraction()`.

---

## Repository Structure

```text
.
├── compose.yaml                     # Local PostgreSQL 17 + pgvector Docker service
├── database/
│   ├── SCHEMA.md                    # Database schema and retrieval/publication rules
│   ├── connection.py                # Lazy psycopg connection helper
│   └── migrations/                  # 001_core.sql, 002_publication.sql
├── ingestion/
│   ├── __main__.py                  # CLI: inspect, validate, reconcile, extract, chunk, embed, load
│   ├── adapters.py, intake.py, metadata.py, pipeline.py, reconciliation.py, treatment_protocols.py
│   ├── chunking.py, embeddings.py, storage.py, transfer.py
│   └── extraction/                  # Page worker, routing, normalization, profiles (CPU & HPC)
├── experiments/
│   ├── extraction/                  # Runtime Docling/Paddle/Poppler adapters, model download scripts, evals
│   └── retrieval/                   # Phase 6 retrieval evaluation script
├── notebooks/
│   └── full_ingestion_hpc.ipynb     # Resumable Kubeflow JupyterLab full-corpus ingestion notebook
├── docs/ingestion/                  # Architecture, metadata/storage specs, schema, HPC instructions
├── tests/                           # Automated unit and rollback-only PostgreSQL integration tests
├── data/                            # Git-ignored local workspace (raw/, ingestion/, docs/)
├── .env.example                     # Template for local PostgreSQL environment variables
├── requirements.txt                 # Local CPU Python dependencies
└── requirements-hpc.txt             # HPC dependencies (preserves container PyTorch 2.5.1+cu124)
```

---

## Local Setup & Dataset Placement

### 1. Place Authorized Datasets Locally

Store raw collections strictly under the Git-ignored `data/raw/` directory (never commit PDFs or `.env` files):

- `data/raw/1.Clinical Protocols & Official Ministry of Health + Official Ministry of Health Directives...`
- `data/raw/2.Treatment Protocols...`
- `data/raw/therapeutic_guidelines_data...`

### 2. Install Local Dependencies (Python 3.11+)

System tools required for extraction: Poppler (`pdftotext`, `pdfinfo`), Tesseract (`eng`, `fra`, `ara`), and optional isolated `docling-venv` / `paddle-venv` environments (see [experiments/extraction/README.md](experiments/extraction/README.md)).

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu
.venv/bin/python -m pip install -r requirements.txt -r experiments/extraction/requirements.txt
```

### 3. Initialize PostgreSQL 17 + pgvector

```sh
cp .env.example .env   # Edit POSTGRES_PASSWORD in .env
docker compose up -d --wait
cat database/migrations/[0-9]*.sql | docker compose exec -T postgres psql -X -U medical_user -d medical_rag -v ON_ERROR_STOP=1
```

Compose exposes PostgreSQL on `127.0.0.1:5433` by default. See [database/SCHEMA.md](database/SCHEMA.md) for schema details.

---

## Running the Ingestion Pipeline

All CLI subcommands run from the repository root under `python -m ingestion`:

```sh
# 1. Inspect, validate, and reconcile metadata
.venv/bin/python -m ingestion inspect --root <collection_root> --collection treatment_protocols --dry-run
.venv/bin/python -m ingestion validate --root <collection_root> --collection treatment_protocols --manifest manifest.jsonl --output data/ingestion/val_out

# 2. Extract using the reconciled selection manifest (resumable)
.venv/bin/python -m ingestion extract --selection-manifest data/ingestion/phase5/extraction_selection_manifest.json --document <ID> --output data/ingestion/ext_out --workers 2 --resume

# 3. Chunk, embed, dry-run validate, and load into PostgreSQL
.venv/bin/python -m ingestion chunk --input data/ingestion/ext_out --output data/ingestion/chk_out
.venv/bin/python -m ingestion embed --input data/ingestion/chk_out --output data/ingestion/emb_out --model intfloat/multilingual-e5-small --device auto

set -a && . ./.env && set +a
.venv/bin/python -m ingestion load \
  --manifest data/ingestion/phase5/extraction_selection_manifest.json \
  --extract-dir data/ingestion/ext_out \
  --chunk-dir data/ingestion/chk_out \
  --embed-dir data/ingestion/emb_out --dry-run
```

For full-scale GPU/HPC execution via Kubeflow JupyterLab, see [notebooks/full_ingestion_hpc.ipynb](notebooks/full_ingestion_hpc.ipynb) and [docs/ingestion/PHASE_9_HPC_INSTRUCTIONS.md](docs/ingestion/PHASE_9_HPC_INSTRUCTIONS.md).

---

## Automated Tests

Database tests require Docker PostgreSQL running and `.env` sourced (every database test runs inside a rolled-back transaction):

```sh
set -a && . ./.env && set +a
.venv/bin/python -m unittest discover -s tests -v
```

---

## Branch Strategy & Collaboration

- `main`: Stable, validated integrated system (PR-only merges after integration testing).
- `nlp-dev`: NLP, database, ingestion, retrieval, LLM, and evaluation development.
- `web-dev`: Web development (`web/frontend/` and `web/backend/` are reserved for the web team).

Contributors branch off their team's development branch and merge via Pull Requests. See [CONTRIBUTING.md](CONTRIBUTING.md) for ownership and shared-contract rules.
