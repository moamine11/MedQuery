# Contributing to MedQuery

## Branch Strategy

MedQuery uses three long-lived branches:

- **`main`**: Stable, validated integrated system. Merged only after cross-team integration testing succeeds. Direct pushes are discouraged and blocked by GitHub branch protection.
- **`nlp-dev`**: Active development branch for the NLP & data engineering team (database, ingestion, extraction, chunking, embeddings, retrieval, LLM, and evaluation).
- **`web-dev`**: Active development branch for the web team (frontend, web backend, and user interface).

## Workflow for Individual Contributors

1. Sync your team's development branch (`nlp-dev` or `web-dev`):
   ```sh
   git fetch origin
   git checkout nlp-dev   # or web-dev
   git pull origin nlp-dev
   ```
2. Create a short-lived feature branch from your team's branch:
   ```sh
   git checkout -b nlp/<short-topic>   # or web/<short-topic>
   ```
   Upcoming NLP feature branches off `nlp-dev`:
   - `nlp/query-processing` — Sara
   - `nlp/retrieval` — Douaa
   - `nlp/llm-experiments` — Amine
3. Commit focused changes, run the relevant automated tests locally, push your feature branch, and open a **Pull Request** targeting your team's branch (`nlp-dev` or `web-dev`).
4. Require at least one teammate review before merging into `nlp-dev` or `web-dev`.
5. Merge into `main` only via Pull Request after end-to-end integration tests pass across both teams.

## Directory Ownership & Shared Contracts

| Area | Owner | Notes |
| --- | --- | --- |
| `ingestion/`, `database/`, `experiments/`, `notebooks/`, `docs/ingestion/`, `tests/` | **NLP team (`nlp-dev`)** | Existing ingestion and database module locations stay fixed. |
| `web/frontend/`, `web/backend/` | **Web team (`web-dev`)** | Reserved paths for frontend and web backend code. |
| `compose.yaml`, `database/migrations/`, `database/SCHEMA.md`, `database/connection.py`, `requirements*.txt`, `.env.example`, `.gitignore`, API contracts | **Shared (coordinated)** | Changes require agreement and review from both teams before merging. |

## Repository Hygiene

Never commit `.env`, credentials, raw PDFs (`data/raw/`), generated pipeline outputs (`data/ingestion/`), model weights, transfer archives (`*.zip`), virtual environments, or database dumps.
