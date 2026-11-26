# Phase 9 — School HPC Execution & Local Load Instructions

This guide covers the manual upload, resumable HPC execution, download, and local PostgreSQL loading workflow for the **98-asset (6,884-page)** MedQuery full-scale ingestion run.

---

## 1. Files to Upload

Upload these three files from `data/ingestion/phase9/transfer/` and `notebooks/`:

1. `notebooks/full_ingestion_hpc.ipynb`
2. `data/ingestion/phase9/transfer/medquery_hpc_code.zip`
3. `data/ingestion/phase9/transfer/medquery_hpc_input.zip`

Before uploading, verify local transfer checksums if needed:

```sh
(cd data/ingestion/phase9/transfer && sha256sum -c transfer_checksums.sha256)
```

---

## 2. Where to Upload in Kubeflow JupyterLab

1. Start your single allowed GPU notebook instance (`H100 NVL MIG 3g.47gb`, 9.6 CPU cores, 38.4 GiB RAM, 30 GiB persistent `/home/jovyan`).
2. In the JupyterLab file browser (which opens at `/home/jovyan`), upload all three files directly into **`/home/jovyan/`**:
   - `/home/jovyan/full_ingestion_hpc.ipynb`
   - `/home/jovyan/medquery_hpc_code.zip`
   - `/home/jovyan/medquery_hpc_input.zip`

All persistent working directories, virtual environments, Hugging Face caches, checkpoints, and outputs are written under `/home/jovyan/medquery_workspace/` so they survive kernel restarts.

---

## 3. How to Open and Run the Notebook

1. Double-click `full_ingestion_hpc.ipynb` in JupyterLab and select the **Python 3 (ipykernel)** kernel (the container kernel with `PyTorch 2.5.1+cu124`).
2. Run **Cell groups 1–4 sequentially** first:
   - **Group 1 — Environment Verification:** confirms CUDA/H100 visibility, `PyTorch 2.5.1+cu124`, cgroup CPU/RAM limits, persistent disk headroom, and Hugging Face connectivity.
   - **Group 2 — Project Setup:** unpacks `medquery_hpc_code.zip` into `/home/jovyan/medquery_workspace/repo`, installs `requirements-hpc.txt` (preserving `torch 2.5.1+cu124`), verifies/installs Tesseract (`ara`, `eng`, `fra`) + Poppler `pdftotext`, creates isolated `docling-venv` and `paddle-venv` environments, downloads pinned Docling/PaddleOCR models, and downloads pinned `intfloat/multilingual-e5-small` (`rev 614241f622f53c4eeff9890bdc4f31cfecc418b3`) + verifies the Phase 5 `tokenizer.json` SHA-256.
   - **Group 3 — Dataset Preparation:** unpacks `medquery_hpc_input.zip` into `/home/jovyan/medquery_workspace/input`, verifies every file against `manifests/checksums.sha256`, rebases portable collection roots, and verifies all 98 PDFs / 6,884 pages.
   - **Group 4 — Small HPC Smoke Test:** runs 4 representative pages (`FR`, `EN`, `AR`, OCR, Docling, Paddle) through extract → chunk → CUDA embed into isolated `data/ingestion/phase9/smoke_run/` and verifies CPU vs. CUDA numerical compatibility.
3. Only after Group 4 succeeds cleanly, run **Cell groups 5–9**:
   - **Group 5 — Full Extraction (`WORKERS = 4` initially):** runs `python -m ingestion extract` with `profile_hpc.json` and `--resume`. Monitor RAM/CPU; if stable and fast headroom remains, you may increase `WORKERS` up to `6` or `8` (keep `3072 MiB * WORKERS <= 24576 MiB`).
   - **Group 6 — Full Chunking:** runs `section_aware` (`min=300`, `target=380`, `max=450`, `overlap=0`).
   - **Group 7 — Full GPU Embeddings:** runs `intfloat/multilingual-e5-small` on `cuda` (`GPU_BATCH_SIZE = 128`, with automatic OOM halving + CPU fallback).
   - **Group 8 — Final Artifact Verification:** checks asset/page completeness, unique SHA-256s, duplicate/missing chunks, citation integrity, embedding coverage, held-content leakage, and timings.
   - **Group 9 — Package Results:** writes `/home/jovyan/medquery_workspace/medquery_hpc_results.zip` and `medquery_hpc_results.zip.sha256`.

---

## 4. How to Resume Interrupted Processing

If the notebook kernel restarts or the session times out mid-run:

- **Interrupted during Group 5 (Extraction):**
  - Every completed page is already saved atomically under `/home/jovyan/medquery_workspace/repo/data/ingestion/phase9/full_run/extract/checkpoints/*.json`.
  - Simply re-open `full_ingestion_hpc.ipynb` and **re-run the Group 5 cell** directly. It re-configures `HF_HOME` and `PHASE3_*` environment variables automatically and passes `--resume`, reusing all completed page checkpoints.
- **Interrupted during Group 7 (Embeddings):**
  - Every completed batch is flushed and fsynced to `full_run/embed/checkpoints/<fingerprint>.jsonl`.
  - Simply **re-run the Group 7 cell** directly; `embed` automatically resumes from the checkpoint file.
- **Interrupted during Group 2 or 3:**
  - Re-run Groups 2 and 3; model downloads and extracted files skip already-present verified artifacts.

---

## 5. What Outputs to Download Afterwards

From `/home/jovyan/medquery_workspace/` in JupyterLab, download:

1. `medquery_hpc_results.zip`
2. `medquery_hpc_results.zip.sha256`

After downloading to your laptop (e.g., into `data/ingestion/phase9/results/`), verify the archive and unpack it:

```sh
mkdir -p data/ingestion/phase9/results
cd data/ingestion/phase9/results
sha256sum -c medquery_hpc_results.zip.sha256
unzip medquery_hpc_results.zip
sha256sum -c results_checksums.sha256
```

---

## 6. How Outputs Will Be Loaded into Local `medical_rag`

Once `medquery_hpc_results.zip` is unpacked into `data/ingestion/phase9/results/` on your laptop (where PostgreSQL 17 + pgvector and the local `validate_col{1,2,3}` inventories reside):

1. **Dry-run verification first (rolled back, 0 committed rows):**

```sh
set -a && . ./.env && set +a
.venv/bin/python -m ingestion load \
  --manifest data/ingestion/phase5/extraction_selection_manifest.json \
  --extract-dir data/ingestion/phase9/results/extract \
  --chunk-dir data/ingestion/phase9/results/chunk \
  --embed-dir data/ingestion/phase9/results/embed \
  --dry-run
```

2. **Transactional commit into `medical_rag` (only after reviewing `verification_report.json` and the dry-run summary):**

```sh
.venv/bin/python -m ingestion load \
  --manifest data/ingestion/phase5/extraction_selection_manifest.json \
  --extract-dir data/ingestion/phase9/results/extract \
  --chunk-dir data/ingestion/phase9/results/chunk \
  --embed-dir data/ingestion/phase9/results/embed
```

### Optional Local-Processing Fallback

If you later choose to hold specific assets off the HPC (for example, generating a subset with `build_transfer_archives(..., exclude_from_hpc={"TG-0028"})` or `hpc_policy="strict_explicit_license_only"`), `local_fallback_manifest.json` is written under `data/ingestion/phase9/transfer/` for local laptop execution (`python -m ingestion extract --selection-manifest data/ingestion/phase9/transfer/local_fallback_manifest.json ...`, followed by `chunk`, `embed`, and `load` alongside the HPC outputs).

---

## 7. Recorded Active-Run HPC Fixes (Integrated Post-Run)

Observed during live Kubeflow HPC execution and integrated into `ingestion/transfer.py` and `notebooks/full_ingestion_hpc.ipynb`:

- `docs/ingestion/metadata.schema.json` is included in `CODE_INCLUDE_FILES` in `ingestion/transfer.py`.
- Tesseract and Poppler were installed on the HPC via Conda.
- An isolated `embedding-venv` (`--system-site-packages`, preserving container `PyTorch 2.5.1+cu124` without modifying base Conda) resolves the base-container NumPy/SciPy compatibility conflict.
- Verified compatible embedding environment versions on HPC: `NumPy 1.26.4`, `SciPy 1.14.1`, `scikit-learn 1.5.2`.
- Pinned `intfloat/multilingual-e5-small` Safetensors weights (`model.safetensors`) are downloaded and verified cleanly, and all notebook pipeline subprocesses execute via `embedding-venv`.
- CUDA embeddings and CPU/GPU numerical compatibility smoke checks passed successfully.

