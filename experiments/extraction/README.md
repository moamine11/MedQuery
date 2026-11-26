# Phase 3 extraction benchmark

Run from the repository root. Preparation joins the existing 23-asset pilot to Phase 2's validated inventory, excludes holds, verifies eligible asset hashes and renders only the selected physical pages. The benchmark reads local PDFs and writes under the ignored `data/ingestion/phase3/` directory. The verified annotations there are required for scoring and the separately labeled reference-guided crop experiment.

## Baseline setup

The measured environment was Ubuntu 24.04 x86-64, Python 3.12 and Poppler `pdftotext` 24.02.0. The three model language packs are from Ubuntu's `1:4.1.0-2` packages and the Tesseract binary is 5.3.4. On Ubuntu 24.04 the local setup below uses existing shared libraries as listed by `ldd`. System package installation is unnecessary when those libraries are available. GNU `/usr/bin/time` supplies the RSS measurement.

```bash
phase3=data/ingestion/phase3
mkdir -p "$phase3/setup"
python3 -m venv "$phase3/setup/venv"
"$phase3/setup/venv/bin/python" -m pip install -r experiments/extraction/requirements.txt

(
    cd "$phase3/setup"
    apt-get download tesseract-ocr=5.3.4-1build5 tesseract-ocr-fra=1:4.1.0-2 \
        tesseract-ocr-ara=1:4.1.0-2 tesseract-ocr-eng=1:4.1.0-2
    for package in tesseract-ocr*.deb; do dpkg-deb -x "$package" tesseract; done
)
```

Use `PHASE3_TESSERACT` and `PHASE3_TESSDATA` to point to another local binary or model directory. Verify model languages with `tesseract --tessdata-dir DIR --list-langs`. Each run captures version and SHA-256 evidence.

## Optional CPU engines

Install these into distinct local environments. The lock files record the tested Linux/Python 3.12 resolution. The companion requirements files record the direct experiment pins. The two download scripts fetch only public model artifacts at pinned revisions. Run the benchmarks offline with the exported package code and local artifacts.

```bash
phase3=data/ingestion/phase3
python3 -m venv "$phase3/setup/docling-venv"
"$phase3/setup/docling-venv/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchvision==0.23.0+cpu
"$phase3/setup/docling-venv/bin/python" -m pip install -r experiments/extraction/docling-linux-cpu.lock.txt
bash experiments/extraction/download_docling_models.sh

python3 -m venv "$phase3/setup/paddle-venv"
"$phase3/setup/paddle-venv/bin/python" -m pip install -r experiments/extraction/paddle-linux-cpu.lock.txt
bash experiments/extraction/download_paddle_models.sh
```

If those environments or models already exist locally, set `PHASE3_DOCLING_PYTHON`, `PHASE3_DOCLING_MODELS`, `PHASE3_PADDLE_PYTHON` or `PHASE3_PADDLE_MODELS`. The Docling worker enables native layout and TableFormer Fast with OCR disabled. The separate Paddle run uses just the five Arabic pilot pages and mobile models. Its benchmark geometry is line-level.

## Run and score

```bash
phase3=data/ingestion/phase3
phase3_python="$phase3/setup/venv/bin/python"
"$phase3_python" experiments/extraction/prepare.py
"$phase3_python" experiments/extraction/benchmark.py --run reproduction --modes \
    pymupdf pymupdf_sorted pymupdf_tables poppler poppler_layout tesseract_full \
    selective_page selective_region selective_refined reference_regions docling paddle_arabic
"$phase3_python" experiments/extraction/score.py --run reproduction
```

Omit optional mode names when their pinned dependencies are unavailable. `--split development` or `--split held_out` restricts runs by document. Run into the same output directory with the same inputs to resume existing measurements. `--retry-failures` retries failed, partial and timed-out attempts. A change to the input or code fingerprint requires a new run name. Compare elapsed time from one benchmark invocation at a time on the same machine.

To use the existing verified references, keep the original manifest inputs and copy the Git-ignored references through the team's approved local handoff if needed. The scorer checks the reference manifest hash, source SHA-256, physical page and source-render hash. The crop arm uses manually drawn regions and is scored separately. Never replace `references/verified.json` with parser output. If a different pilot is authorized, create fresh annotations by visual inspection. The Phase 3 reference set covers exact excerpts and selected relationships rather than full-page transcriptions.

The measured outputs and per-engine snapshots are under `data/ingestion/phase3/runs/phase3-final/`. Each run's `summary.json` contains aggregate and subset comparisons. Its `error_analysis.json` contains source locators, reference excerpts, deviations, structure checks and route decisions. Its `review_queue.json` carries original-page locations for unresolved items. [Method and results](../../docs/ingestion/EXTRACTION_EXPERIMENTS.md)
