"""Run bounded, isolated page experiments. Outputs and failures are resumable per page."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

from prepare import CONFIG, REPO, local_output, write_json

MODES = ("pymupdf", "pymupdf_sorted", "pymupdf_tables", "poppler", "poppler_layout",
         "tesseract_full", "selective_page", "selective_region", "selective_refined", "reference_regions", "docling", "paddle_arabic")
SCRIPT = Path(__file__).resolve()
THREADS = ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def setup_paths(out):
    tesseract_root = out / "setup/tesseract/usr"
    return {
        "tesseract": Path(os.environ.get("PHASE3_TESSERACT", tesseract_root / "bin/tesseract")),
        "models": Path(os.environ.get("PHASE3_TESSDATA", tesseract_root / "share/tesseract-ocr/5/tessdata")),
        "docling_python": Path(os.environ.get("PHASE3_DOCLING_PYTHON", out / "setup/docling-venv/bin/python")),
        "docling_models": Path(os.environ.get("PHASE3_DOCLING_MODELS", out / "setup/docling_models")),
        "paddle_arabic_python": Path(os.environ.get("PHASE3_PADDLE_PYTHON", out / "setup/paddle-venv/bin/python")),
        "paddle_arabic_models": Path(os.environ.get("PHASE3_PADDLE_MODELS", out / "setup/paddle_models")),
    }


def worker(mode, info, config, paths):
    import pymupdf

    from engines import hybrid, native, poppler, result, tess

    path = (REPO / config["collection_root"] / info["path"]).resolve()
    if mode == "docling":
        from docling_adapter import extract
        return extract(path, info, paths["docling_models"], config)
    if mode == "paddle_arabic":
        from paddle_adapter import extract
        return extract(path, info, paths["paddle_arabic_models"], config)
    if mode.startswith("poppler"):
        return poppler(path, info["physical_page"], mode, config["timeout_seconds"])
    with pymupdf.open(path) as pdf:
        page = pdf[info["physical_page"] - 1]
        if mode == "tesseract_full":
            return tess(page, config, paths["tesseract"], paths["models"], info["language"])
        if mode == "reference_regions":
            baseline = native(page, config)
            crops = [tess(page, config, paths["tesseract"], paths["models"], info["language"], box)
                     for box in info["oracle_regions"]]
            regions = [line for crop in crops for line in crop["lines"]]
            return result(regions, diagnostics={
                "action": "region_ocr", "regions": info["oracle_regions"], "reference_guided": True,
                "baseline_native_words": sum(map(len, baseline["lines"])),
            })
        if mode.startswith("selective"):
            return hybrid(page, config, paths["tesseract"], paths["models"], info["language"],
                          allow_regions=mode != "selective_page", refined=mode == "selective_refined")
        return native(page, config, sorted_text=mode == "pymupdf_sorted", structure=mode == "pymupdf_tables")


def run_command(mode, info, config, paths, log):
    python = str(paths.get(f"{mode}_python", sys.executable))
    command = ["/usr/bin/time", "-q", "-f", "%M", "-o", str(log), python, str(SCRIPT), "--worker", mode]
    env = {**os.environ, **dict.fromkeys(THREADS, str(config["ocr_workers"])),
           "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "LC_ALL": "C.UTF-8",
           "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"}
    started = perf_counter()
    try:
        timeout = config["structured_timeout_seconds"] if mode == "docling" else config["timeout_seconds"]
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 start_new_session=True, env=env)
        try:
            stdout, stderr = child.communicate(json.dumps(info).encode(), timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, 9)
            child.communicate()
            return {"status": "timeout", "elapsed_seconds": perf_counter() - started, "peak_rss_kib": None}
        elapsed = perf_counter() - started
        peak = int(log.read_text().strip()) if log.exists() and log.read_text().strip().isdigit() else None
        if child.returncode:
            return {"status": "failed", "elapsed_seconds": elapsed, "peak_rss_kib": peak,
                    "stderr": stderr.decode(errors="replace")[-6000:], "exit_code": child.returncode}
        try:
            payload = json.loads(stdout)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {"status": "failed", "elapsed_seconds": elapsed, "peak_rss_kib": peak,
                    "stderr": "Worker returned invalid JSON. " + stderr.decode(errors="replace")[-6000:]}
    finally:
        log.unlink(missing_ok=True)
    status = "partial" if payload["diagnostics"].get("status") == "partial_success" else "ok"
    return {"status": status, "elapsed_seconds": elapsed, "peak_rss_kib": peak,
            "output": payload, "stderr": stderr.decode(errors="replace")[-6000:]}


def version(command):
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=15)
        output = (process.stdout + process.stderr).splitlines()
        return output[0] if process.returncode == 0 and output else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES[:-3]))
    parser.add_argument("--split", choices=("development", "held_out", "all"), default="all")
    parser.add_argument("--run", default="phase3-final", help="Run directory name under the ignored experiment directory")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    out = local_output(REPO / config["working_dir"])
    paths = setup_paths(out)
    if args.worker:
        info = json.load(sys.stdin)
        # Some libraries print progress to stdout. Keep the worker JSON stream clean.
        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            payload = worker(args.worker, info, config, paths)
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        return

    if Path(args.run).name != args.run or args.run in {"", ".", "..", "setup", "references", "review_renders"}:
        parser.error("--run must be a single new or existing run directory name")
    run_dir = local_output(out / "runs" / args.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    pages = json.loads(manifest.read_text())["pages"]
    versions = {
        "python": platform.python_version(), "platform": platform.platform(),
        "pymupdf": version([sys.executable, "-c", "import pymupdf; print(pymupdf.__version__)"]),
        "poppler": version(["pdftotext", "-v"]),
        "tesseract": version([paths["tesseract"], "--version"]),
        "tessdata_sha256": {p.name: file_hash(p) for p in sorted(paths["models"].glob("*.traineddata"))},
    }
    script_inputs = [CONFIG, manifest, SCRIPT, SCRIPT.with_name("engines.py")]
    identity = {path.name: file_hash(path) for path in script_inputs if path.exists()}
    corpus_root = (REPO / config["collection_root"]).resolve()
    verified = set()
    for page in pages:
        relative = page["path"]
        if relative in verified:
            continue
        source = (corpus_root / relative).resolve()
        if not source.is_relative_to(corpus_root) or file_hash(source) != page["sha256"]:
            raise ValueError(f"Source is unconfined or has changed: {page['asset_id']}")
        verified.add(relative)
    for mode in args.modes:
        directory = local_output(run_dir / mode)
        directory.mkdir(exist_ok=True)
        environment, inputs, regions = dict(versions), dict(identity), {}
        if mode in {"docling", "paddle_arabic"}:
            packages = ["docling", "torch", "docling-parse", "docling-ibm-models"] if mode == "docling" else [
                "paddleocr", "paddlepaddle", "paddlex",
            ]
            adapter = SCRIPT.with_name("docling_adapter.py" if mode == "docling" else "paddle_adapter.py")
            inputs[adapter.name] = file_hash(adapter)
            model_dir = paths[f"{mode}_models"]
            environment["packages"] = {
                package: version([
                    paths[f"{mode}_python"], "-c",
                    f"import importlib.metadata as m; print(m.version({package!r}))",
                ]) for package in packages
            }
            environment["model_sha256"] = {
                p.relative_to(model_dir).as_posix(): file_hash(p)
                for p in sorted(model_dir.rglob("*")) if p.is_file()
            }
        if mode == "reference_regions":
            reference = out / "references/verified.json"
            inputs["reference"] = file_hash(reference)
            regions = {item["key"]: item["oracle_regions"] for item in json.loads(reference.read_text())["pages"]
                       if item.get("oracle_regions")}
        snapshot = {"input_sha256": inputs, "environment": environment, "configuration": config}
        context = directory / "run.json"
        if context.exists() and json.loads(context.read_text()) != snapshot:
            raise ValueError(f"Configuration or environment changed for {mode}. Use a new --run.")
        write_json(context, snapshot)
        for page in pages:
            if mode == "paddle_arabic" and page["language"] != "ar":
                continue
            if mode == "reference_regions":
                if page["key"] not in regions:
                    continue
                page = {**page, "oracle_regions": regions[page["key"]]}
            if args.split != "all" and page["split"] != args.split:
                continue
            destination = directory / f"{page['key']}.json"
            if destination.exists() and (not args.retry_failures or json.loads(destination.read_text())["status"] == "ok"):
                continue
            measurement = run_command(mode, page, config, paths, destination.with_suffix(".rss"))
            write_json(destination, {"page": page, "mode": mode, **measurement})
            print(f"{mode} {page['key']} {measurement['status']} {measurement['elapsed_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
