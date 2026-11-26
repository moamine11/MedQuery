"""Validated profiles and reproducible engine identities."""

from hashlib import file_digest, sha256
import importlib.metadata
import json
from pathlib import Path
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ADAPTERS = ROOT / "experiments/extraction"
SETUP = ROOT / "data/ingestion/phase3/setup"
DEFAULT_PROFILE = Path(__file__).with_name("profile.json")
HPC_PROFILE = Path(__file__).with_name("profile_hpc.json")
THREAD_VARS = ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")


def canonical(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(data):
    return sha256(canonical(data)).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def load_profile(path=None, *, workers=None):
    profile = json.loads(DEFAULT_PROFILE.read_text())
    if path:
        override = json.loads(path.read_text())
        if not isinstance(override, dict) or set(override) - profile.keys():
            raise ValueError("Profile must be an object with supported setting names")
        for key, value in override.items():
            if type(value) is not type(profile[key]) and not (
                type(profile[key]) is float and type(value) is int
            ):
                raise ValueError(f"Wrong type for profile setting {key}")
        profile.update(override)
    if workers is not None:
        profile["workers"] = workers
    for field, low, high in (
        ("workers", 1, 16), ("worker_memory_mib", 192, 8192), ("memory_budget_mib", 192, 65536),
        ("page_timeout_seconds", 1, 600), ("optional_timeout_seconds", 1, 300),
        ("timeout_seconds", 1, 180), ("retries", 0, 2), ("max_pdf_mib", 1, 512),
        ("max_pixels", 100000, 50000000), ("dpi", 100, 400), ("ocr_workers", 1, 4),
        ("minimum_letters", 1, 500), ("region_max_count", 0, 12), ("arabic_min_letters", 1, 100),
        ("minimum_column_lines", 2, 20), ("minimum_ocr_confidence", 0, 100),
        ("heading_font_ratio", 1.01, 3),
    ):
        if not low <= profile[field] <= high:
            raise ValueError(f"{field} must be in [{low}, {high}]")
    for field in ("garble_fraction", "scan_area", "blank_edge", "region_min_area", "region_min_detail",
                  "region_padding", "native_region_max_text_density", "arabic_latin_fraction",
                  "arabic_line_disagreement"):
        if not 0 <= profile[field] <= 1:
            raise ValueError(f"{field} must be between 0 and 1")
    langs = profile["ocr_languages"]
    if not {"ar", "en", "fr", "mixed", "und"} <= langs.keys() or any(
        not isinstance(value, str) or not value or
        any(not part.isascii() or not part.isalpha() for part in value.split("+")) for value in langs.values()
    ):
        raise ValueError("OCR languages must specify local language packs for ar, en, fr, mixed and und")
    if profile["worker_memory_mib"] * profile["workers"] > profile["memory_budget_mib"]:
        raise ValueError("Worker address-space reservations exceed the configured memory budget")
    return profile


def worker_environment(threads):
    return {**os.environ, **dict.fromkeys(THREAD_VARS, str(threads)), "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True", "LC_ALL": "C.UTF-8"}


def command_version(command):
    try:
        proc = subprocess.run(command, check=True, capture_output=True, text=True, timeout=8)
        return (proc.stdout + proc.stderr).splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def local_tools(profile):
    tess_root = SETUP / "tesseract/usr"
    binary = tess_root / "bin/tesseract"
    tess_bin = os.getenv("MEDQUERY_TESSERACT") or os.getenv("PHASE3_TESSERACT") or (
        str(binary) if binary.is_file() else shutil.which("tesseract")
    )
    conda_tessdata = str(Path(tess_bin).resolve().parents[1] / "share/tessdata") if tess_bin else None
    paths = {
        "tesseract": tess_bin,
        "tessdata": os.getenv("MEDQUERY_TESSDATA") or os.getenv("PHASE3_TESSDATA") or os.getenv("TESSDATA_PREFIX") or
        str(tess_root / "share/tesseract-ocr/5/tessdata"),
        "poppler": shutil.which("pdftotext") if profile["poppler_enabled"] else None,
    }
    if not Path(paths["tessdata"]).is_dir():
        candidates = [Path("/usr/share/tesseract-ocr/5/tessdata"), Path("/usr/share/tessdata")]
        if conda_tessdata:
            candidates.insert(0, Path(conda_tessdata))
        paths["tessdata"] = next((str(p) for p in candidates if p.is_dir()), paths["tessdata"])
    versions = {package: importlib.metadata.version(package) for package in ("PyMuPDF", "Pillow")}
    for name in ("docling", "paddle"):
        if not profile[f"{name}_enabled"]:
            continue
        python = Path(
            os.getenv(f"MEDQUERY_{name.upper()}_PYTHON")
            or os.getenv(f"PHASE3_{name.upper()}_PYTHON")
            or (SETUP / f"{name}-venv/bin/python")
        )
        model_dir = Path(
            os.getenv(f"MEDQUERY_{name.upper()}_MODELS")
            or os.getenv(f"PHASE3_{name.upper()}_MODELS")
            or (SETUP / f"{name}_models")
        )
        if python.is_file() and model_dir.is_dir():
            packages = ("docling", "docling-core", "docling-parse", "docling-ibm-models", "torch", "transformers") if name == "docling" else (
                "paddleocr", "paddlex", "paddlepaddle",
            )
            versions[name] = {
                "packages": command_version([python, "-c",
                    f"from importlib.metadata import version; print({{p: version(p) for p in {packages!r}}})"]),
                "models": {p.relative_to(model_dir).as_posix(): file_hash(p)
                           for p in sorted(model_dir.rglob("*")) if p.is_file()},
            }
            paths.update({name: str(python.absolute()), f"{name}_models": str(model_dir.resolve())})
    versions.update({
        "tesseract": command_version([paths["tesseract"], "--version"]) if paths["tesseract"] else None,
        "poppler": command_version([paths["poppler"], "-v"]) if paths["poppler"] else None,
        "tessdata": {p.name: file_hash(p) for p in sorted(Path(paths["tessdata"]).glob("*.traineddata"))},
    })
    files = list(Path(__file__).parent.glob("*.py")) + [ADAPTERS / filename for filename in (
        "engines.py", "docling_adapter.py", "paddle_adapter.py", "prepare.py",
    )]
    versions["code"] = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(files)}
    return paths, versions
