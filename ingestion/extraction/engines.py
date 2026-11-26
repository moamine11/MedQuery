"""Adapt the frozen Phase 3 engines to a production page context."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

import pymupdf

from experiments.extraction import engines as benchmark

from .config import ADAPTERS


@dataclass
class PageEngines:
    path: Path
    physical_page: int
    page: pymupdf.Page
    profile: dict
    tools: dict

    def extract(self, name, language, region=None):
        if name == "native":
            return benchmark.native(self.page, self.profile)
        if name == "native_tables":
            return benchmark.native(self.page, self.profile, structure=True)
        if name == "tesseract":
            if not self.tools["tesseract"]:
                raise FileNotFoundError("Tesseract binary missing")
            return benchmark.tess(self.page, self.profile, self.tools["tesseract"], self.tools["tessdata"],
                                  language, region)
        if name == "poppler":
            if not self.tools["poppler"]:
                raise FileNotFoundError("Poppler binary missing")
            # The adapter invokes pdftotext by name. Validate its geometry against the displayed page.
            raw = benchmark.poppler(self.path, self.physical_page, "poppler", self.profile["timeout_seconds"])
            diag = raw["diagnostics"]
            if abs(diag["bbox_stream_width_pt"] - self.page.rect.width) > 1 or abs(
                diag["bbox_stream_height_pt"] - self.page.rect.height
            ) > 1:
                raise ValueError("Poppler page geometry does not match displayed CropBox")
            return raw
        if name in {"docling", "paddle"}:
            if not self.tools.get(name):
                raise FileNotFoundError(f"{name} environment or model directory missing")
            proc = subprocess.run(
                [self.tools[name], "-m", "ingestion.extraction.engines", name],
                input=json.dumps({"path": str(self.path), "physical_page": self.physical_page,
                                  "language": language, "models": self.tools[f"{name}_models"],
                                  "profile": self.profile}),
                capture_output=True, text=True, check=True, timeout=self.profile["optional_timeout_seconds"],
            )
            return json.loads(proc.stdout)
        raise ValueError(f"Unknown extraction engine {name}")


def optional_worker(name, request):
    # These historical adapters import their siblings by file name.
    sys.path.insert(0, str(ADAPTERS))
    if name == "docling":
        from docling_adapter import extract
    else:
        from paddle_adapter import extract
    profile = {**request["profile"], "working_dir": os.getenv("MEDQUERY_EXTRACTION_WORK_DIR", "data/ingestion/phase4")}
    return extract(Path(request["path"]), request, Path(request["models"]), profile)


if __name__ == "__main__":
    from contextlib import redirect_stdout

    with redirect_stdout(sys.stderr):
        raw = optional_worker(sys.argv[1], json.load(sys.stdin))
    print(json.dumps(raw, ensure_ascii=False, allow_nan=False))
