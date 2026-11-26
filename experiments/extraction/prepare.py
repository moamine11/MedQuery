"""Resolve the unheld pilot against Phase 2 and render source evidence locally."""

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import pymupdf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ingestion.intake import asset_parts, open_local  # noqa: E402

CONFIG = Path(__file__).with_name("config.json")


def load_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def local_output(path):
    data = REPO / "data"
    if data.is_symlink() or not path.resolve().is_relative_to(data):
        raise ValueError("Output must be inside the repository's ignored data directory.")
    return path


def prepare(config):
    root = REPO / config["collection_root"]
    phase2 = REPO / config["phase2"]
    out = local_output(REPO / config["working_dir"])
    renders = out / "review_renders"
    renders.mkdir(parents=True, exist_ok=True)
    observations = {row["path"]: row for row in load_jsonl(phase2 / "files.jsonl")}
    inventory = {asset["path"]: entry for entry in load_jsonl(phase2 / "inventory.jsonl")
                 for asset in entry.get("metadata", {}).get("assets", [])}
    pages, exclusions, seen = [], [], set()
    with (REPO / config["pilot"]).open(encoding="utf-8", newline="") as stream:
        pilot = list(csv.DictReader(stream))
    if len(pilot) != 23:
        raise ValueError("Expected the existing 23-asset pilot.")
    for row in pilot:
        path, asset_id = row["path"], row["asset_id"]
        if asset_id in seen or not re.fullmatch(r"[A-Za-z0-9_-]+", asset_id):
            raise ValueError(f"Duplicate or invalid pilot asset ID: {asset_id}")
        seen.add(asset_id)
        entry, probe = inventory[path], observations[path]
        metadata = entry["metadata"]
        if metadata["record_id"] != row["record_id"]:
            raise ValueError(f"Pilot record mismatch: {asset_id}")
        reasons = []
        if metadata["intake_review"] == "hold":
            reasons.append("intake_hold")
        if entry["validation_state"] != "valid":
            reasons.append("invalid_inventory")
        if probe.get("error"):
            reasons.append(probe["error"]["code"])
        samples = [int(number) for number in row["sample_physical_pages"].split(",")]
        if reasons:
            exclusions.append({"asset_id": asset_id, "physical_pages": samples, "reasons": reasons})
            continue
        expected = probe["observed"]
        if (expected["sha256"], expected["pages"], expected["bytes"]) != (
            row["sha256"], int(row["physical_pages"]), int(row["bytes"])
        ):
            raise ValueError(f"Phase 1 / Phase 2 asset mismatch: {asset_id}")
        if len(set(samples)) != len(samples) or any(n < 1 or n > expected["pages"] for n in samples):
            raise ValueError(f"Invalid pilot page selection: {asset_id}")
        with open_local(root, asset_parts(path)) as stream:
            if os.fstat(stream.fileno()).st_size != expected["bytes"]:
                raise ValueError(f"Source size mismatch: {asset_id}")
            source = stream.read()
        if hashlib.sha256(source).hexdigest() != expected["sha256"]:
            raise ValueError(f"Source hash mismatch: {asset_id}")
        with pymupdf.open(stream=source, filetype="pdf") as pdf:
            if pdf.is_encrypted or pdf.page_count != expected["pages"]:
                raise ValueError(f"Source page tree or encryption changed: {asset_id}")
            for number in samples:
                page = pdf[number - 1]
                key = f"{asset_id}__{number:04}"
                render = renders / f"{key}.png"
                page.get_pixmap(dpi=config["review_dpi"], colorspace=pymupdf.csRGB, alpha=False).save(render)
                pages.append({
                    "key": key, "path": path, "asset_id": asset_id, "physical_page": number,
                    "language": row["language"], "stratum": row["stratum"], "sha256": row["sha256"],
                    "split": "development" if asset_id in config["development_assets"] else "held_out",
                    "pdf_page_label": page.get_label() or None, "rotation": page.rotation,
                    "width_pt": page.rect.width, "height_pt": page.rect.height,
                    "review_render": str(render.relative_to(REPO)),
                    "review_render_sha256": hashlib.file_digest(render.open("rb"), "sha256").hexdigest(),
                })
    write_json(out / "manifest.json", {"pilot_assets": len(pilot), "pages": pages, "exclusions": exclusions})
    print(f"{len(pages)} physical pages from {len({page['asset_id'] for page in pages})} assets. "
          f"Excluded {len(exclusions)} assets.")


if __name__ == "__main__":
    prepare(json.loads(CONFIG.read_text()))
