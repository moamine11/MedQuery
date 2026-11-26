"""Run PDF extraction with page checkpoints and bounded CPU workers."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
from time import perf_counter

from ingestion.intake import IntakeError, asset_parts, open_local

from . import CONTRACT_VERSION, PIPELINE_VERSION
from .assemble import finalize_asset, review_entries
from .config import ROOT, canonical, fingerprint, worker_environment


@contextmanager
def atomic_file(target):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)


def write_json(target, payload):
    with atomic_file(target) as stream:
        stream.write(canonical(payload).decode() + "\n")


def append_json(stream, row):
    stream.write(canonical(row).decode() + "\n")


@contextmanager
def snapshot(root, asset, directory, limit):
    if asset.size > limit:
        raise IntakeError("asset_size_limit", "Asset exceeds the configured byte limit")
    with tempfile.TemporaryDirectory(dir=directory, prefix="source-") as temporary:
        target = Path(temporary) / "snapshot.pdf"
        digest, total = sha256(), 0
        with open_local(root, asset_parts(asset.path)) as source, target.open("wb") as output:
            while (part := source.read(1024 * 1024)):
                total += len(part)
                if total > asset.size:
                    raise IntakeError("source_size_changed", "PDF bytes differ from validated inventory")
                digest.update(part)
                output.write(part)
        if total != asset.size or digest.hexdigest() != asset.sha256:
            raise IntakeError("source_hash_changed", "PDF bytes differ from validated inventory")
        yield target


def checkpoint_read(target, key):
    try:
        data = json.loads(target.read_text())
        return data["page"] if data["key"] == key and data["checksum"] == fingerprint(data["page"]) else None
    except (OSError, ValueError, TypeError, KeyError):
        return None


def checkpoint_write(target, key, page):
    write_json(target, {"key": key, "checksum": fingerprint(page), "page": page})


def one_attempt(request, scratch):
    path = scratch / (fingerprint(request["context"]) + ".rss")
    path.unlink(missing_ok=True)
    worker_cmd = [sys.executable, "-m", "ingestion.extraction.worker"]
    time_bin = Path("/usr/bin/time")
    if time_bin.is_file() and os.access(time_bin, os.X_OK):
        command = [str(time_bin), "-q", "-f", "%M", "-o", str(path), *worker_cmd]
    else:
        command = worker_cmd
    start = perf_counter()
    proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, start_new_session=True,
                            env=worker_environment(request["profile"]["ocr_workers"]))
    state, output, error = "ok", None, None
    try:
        try:
            stdout, _ = proc.communicate(canonical(request), timeout=request["profile"]["page_timeout_seconds"])
        except BaseException as exc:
            # Kill the group, including OCR subprocesses, on deadline or interrupt.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
            if not isinstance(exc, subprocess.TimeoutExpired):
                raise
            state, error = "timeout", "page_deadline_exceeded"
        if state == "ok":
            if proc.returncode:
                state, error = "failed", "page_worker_exit_" + str(proc.returncode)
            else:
                try:
                    output = json.loads(stdout)
                    if any(output.get(k) != v for k, v in request["context"].items()):
                        raise ValueError("Page context does not match")
                except (ValueError, AttributeError):
                    state, error = "failed", "invalid_page_worker_output"
        peak = int(path.read_text().strip()) if path.is_file() and path.read_text().strip().isdigit() else None
    finally:
        path.unlink(missing_ok=True)
    return output, {"outcome": state, "elapsed_seconds": round(perf_counter() - start, 6),
                    "peak_rss_kib": peak, "error": error}


def failed_page(context, reason):
    return {
        "contract_version": CONTRACT_VERSION, **context, "status": "failed", "retrieval_eligible": False,
        "source_text": "", "blocks": [], "tables": [], "figures": [], "candidates": {},
        "printed_page_label": None, "pdf_page_label": None,
        "routing": {"route": "failed"}, "diagnostics": [
            {"code": reason, "severity": "review", "bbox": [0, 0, 1, 1]},
        ],
    }


def process_job(request, key, checkpoint, scratch, resume):
    if resume and (previous := checkpoint_read(checkpoint, key)):
        return previous, {"outcome": "reused", "attempts": []}
    attempts = []
    for _ in range(request["profile"]["retries"] + 1):
        page, observation = one_attempt(request, scratch)
        attempts.append(observation)
        if observation["outcome"] == "ok":
            checkpoint_write(checkpoint, key, page)
            return page, {"outcome": "completed", "attempts": attempts}
    return failed_page(request["context"], attempts[-1]["error"]), {"outcome": "failed", "attempts": attempts}


def run(root, assets, excluded, output, profile, tools, versions, *, resume, progress, asset_roots=None):
    start = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    scope = fingerprint({"assets": [asdict(asset) for asset in assets], "exclusions": excluded})
    content_settings = {k: v for k, v in profile.items() if k not in {"workers", "memory_budget_mib", "retries"}}
    config_key = fingerprint({"profile": content_settings, "engines": versions, "pipeline": PIPELINE_VERSION})
    identity = {"contract_version": CONTRACT_VERSION, "pipeline_version": PIPELINE_VERSION,
                "input_fingerprint": scope, "configuration_fingerprint": config_key}

    with (output / ".extraction.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another extraction is writing to this output directory") from error
        identity_path = output / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Output directory belongs to a different selection or extraction configuration")
        write_json(identity_path, identity)
        return run_locked(root, assets, excluded, output, profile, tools, versions, identity, resume, progress, start,
                          asset_roots=asset_roots)


def run_locked(root, assets, excluded, output, profile, tools, versions, identity, resume, progress, start,
               *, asset_roots=None):
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    states, engines, stats = Counter(), Counter(), []
    documents, timings = [], []
    with (
        ThreadPoolExecutor(max_workers=profile["workers"]) as pool,
        tempfile.TemporaryDirectory(dir=output, prefix="work-") as scratch_dir,
        atomic_file(output / "pages.jsonl") as page_file,
        atomic_file(output / "sections.jsonl") as section_file,
        atomic_file(output / "review_queue.jsonl") as review_file,
    ):
        scratch = Path(scratch_dir)
        total_assets = len(assets)
        for asset_idx, asset in enumerate(assets, 1):
            progress(f"[{asset_idx}/{total_assets}] Extracting {asset.key} ({len(asset.selected_pages)} pages)")
            contexts = [asset.context(n) for n in asset.selected_pages]
            asset_root = (asset_roots or {}).get(asset.key, root)
            done_pages = 0
            done_lock = threading.Lock()
            total_pages = len(contexts)
            try:
                with snapshot(asset_root, asset, scratch, profile["max_pdf_mib"] * 1024**2) as path:
                    def extract(context):
                        nonlocal done_pages
                        key = fingerprint({"context": context, "configuration": identity["configuration_fingerprint"],
                                           "source_bytes": asset.size, "physical_pages": asset.pages})
                        out = process_job({"snapshot": str(path), "context": context,
                                           "expected_pages": asset.pages, "profile": profile, "tools": tools},
                                          key, checkpoints / f"{key}.json", scratch, resume)
                        with done_lock:
                            done_pages += 1
                            if total_pages >= 50 and (done_pages % 25 == 0 or done_pages == total_pages):
                                progress(f"  [{asset.key}] {done_pages}/{total_pages} pages complete")
                        return out
                    measured = list(pool.map(extract, contexts))
            except IntakeError as error:
                measured = [(failed_page(context, error.code), {"outcome": "failed", "attempts": []})
                            for context in contexts]
            pages = [page for page, _ in measured]
            for section in finalize_asset(pages):
                append_json(section_file, {"asset_key": asset.key, **section})
            asset_states = Counter()
            review_count = 0
            for page, events in measured:
                append_json(page_file, page)
                for issue in review_entries(page, fingerprint):
                    append_json(review_file, issue)
                    review_count += 1
                states[page["status"]] += 1
                asset_states[page["status"]] += 1
                stats.extend(events["attempts"])
                engines.update(page["candidates"].keys())
                timings.append({"asset_key": asset.key, "physical_page": page["physical_page"], **events})
            documents.append({"asset_key": asset.key, "record_id": asset.record_id,
                              "asset_sha256": asset.sha256, "path": asset.path,
                              "physical_page_count": asset.pages, "selected_pages": list(asset.selected_pages),
                              "status_counts": asset_states, "review_items": review_count,
                              "diagnostics": [
                                  {"code": code, "count": count}
                                  for code, count in (("unextracted_physical_pages", asset.pages - len(pages)),
                                                      ("failed_pages", asset_states["failed"]),
                                                      ("pages_needing_review", asset_states["needs_review"]))
                                  if count],
                              "retrieval_eligible": False})
    write_json(output / "documents.json", {"assets": documents, "exclusions": excluded})
    write_json(output / "attempts.json", timings)
    summary = {
        **identity, "assets": len(assets), "excluded_assets": len(excluded), "page_statuses": states,
        "engine_candidates": engines, "page_count": sum(states.values()),
        "review_items": sum(doc["review_items"] for doc in documents),
        "reused_pages": sum(event["outcome"] == "reused" for event in timings),
        "page_worker_attempts": len(stats), "page_worker_failures": sum(s["outcome"] != "ok" for s in stats),
        "peak_worker_rss_mib": round(max((s["peak_rss_kib"] or 0 for s in stats), default=0) / 1024, 3),
        "elapsed_seconds": round(perf_counter() - start, 6), "engines": versions, "profile": profile,
    }
    write_json(output / "summary.json", summary)
    return summary
