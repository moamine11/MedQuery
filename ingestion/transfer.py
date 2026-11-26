"""Portable corpus verification, HPC transfer inventory, and archive packaging for Phase 9."""

from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import zipfile

from ingestion.chunking import read_jsonl, validate_chunk_citations
from ingestion.embeddings import DEFAULT_MODEL, get_model_spec, validate_vector
from ingestion.extraction.config import ROOT
from ingestion.extraction.selection import rebase_selection_manifest, verify_manifest_assets

CODE_INCLUDE_DIRS = ("ingestion", "database", "tests", "experiments/extraction")
CODE_INCLUDE_FILES = (
    "requirements.txt",
    "requirements-hpc.txt",
    "notebooks/full_ingestion_hpc.ipynb",
    "docs/ingestion/PHASE_9_HPC_INSTRUCTIONS.md",
    "docs/ingestion/metadata.schema.json",
)
FORBIDDEN_ARCHIVE_PATTERNS = (
    ".env",
    ".pgpass",
    ".venv/",
    "venv/",
    ".git/",
    "__pycache__/",
    ".pyc",
    ".dump",
    ".backup",
    ".sql.gz",
    "linux-cpu.lock.txt",
)


def _sha256_file(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_uri(collection, asset_path):
    clean = str(asset_path).lstrip("/")
    return f"medquery://{collection}/{clean}"


def _load_inventory_context(inventories_by_collection, *, root=ROOT):
    index = {}
    for col, rel_or_abs in sorted(inventories_by_collection.items()):
        inv_path = Path(rel_or_abs)
        if not inv_path.is_absolute():
            inv_path = (root / inv_path).resolve()
        if not inv_path.is_file():
            continue
        for entry in read_jsonl(inv_path):
            om = entry.get("original_metadata") or {}
            meta = entry.get("metadata") or om
            rec_id = meta.get("record_id")
            if not rec_id:
                continue
            for am in om.get("assets", []):
                path = am.get("path")
                if not path:
                    continue
                csv_row = am.get("adapter_context", {}).get("original_csv_row") or {}
                reuse = meta.get("reuse", {})
                a_reuse = am.get("reuse", {})
                index[(col, rec_id, path)] = {
                    "intake_review": meta.get("intake_review", "not_requested"),
                    "reuse_status": csv_row.get("reuse_status") or reuse.get("use_status") or "unknown",
                    "external_embedding_status": (
                        a_reuse.get("external_embedding_status")
                        or reuse.get("external_embedding_status")
                        or "unknown"
                    ),
                    "license_note": csv_row.get("license") or a_reuse.get("license_note") or reuse.get("license_code") or "",
                    "ai_reuse_condition": csv_row.get("ai_reuse_condition") or "",
                    "acquisition_method": csv_row.get("origin") or am.get("acquisition", {}).get("method") or "",
                    "access_status": csv_row.get("access_status") or "",
                }
    return index


def _classify_reuse_tier(ctx):
    status = ctx.get("reuse_status", "unknown")
    cond = ctx.get("ai_reuse_condition", "").lower()
    if status in {"conditional_cc_by", "public_domain_with_exceptions", "conditional_public_reuse"}:
        return "open_or_public_reuse"
    if status == "conditional_noncommercial" or "cc by-nc-sa" in ctx.get("license_note", "").lower():
        return "noncommercial_sharealike_academic"
    if status == "conditional_noncommercial_no_derivatives":
        return "noncommercial_tdm_no_derivatives"
    if "retain metadata only" in cond:
        return "unknown_rights_403_note"
    return "unknown_academic_research_only"


def build_transfer_inventory(
    manifest_path,
    *,
    root=ROOT,
    hpc_policy="private_academic_baseline",
    exclude_from_hpc=None,
):
    """Distinguish local-processing vs school-HPC eligibility across all manifest assets."""
    if hpc_policy not in {"private_academic_baseline", "strict_explicit_license_only"}:
        raise ValueError(f"Unsupported hpc_policy: {hpc_policy!r}")

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    inv_ctx = _load_inventory_context(manifest.get("inventories", {}), root=root)
    manual_exclude = set(exclude_from_hpc or ())
    records = []

    for item in manifest.get("eligible_assets", []):
        col = item["collection"]
        rec_id = item["record_id"]
        path = item["asset_path"]
        key = item["asset_key"]
        ctx = inv_ctx.get((col, rec_id, path), {})
        tier = _classify_reuse_tier(ctx)

        if key in manual_exclude or rec_id in manual_exclude:
            hpc_ok = False
            hpc_reason = "excluded_by_operator_hpc_policy"
        elif hpc_policy == "strict_explicit_license_only" and tier in {
            "unknown_academic_research_only",
            "unknown_rights_403_note",
        }:
            hpc_ok = False
            hpc_reason = "unknown_rights_held_for_local_fallback"
        else:
            hpc_ok = True
            hpc_reason = "authorized_private_academic_hpc_no_external_api"

        records.append({
            "asset_key": key,
            "record_id": rec_id,
            "collection": col,
            "asset_path": path,
            "portable_uri": _portable_uri(col, path),
            "hpc_relative_path": f"raw/{col}/{path}" if hpc_ok else None,
            "local_collection_root": item["collection_root"],
            "asset_role": item["asset_role"],
            "attachment_type": item.get("attachment_type"),
            "primary_asset_key": item.get("primary_asset_key"),
            "annex_asset_keys": item.get("annex_asset_keys", []),
            "target_categories": item.get("target_categories", [col]),
            "sha256": item["sha256"],
            "bytes": item["bytes"],
            "pages": item["pages"],
            "language": item["language"],
            "source_authority": item.get("source_authority"),
            "phase2_classification": item.get("phase2_classification"),
            "v1_scope_bucket": item.get("v1_scope_bucket"),
            "document_use_status": item.get("document_use_status", "unknown"),
            "asset_use_status": item.get("asset_use_status", "unknown"),
            "external_embedding_status": ctx.get("external_embedding_status", "unknown"),
            "reuse_tier": tier,
            "license_note": ctx.get("license_note", ""),
            "acquisition_method": ctx.get("acquisition_method", ""),
            "local_eligible": True,
            "hpc_authorized": hpc_ok,
            "local_fallback_required": not hpc_ok,
            "hpc_authorization_reason": hpc_reason,
            "exclusion_reasons": [],
        })

    # Keep primary/annex bundle atomic on HPC too: if primary is local-only, keep its annexes local-only.
    hpc_keys = {r["asset_key"] for r in records if r["hpc_authorized"]}
    for rec in records:
        if rec["hpc_authorized"] and rec["asset_role"] != "primary" and rec["primary_asset_key"] not in hpc_keys:
            rec["hpc_authorized"] = False
            rec["hpc_relative_path"] = None
            rec["local_fallback_required"] = True
            rec["hpc_authorization_reason"] = "primary_asset_held_for_local_fallback"

    for item in manifest.get("excluded_assets", []):
        col = item["collection"]
        path = item["asset_path"]
        records.append({
            "asset_key": item["asset_key"],
            "record_id": item["record_id"],
            "collection": col,
            "asset_path": path,
            "portable_uri": _portable_uri(col, path),
            "hpc_relative_path": None,
            "local_collection_root": item.get("collection_root", ""),
            "asset_role": item.get("asset_role"),
            "attachment_type": item.get("attachment_type"),
            "primary_asset_key": None,
            "annex_asset_keys": [],
            "target_categories": item.get("target_categories", []),
            "sha256": item.get("sha256", ""),
            "bytes": item.get("bytes", 0),
            "pages": item.get("pages", 0),
            "language": item.get("language", "und"),
            "source_authority": None,
            "phase2_classification": item.get("phase2_classification"),
            "v1_scope_bucket": item.get("v1_scope_bucket"),
            "document_use_status": "unknown",
            "asset_use_status": "unknown",
            "external_embedding_status": "unknown",
            "reuse_tier": "excluded_or_held",
            "license_note": "",
            "acquisition_method": "",
            "local_eligible": False,
            "hpc_authorized": False,
            "local_fallback_required": False,
            "hpc_authorization_reason": "excluded_from_v1_and_hpc",
            "exclusion_reasons": item.get("exclusion_reasons", []),
        })

    local_el = [r for r in records if r["local_eligible"]]
    hpc_el = [r for r in records if r["hpc_authorized"]]
    fallback = [r for r in records if r["local_fallback_required"]]
    excluded = [r for r in records if not r["local_eligible"]]

    summary = {
        "schema_version": 1,
        "hpc_policy": hpc_policy,
        "total_active_assets": len(records),
        "total_active_pages": sum(r["pages"] for r in records),
        "local_eligible_assets": len(local_el),
        "local_eligible_pages": sum(r["pages"] for r in local_el),
        "hpc_authorized_assets": len(hpc_el),
        "hpc_authorized_pages": sum(r["pages"] for r in hpc_el),
        "hpc_authorized_bytes": sum(r["bytes"] for r in hpc_el),
        "local_fallback_assets": len(fallback),
        "local_fallback_pages": sum(r["pages"] for r in fallback),
        "excluded_or_held_assets": len(excluded),
        "excluded_or_held_pages": sum(r["pages"] for r in excluded),
        "hpc_by_collection": dict(Counter(r["collection"] for r in hpc_el)),
        "hpc_by_reuse_tier": dict(Counter(r["reuse_tier"] for r in hpc_el)),
        "excluded_by_primary_reason": dict(
            Counter((r["exclusion_reasons"] or ["excluded"])[0] for r in excluded)
        ),
    }
    return {"summary": summary, "assets": records}


def write_transfer_inventory(inventory, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "transfer_inventory.json"
    csv_path = out / "transfer_inventory.csv"
    json_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "asset_key", "record_id", "collection", "asset_path", "portable_uri", "hpc_relative_path",
        "asset_role", "attachment_type", "primary_asset_key", "target_categories", "sha256",
        "bytes", "pages", "language", "source_authority", "phase2_classification", "v1_scope_bucket",
        "reuse_tier", "local_eligible", "hpc_authorized", "local_fallback_required",
        "hpc_authorization_reason", "exclusion_reasons",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in inventory["assets"]:
            flat = {k: row.get(k) for k in fieldnames}
            flat["target_categories"] = "|".join(row.get("target_categories") or [])
            flat["exclusion_reasons"] = "|".join(row.get("exclusion_reasons") or [])
            writer.writerow(flat)
    return json_path, csv_path


def _iter_code_files(root=ROOT):
    files = []
    for rel in CODE_INCLUDE_FILES:
        p = root / rel
        if p.is_file():
            files.append((rel, p))
    for drel in CODE_INCLUDE_DIRS:
        base = root / drel
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            if any(pat in rel for pat in FORBIDDEN_ARCHIVE_PATTERNS):
                continue
            if p.suffix in {".pyc", ".pyo", ".pdf"} or rel.startswith("tests/fixtures/"):
                continue
            if drel == "experiments/extraction" and p.name not in {
                "engines.py", "docling_adapter.py", "paddle_adapter.py", "prepare.py",
                "requirements-docling.txt", "requirements-paddle.txt", "requirements.txt",
                "download_docling_models.sh", "download_paddle_models.sh", "README.md",
            }:
                continue
            files.append((rel, p))
    # Deduplicate and sort deterministically.
    seen = {}
    for rel, p in files:
        seen[rel] = p
    return sorted(seen.items())


def build_transfer_archives(
    manifest_path,
    transfer_dir,
    *,
    root=ROOT,
    hpc_policy="private_academic_baseline",
    exclude_from_hpc=None,
):
    """Verify eligible PDFs and generate medquery_hpc_code.zip and medquery_hpc_input.zip."""
    transfer_dir = Path(transfer_dir)
    transfer_dir.mkdir(parents=True, exist_ok=True)

    verification = verify_manifest_assets(manifest_path, root=root, check_pdf_pages=True)
    if not verification["ok"]:
        raise ValueError(f"Manifest asset verification failed: {verification['issues']}")

    inventory = build_transfer_inventory(
        manifest_path,
        root=root,
        hpc_policy=hpc_policy,
        exclude_from_hpc=exclude_from_hpc,
    )
    inv_json, inv_csv = write_transfer_inventory(inventory, transfer_dir)

    hpc_assets = [r for r in inventory["assets"] if r["hpc_authorized"]]
    fallback_keys = {r["asset_key"] for r in inventory["assets"] if r["local_fallback_required"]}

    # Build portable manifests in a staging subfolder before packing input archive.
    staging = transfer_dir / "_staging_manifests"
    staging.mkdir(parents=True, exist_ok=True)

    portable_roots = {
        col: f"raw/{col}"
        for col in sorted({r["collection"] for r in hpc_assets})
    }
    hpc_manifest_path = staging / "extraction_selection_manifest.json"
    hpc_manifest = rebase_selection_manifest(
        manifest_path,
        hpc_manifest_path,
        portable_roots,
        root=root,
        inventories={},
        exclude_asset_keys=fallback_keys,
        exclusion_reason="held_for_local_fallback_not_hpc_authorized",
    )

    # Also build optional local fallback manifest if any local-only assets exist.
    fallback_manifest_path = transfer_dir / "local_fallback_manifest.json"
    if fallback_keys:
        orig = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        keep_keys = {r["asset_key"] for r in inventory["assets"] if r["local_fallback_required"]}
        exclude_for_fallback = {
            item["asset_key"] for item in orig["eligible_assets"] if item["asset_key"] not in keep_keys
        }
        rebase_selection_manifest(
            manifest_path,
            fallback_manifest_path,
            orig["collection_roots"],
            root=root,
            exclude_asset_keys=exclude_for_fallback,
            exclusion_reason="processed_on_hpc",
        )
    elif fallback_manifest_path.exists():
        fallback_manifest_path.unlink()

    path_map = {
        "schema_version": 1,
        "collection_roots_hpc_relative": portable_roots,
        "assets": [
            {
                "asset_key": r["asset_key"],
                "record_id": r["record_id"],
                "collection": r["collection"],
                "asset_path": r["asset_path"],
                "portable_uri": r["portable_uri"],
                "hpc_relative_path": r["hpc_relative_path"],
                "sha256": r["sha256"],
                "bytes": r["bytes"],
                "pages": r["pages"],
                "target_categories": r["target_categories"],
            }
            for r in hpc_assets
        ],
    }
    path_map_file = staging / "portable_path_map.json"
    path_map_file.write_text(json.dumps(path_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 1. Build medquery_hpc_code.zip
    code_zip = transfer_dir / "medquery_hpc_code.zip"
    code_files = _iter_code_files(root=root)
    with zipfile.ZipFile(code_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel, src in code_files:
            zf.write(src, arcname=rel)

    # 2. Build medquery_hpc_input.zip
    input_zip = transfer_dir / "medquery_hpc_input.zip"
    checksum_lines = []
    manifest_obj = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    roots_by_col = manifest_obj["collection_roots"]
    input_entries = []
    for r in sorted(hpc_assets, key=lambda x: (x["collection"], x["asset_key"])):
        col_root = Path(roots_by_col[r["collection"]])
        if not col_root.is_absolute():
            col_root = (root / col_root).resolve()
        src_pdf = col_root / r["asset_path"]
        arc_pdf = r["hpc_relative_path"]
        input_entries.append((arc_pdf, src_pdf, r["sha256"]))
        checksum_lines.append(f"{r['sha256']}  {arc_pdf}")

    for arc_meta, src_meta in (
        ("manifests/extraction_selection_manifest.json", hpc_manifest_path),
        ("manifests/transfer_inventory.json", inv_json),
        ("manifests/transfer_inventory.csv", inv_csv),
        ("manifests/portable_path_map.json", path_map_file),
    ):
        digest = _sha256_file(src_meta)
        input_entries.append((arc_meta, src_meta, digest))
        checksum_lines.append(f"{digest}  {arc_meta}")

    checksums_file = staging / "checksums.sha256"
    checksums_file.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    input_entries.append(("manifests/checksums.sha256", checksums_file, _sha256_file(checksums_file)))

    with zipfile.ZipFile(input_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for arcname, src, _ in input_entries:
            zf.write(src, arcname=arcname)

    code_check = verify_transfer_archive(code_zip, kind="code")
    input_check = verify_transfer_archive(
        input_zip,
        kind="input",
        expected_pdf_count=len(hpc_assets),
        expected_pdf_sha256s={r["sha256"] for r in hpc_assets},
    )

    transfer_checksums = transfer_dir / "transfer_checksums.sha256"
    transfer_checksums.write_text(
        f"{_sha256_file(code_zip)}  {code_zip.name}\n"
        f"{_sha256_file(input_zip)}  {input_zip.name}\n"
        f"{_sha256_file(inv_json)}  {inv_json.name}\n",
        encoding="utf-8",
    )

    report = {
        "corpus_verification": verification,
        "transfer_summary": inventory["summary"],
        "hpc_manifest_summary": hpc_manifest["summary"],
        "archives": {
            "code": code_check,
            "input": input_check,
        },
        "estimated_hpc_footprint_gib": round(
            (code_check["uncompressed_bytes"] + input_check["uncompressed_bytes"] + 4.0 * 1024**3) / 1024**3,
            2,
        ),
    }
    (transfer_dir / "archive_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def verify_transfer_archive(archive_path, *, kind, expected_pdf_count=None, expected_pdf_sha256s=None):
    """Validate zip integrity, forbidden exclusions, and manifest/PDF checksums."""
    archive_path = Path(archive_path)
    with zipfile.ZipFile(archive_path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt zip member {bad} in {archive_path}")
        infos = zf.infolist()
        names = [info.filename for info in infos]
        for name in names:
            if any(pat in name for pat in FORBIDDEN_ARCHIVE_PATTERNS):
                raise ValueError(f"Forbidden file leaked into {kind} archive: {name}")

        uncompressed = sum(info.file_size for info in infos)
        pdf_names = [n for n in names if n.lower().endswith(".pdf")]
        if kind == "code":
            if pdf_names:
                raise ValueError(f"Code archive unexpectedly contains PDFs: {pdf_names[:5]}")
        elif kind == "input":
            if expected_pdf_count is not None and len(pdf_names) != expected_pdf_count:
                raise ValueError(f"Expected {expected_pdf_count} PDFs in input archive, found {len(pdf_names)}")
            if any(not n.startswith("raw/") for n in pdf_names):
                raise ValueError("Input archive contains PDFs outside raw/ authorized tree")
            if "manifests/checksums.sha256" not in names:
                raise ValueError("Input archive missing manifests/checksums.sha256")
            lines = zf.read("manifests/checksums.sha256").decode("utf-8").splitlines()
            actual_shas = set()
            for line in lines:
                if not line.strip():
                    continue
                expected_hex, rel = line.split("  ", 1)
                computed = sha256(zf.read(rel)).hexdigest()
                if computed != expected_hex:
                    raise ValueError(f"Checksum mismatch inside archive for {rel}: {computed} != {expected_hex}")
                if rel.lower().endswith(".pdf"):
                    actual_shas.add(computed)
            if expected_pdf_sha256s is not None and actual_shas != set(expected_pdf_sha256s):
                raise ValueError("Input archive PDF SHA-256 set does not match authorized HPC assets")
        else:
            raise ValueError(f"Unknown archive kind {kind!r}")

    return {
        "path": str(archive_path),
        "file_count": len(names),
        "pdf_count": len(pdf_names),
        "compressed_bytes": archive_path.stat().st_size,
        "compressed_mib": round(archive_path.stat().st_size / 1024**2, 2),
        "uncompressed_bytes": uncompressed,
        "uncompressed_mib": round(uncompressed / 1024**2, 2),
        "sha256": _sha256_file(archive_path),
        "verified": True,
    }


def verify_phase9_outputs(manifest_path, extract_dir, chunk_dir, embed_dir, *, expected_model=DEFAULT_MODEL):
    """Verify full-scale HPC extraction, chunking, and embedding artifacts before archiving/loading."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    eligible_items = {item["asset_key"]: item for item in manifest.get("eligible_assets", [])}
    excluded_keys = {item["asset_key"] for item in manifest.get("excluded_assets", [])}
    excluded_shas = {item["sha256"] for item in manifest.get("excluded_assets", []) if item.get("sha256")}

    extract_dir = Path(extract_dir)
    chunk_dir = Path(chunk_dir)
    embed_dir = Path(embed_dir)

    pages = read_jsonl(extract_dir / "pages.jsonl")
    sections = read_jsonl(extract_dir / "sections.jsonl") if (extract_dir / "sections.jsonl").is_file() else []
    review_items = read_jsonl(extract_dir / "review_queue.jsonl") if (extract_dir / "review_queue.jsonl").is_file() else []
    chunks = read_jsonl(chunk_dir / "chunks.jsonl")
    chunk_summary = json.loads((chunk_dir / "chunk_summary.json").read_text(encoding="utf-8"))
    emb_model = json.loads((embed_dir / "embedding_model.json").read_text(encoding="utf-8"))
    embeddings = read_jsonl(embed_dir / "chunk_embeddings.jsonl")

    expected_pages = sum(item["pages"] for item in eligible_items.values())
    processed_asset_keys = {p["asset_key"] for p in pages}
    processed_shas = {p["asset_sha256"] for p in pages}

    page_statuses = Counter(p.get("status") for p in pages)
    pages_by_asset = Counter(p["asset_key"] for p in pages)
    incomplete_assets = []
    for key, item in sorted(eligible_items.items()):
        got = pages_by_asset.get(key, 0)
        failed_here = sum(1 for p in pages if p["asset_key"] == key and p.get("status") in {"failed", "unreadable"})
        if got != item["pages"] or failed_here > 0:
            incomplete_assets.append({
                "asset_key": key,
                "expected_pages": item["pages"],
                "processed_pages": got,
                "failed_or_unreadable": failed_here,
            })

    held_leakage_keys = sorted(processed_asset_keys & excluded_keys)
    # Only flag excluded SHAs if not legitimately shared with an eligible deduplicated asset.
    eligible_shas = {item["sha256"] for item in eligible_items.values()}
    held_leakage_shas = sorted((processed_shas & excluded_shas) - eligible_shas)

    chunk_ids = [c["chunk_id"] for c in chunks]
    duplicate_chunk_ids = len(chunk_ids) - len(set(chunk_ids))
    citation_issues = validate_chunk_citations(chunks, pages)

    spec = get_model_spec(expected_model, revision=emb_model.get("model_revision"))
    emb_by_chunk = {}
    emb_errors = []
    for row in embeddings:
        cid = row.get("chunk_id")
        if cid in emb_by_chunk:
            emb_errors.append((cid, "duplicate_embedding"))
            continue
        if row.get("model_name") != spec["model_name"] or row.get("model_revision") != spec["model_revision"]:
            emb_errors.append((cid, "model_or_revision_mismatch"))
            continue
        try:
            validate_vector(row["embedding"], spec["dimensions"], distance_metric=spec["distance_metric"])
        except ValueError as err:
            emb_errors.append((cid, str(err)))
            continue
        emb_by_chunk[cid] = row

    missing_embeddings = [c["chunk_id"] for c in chunks if c["chunk_id"] not in emb_by_chunk]
    extra_embeddings = [cid for cid in emb_by_chunk if cid not in set(chunk_ids)]

    return {
        "expected_assets": len(eligible_items),
        "processed_assets": len(processed_asset_keys),
        "unique_processed_sha256": len(processed_shas),
        "expected_pages": expected_pages,
        "processed_pages": len(pages),
        "page_statuses": dict(page_statuses),
        "sections_count": len(sections),
        "review_items_count": len(review_items),
        "incomplete_assets": incomplete_assets,
        "held_leakage_keys": held_leakage_keys,
        "held_leakage_shas": held_leakage_shas,
        "chunks_count": len(chunks),
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "unchunkable_documents": chunk_summary.get("unchunkable_documents", []),
        "citation_issues_count": len(citation_issues),
        "embeddings_count": len(embeddings),
        "embedding_errors_count": len(emb_errors),
        "missing_embeddings_count": len(missing_embeddings),
        "extra_embeddings_count": len(extra_embeddings),
        "model_name": emb_model.get("model_name"),
        "model_revision": emb_model.get("model_revision"),
        "dimensions": emb_model.get("dimensions"),
        "storage_compatible": (
            not held_leakage_keys
            and not held_leakage_shas
            and duplicate_chunk_ids == 0
            and not citation_issues
            and not emb_errors
            and not missing_embeddings
            and not extra_embeddings
        ),
        "all_pages_complete": len(incomplete_assets) == 0 and len(pages) == expected_pages,
    }
