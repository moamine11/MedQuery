"""Select authorized inputs from the validated Phase 2 inventory."""

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re

from ingestion.intake import asset_parts
from ingestion.metadata import validate

from .config import ROOT, canonical, fingerprint


@dataclass(frozen=True)
class Asset:
    record_id: str
    key: str
    path: str
    sha256: str
    size: int
    pages: int
    language: str
    role: str
    attachment: str | None
    revision_key: str | None
    acquisition_url: str | None
    document_use: str
    asset_use: str
    selected_pages: tuple[int, ...]

    def context(self, number):
        return {"asset_key": self.key, "record_id": self.record_id, "asset_sha256": self.sha256,
                "source_path": self.path, "physical_page": number, "language": self.language,
                "asset_role": self.role, "attachment_type": self.attachment,
                "revision_key": self.revision_key, "acquisition_url": self.acquisition_url,
                "document_use_status": self.document_use, "asset_use_status": self.asset_use}


def page_numbers(value, upper):
    result = set()
    for part in value.split(","):
        if not re.fullmatch(r"[1-9]\d*(?:-[1-9]\d*)?", part):
            raise ValueError("Pages must be a comma-separated list of physical numbers or ranges")
        start, end = (int(n) for n in (part.split("-", 1) if "-" in part else [part, part]))
        if end < start or end > upper:
            raise ValueError(f"Page range {part} exceeds the valid physical page range 1–{upper}")
        result.update(range(start, end + 1))
    return result


def read_pilot(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len({row["asset_id"] for row in rows}) != len(rows) or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Pilot repeats an asset identifier or source path")
    return {row["path"]: row for row in rows}


def select(inventory, pilot, documents, page_filters):
    pilot_rows = read_pilot(pilot) if pilot else {}
    if not pilot and not documents:
        raise ValueError("Select a --pilot or at least one --document")
    pilot_ids = {row["asset_id"] for row in pilot_rows.values()}
    requested = set(documents)
    matches, filter_matches, pilot_matches, seen_keys = set(), set(), set(), set()
    assets, excluded = [], []
    with inventory.open(encoding="utf-8") as stream:
        inventory_rows = [json.loads(line) for line in stream if line.strip()]
    if len({row.get("collection") for row in inventory_rows}) != 1:
        raise ValueError("Use one validated collection inventory")
    for entry in inventory_rows:
        raw = entry.get("metadata") or entry.get("original_metadata") or {}
        record_id = raw.get("record_id")
        targets = []
        for asset in raw.get("assets", []):
            path = asset.get("path")
            row = pilot_rows.get(path)
            aliases = {record_id}
            if row:
                aliases.add(row["asset_id"])
            if (pilot and not row) or (requested and not aliases & requested):
                continue
            matches.update(aliases & requested)
            if row:
                pilot_matches.add(path)
            targets.append((asset, row, aliases))

        if not targets:
            continue
        if entry.get("validation_state") == "valid":
            metadata, issues = validate(raw)
            if any(i["severity"] == "error" for i in issues):
                raise ValueError(f"Validated inventory entry no longer passes metadata validation: {record_id}")
        else:
            metadata = raw
        for asset, row, aliases in targets:
            path = asset["path"]
            key = row["asset_id"] if row else record_id if asset.get("role") == "primary" else (
                record_id + "@" + fingerprint(path)[:12]
            )
            if key in seen_keys:
                raise ValueError(f"Selected inventory repeats asset key {key}")
            seen_keys.add(key)
            aliases.add(key)
            filters = []
            for i, spec in enumerate(page_filters):
                target, separator, numbers = spec.partition(":")
                if not separator:
                    filters.append((i, target))
                elif target == key or target in aliases and target not in pilot_ids:
                    filters.append((i, numbers))
            filter_matches.update(i for i, _ in filters)
            if page_filters and not filters:
                continue
            observed = entry.get("observations", {}).get(path, {})
            doc_use = metadata.get("reuse", {}).get("use_status", "unknown")
            asset_use = asset.get("reuse", {}).get("use_status", "unknown")
            if entry.get("validation_state") != "valid" or metadata.get("intake_review") == "hold" or (
                {"restricted", "prohibited"} & {doc_use, asset_use}
            ):
                excluded.append({"asset_key": key, "record_id": record_id, "asset_sha256": observed.get("sha256"),
                                 "path": path, "reason": "inventory_hold_invalid_or_permission_denied"})
                continue
            size, count, sha = (observed.get(k) for k in ("bytes", "pages", "sha256"))
            if type(size) is not int or size < 1 or type(count) is not int or count < 1 or (
                not isinstance(sha, str) or not re.fullmatch("[a-f0-9]{64}", sha)
            ):
                raise ValueError(f"Missing validated file observation: {key}")
            asset_parts(path)
            if Path(path).suffix.casefold() != ".pdf":
                raise ValueError(f"Selected asset is not a PDF: {key}")
            if row and (row["record_id"], row["sha256"], int(row["bytes"]), int(row["physical_pages"])) != (
                record_id, sha, size, count
            ):
                raise ValueError(f"Pilot does not match the validated inventory: {key}")
            pages = set()
            if filters:
                for _, spec in filters:
                    pages.update(page_numbers(spec, count))
            elif row:
                pages = page_numbers(row["sample_physical_pages"], count)
            else:
                pages = set(range(1, count + 1))
            assets.append(Asset(record_id, key, path, sha, size, count, metadata["language"],
                                asset["role"], asset.get("attachment_type"), entry.get("revision_key"),
                                asset.get("acquisition", {}).get("resolved_url"),
                                doc_use, asset_use, tuple(sorted(pages))))
    if requested - matches:
        raise ValueError(f"Unknown or out-of-pilot document IDs: {sorted(requested - matches)}")
    if pilot and not requested and set(pilot_rows) - pilot_matches:
        raise ValueError("Pilot contains paths absent from the validated inventory")
    if len(filter_matches) != len(page_filters):
        raise ValueError("A physical page filter did not match a selected asset")
    if not assets and not excluded:
        raise ValueError("No assets matched the selection")
    return sorted(assets, key=lambda a: (a.record_id, a.key)), sorted(excluded, key=lambda a: a["asset_key"])


def read_jsonl(path):
    """Read a UTF-8 JSONL file strictly by physical lines (never str.splitlines)."""
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            stripped = line.rstrip("\r\n")
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _asset_key(record_id, asset):
    return record_id if asset.get("role") == "primary" else f"{record_id}@{fingerprint(asset['path'])[:12]}"


def _inventory_index(inventories_by_collection):
    """Index inventory entries and identify every SHA-256 touched by a hold, invalid state, or restriction."""
    entries = {}
    sha_holds = defaultdict(list)
    sha_categories = defaultdict(set)
    for collection, inv_path in sorted(inventories_by_collection.items()):
        for entry in read_jsonl(inv_path):
            if entry.get("collection") != collection:
                raise ValueError(f"Inventory {inv_path} collection mismatch: expected {collection}")
            raw = entry.get("metadata") or entry.get("original_metadata") or {}
            record_id = raw.get("record_id")
            if entry.get("validation_state") == "valid":
                metadata, issues = validate(raw)
                if any(i["severity"] == "error" for i in issues):
                    raise ValueError(f"Validated inventory entry no longer passes metadata validation: {record_id}")
            else:
                metadata = raw
            doc_use = metadata.get("reuse", {}).get("use_status", "unknown")
            intake_review = metadata.get("intake_review", "not_requested")
            classification = entry.get("classification", "unresolved")
            assets = metadata.get("assets", [])
            annex_keys = [_asset_key(record_id, a) for a in assets if a.get("role") != "primary"]
            primary_key = next((_asset_key(record_id, a) for a in assets if a.get("role") == "primary"), None)
            for asset in assets:
                path = asset.get("path")
                obs = entry.get("observations", {}).get(path, {})
                sha = obs.get("sha256", "")
                asset_use = asset.get("reuse", {}).get("use_status", "unknown")
                hold_reasons = []
                if entry.get("validation_state") != "valid":
                    hold_reasons.append("invalid_record")
                if intake_review == "hold":
                    hold_reasons.append("intake_review_hold")
                if {"restricted", "prohibited"} & {doc_use, asset_use}:
                    hold_reasons.append("permission_restricted_or_prohibited")
                if sha:
                    sha_categories[sha].add(collection)
                    if hold_reasons:
                        sha_holds[sha].append({"collection": collection, "record_id": record_id, "reasons": hold_reasons})
                entries[(collection, record_id, path)] = {
                    "collection": collection,
                    "record_id": record_id,
                    "asset_key": _asset_key(record_id, asset),
                    "path": path,
                    "role": asset.get("role"),
                    "attachment_type": asset.get("attachment_type"),
                    "revision_key": entry.get("revision_key"),
                    "acquisition_url": asset.get("acquisition", {}).get("resolved_url"),
                    "language": metadata.get("language", "und"),
                    "content_languages": metadata.get("content_languages", []),
                    "title": metadata.get("title", ""),
                    "document_type": metadata.get("document_type", "unknown"),
                    "source_authority": metadata.get("authority", "unknown"),
                    "classification": classification,
                    "validation_state": entry.get("validation_state"),
                    "intake_review": intake_review,
                    "document_use_status": doc_use,
                    "asset_use_status": asset_use,
                    "hold_reasons": hold_reasons,
                    "observed": obs,
                    "primary_asset_key": primary_key,
                    "annex_asset_keys": annex_keys,
                }
    return entries, sha_holds, sha_categories


def build_selection_manifest(plan_path, inventories_by_collection, *, root=ROOT):
    """Combine the unified ingestion plan with extraction eligibility rules into an executable manifest."""
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    entries, sha_holds, sha_categories = _inventory_index(inventories_by_collection)
    roots = dict(plan.get("collection_roots", {}))
    eligible, excluded = [], []
    seen_shas = set()
    seen_keys = set()

    for row in plan.get("assets", []):
        col = row["collection"]
        rec_id = row["record_id"]
        path = row["asset_path"]
        sha = row.get("sha256", "")
        inv = entries.get((col, rec_id, path))
        target_cats = sorted(
            {c for c in (row.get("target_categories", "").split("|") if row.get("target_categories") else []) if c}
            | sha_categories.get(sha, set())
            | ({col} if not row.get("record_id", "").startswith("EXCLUDED:") else set())
        )
        reasons = []
        if not row.get("technical_extraction_eligible"):
            reasons.append(row.get("exclusion_or_hold_reason") or row.get("v1_scope_bucket") or "plan_ineligible")
        if row.get("v1_scope_bucket") == "held_or_unresolved" or row.get("phase2_classification") == "unresolved":
            reasons.append("unresolved_or_held_in_plan")
        if inv is None:
            if not reasons:
                reasons.append("missing_from_validated_inventory")
        else:
            if inv["classification"] == "unresolved":
                reasons.append("unresolved_classification")
            reasons.extend(inv["hold_reasons"])
            if sha and sha in sha_holds and not inv["hold_reasons"]:
                reasons.append("duplicate_circumvents_hold")
        if sha and sha in seen_shas and not reasons:
            reasons.append("duplicate_sha256")

        if reasons:
            excluded.append({
                "asset_key": inv["asset_key"] if inv else rec_id,
                "record_id": rec_id,
                "collection": col,
                "collection_root": roots.get(col, row.get("collection_root", "")),
                "asset_path": path,
                "asset_role": row.get("asset_role"),
                "attachment_type": inv["attachment_type"] if inv else None,
                "sha256": sha,
                "bytes": row.get("bytes", 0),
                "pages": row.get("pages", 0),
                "language": row.get("language", "und"),
                "v1_scope_bucket": row.get("v1_scope_bucket"),
                "phase2_classification": row.get("phase2_classification"),
                "target_categories": target_cats,
                "exclusion_reasons": list(dict.fromkeys(reasons)),
            })
            continue

        obs = inv["observed"]
        size, count, obs_sha = (obs.get(k) for k in ("bytes", "pages", "sha256"))
        if type(size) is not int or size < 1 or type(count) is not int or count < 1 or (
            not isinstance(obs_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", obs_sha)
        ):
            raise ValueError(f"Missing validated file observation: {inv['asset_key']}")
        asset_parts(path)
        if Path(path).suffix.casefold() != ".pdf":
            raise ValueError(f"Selected asset is not a PDF: {inv['asset_key']}")
        key = inv["asset_key"]
        if key in seen_keys:
            raise ValueError(f"Manifest repeats eligible asset key {key}")
        seen_keys.add(key)
        seen_shas.add(obs_sha)
        eligible.append({
            "asset_key": key,
            "record_id": rec_id,
            "collection": col,
            "collection_root": roots.get(col, row.get("collection_root", "")),
            "inventory_path": str(Path(inventories_by_collection[col]).as_posix()),
            "asset_path": path,
            "asset_role": inv["role"],
            "attachment_type": inv["attachment_type"],
            "primary_asset_key": inv["primary_asset_key"],
            "annex_asset_keys": inv["annex_asset_keys"],
            "sha256": obs_sha,
            "bytes": size,
            "pages": count,
            "language": inv["language"],
            "content_languages": inv["content_languages"],
            "title": inv["title"],
            "document_type": inv["document_type"],
            "source_authority": inv["source_authority"],
            "phase2_classification": inv["classification"],
            "v1_scope_bucket": row.get("v1_scope_bucket"),
            "revision_key": inv["revision_key"],
            "acquisition_url": inv["acquisition_url"],
            "target_categories": target_cats,
            "document_use_status": inv["document_use_status"],
            "asset_use_status": inv["asset_use_status"],
            "clinical_review": row.get("clinical_review", "not_assessed"),
            "treatment_guidance_eligible": False,
            "retrieval_eligible": False,
        })

    # Ensure no annex (correction/supplement) is orphaned if its primary was excluded.
    eligible_keys = {item["asset_key"] for item in eligible}
    orphaned = [item for item in eligible if item["asset_role"] != "primary" and item["primary_asset_key"] not in eligible_keys]
    if orphaned:
        orphan_keys = {item["asset_key"] for item in orphaned}
        eligible = [item for item in eligible if item["asset_key"] not in orphan_keys]
        for item in orphaned:
            excluded.append({
                **{k: item[k] for k in ("asset_key", "record_id", "collection", "collection_root", "asset_path",
                                         "asset_role", "attachment_type", "sha256", "bytes", "pages", "language",
                                         "v1_scope_bucket", "phase2_classification", "target_categories")},
                "exclusion_reasons": ["primary_asset_excluded"],
            })

    eligible.sort(key=lambda item: (item["collection"], item["record_id"], item["asset_key"]))
    excluded.sort(key=lambda item: (item["collection"], item["record_id"], item["asset_key"]))

    by_collection = {}
    for col in sorted(roots):
        col_el = [item for item in eligible if item["collection"] == col]
        col_ex = [item for item in excluded if item["collection"] == col]
        by_collection[col] = {
            "eligible_assets": len(col_el),
            "eligible_pages": sum(item["pages"] for item in col_el),
            "excluded_assets": len(col_ex),
            "excluded_pages": sum(item["pages"] for item in col_ex),
        }

    summary = {
        "total_active_assets": len(eligible) + len(excluded),
        "total_active_pages": sum(item["pages"] for item in eligible) + sum(item["pages"] for item in excluded),
        "plan_technical_eligible_assets": plan.get("summary", {}).get("technical_extraction_eligible", 0),
        "plan_technical_eligible_pages": plan.get("summary", {}).get("technical_extraction_eligible_pages", 0),
        "executable_eligible_assets": len(eligible),
        "executable_eligible_pages": sum(item["pages"] for item in eligible),
        "unique_eligible_sha256": len({item["sha256"] for item in eligible}),
        "excluded_or_held_assets": len(excluded),
        "excluded_or_held_pages": sum(item["pages"] for item in excluded),
        "by_collection": by_collection,
        "eligible_by_bucket": dict(Counter(item["v1_scope_bucket"] for item in eligible)),
        "eligible_by_language": dict(Counter(item["language"] for item in eligible)),
        "excluded_by_primary_reason": dict(Counter(item["exclusion_reasons"][0] for item in excluded)),
    }
    return {
        "schema_version": 1,
        "source_plan": str(Path(plan_path).as_posix()),
        "inventories": {k: str(Path(v).as_posix()) for k, v in sorted(inventories_by_collection.items())},
        "collection_roots": roots,
        "summary": summary,
        "eligible_assets": eligible,
        "excluded_assets": excluded,
    }


def select_from_manifest(manifest_path, documents, page_filters, *, root=ROOT):
    """Load executable Asset targets from a reconciled extraction selection manifest."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or "eligible_assets" not in manifest:
        raise ValueError("Invalid extraction selection manifest")
    requested = set(documents)
    matches, filter_matches, seen_keys, seen_shas = set(), set(), set(), set()
    assets, asset_roots = [], {}
    excluded = [
        {"asset_key": item["asset_key"], "record_id": item["record_id"], "asset_sha256": item.get("sha256"),
         "path": item["asset_path"], "reason": "|".join(item.get("exclusion_reasons", ["excluded"]))}
        for item in manifest.get("excluded_assets", [])
    ]
    root_overrides = {}
    if raw_env := os.getenv("MEDQUERY_COLLECTION_ROOTS"):
        parsed_env = json.loads(raw_env)
        if not isinstance(parsed_env, dict):
            raise ValueError("MEDQUERY_COLLECTION_ROOTS must be a JSON object mapping collection codes to paths")
        root_overrides = {str(k): str(v) for k, v in parsed_env.items()}

    eligible_items = list(manifest["eligible_assets"])
    eligible_keys = {item["asset_key"] for item in eligible_items}
    for item in eligible_items:
        key = item["asset_key"]
        record_id = item["record_id"]
        aliases = {record_id, key}
        if requested and not (aliases & requested):
            continue
        matches.update(aliases & requested)
        if key in seen_keys:
            raise ValueError(f"Manifest repeats asset key {key}")
        seen_keys.add(key)
        sha = item["sha256"]
        if sha in seen_shas:
            raise ValueError(f"Manifest repeats eligible SHA-256 {sha}")
        seen_shas.add(sha)
        if item.get("document_use_status") in {"restricted", "prohibited"} or (
            item.get("asset_use_status") in {"restricted", "prohibited"}
        ) or item.get("phase2_classification") == "unresolved":
            raise ValueError(f"Ineligible asset present in manifest eligible_assets: {key}")

        filters = []
        for i, spec in enumerate(page_filters):
            target, separator, numbers = spec.partition(":")
            if not separator:
                filters.append((i, target))
            elif target == key or (target not in eligible_keys and target in aliases):
                filters.append((i, numbers))
        filter_matches.update(i for i, _ in filters)
        if page_filters and not filters:
            continue

        count = item["pages"]
        pages = set()
        if filters:
            for _, spec in filters:
                pages.update(page_numbers(spec, count))
        else:
            pages = set(range(1, count + 1))

        asset_parts(item["asset_path"])
        raw_root = root_overrides.get(item["collection"], item["collection_root"])
        col_root = Path(raw_root)
        resolved_root = col_root if col_root.is_absolute() else (root / col_root).resolve()
        asset_roots[key] = resolved_root
        assets.append(Asset(
            record_id, key, item["asset_path"], sha, item["bytes"], count, item["language"],
            item["asset_role"], item.get("attachment_type"), item.get("revision_key"),
            item.get("acquisition_url"), item.get("document_use_status", "unknown"),
            item.get("asset_use_status", "unknown"), tuple(sorted(pages)),
        ))

    if requested - matches:
        raise ValueError(f"Unknown or ineligible document IDs: {sorted(requested - matches)}")
    if len(filter_matches) != len(page_filters):
        raise ValueError("A physical page filter did not match a selected asset")
    if not assets:
        raise ValueError("No eligible assets matched the selection")
    return sorted(assets, key=lambda a: (a.record_id, a.key)), sorted(excluded, key=lambda a: a["asset_key"]), asset_roots


def verify_manifest_assets(manifest_path, *, root=ROOT, check_pdf_pages=True):
    """Verify every eligible asset exists under its collection root and matches bytes, SHA-256, and pages."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    eligible = manifest.get("eligible_assets", [])
    eligible_keys = {item["asset_key"] for item in eligible}
    seen_shas = set()
    issues = []
    verified_bytes = 0
    verified_pages = 0

    pymupdf = None
    if check_pdf_pages:
        import pymupdf as _pymupdf
        pymupdf = _pymupdf

    for item in eligible:
        key = item["asset_key"]
        col_root = Path(item["collection_root"])
        resolved_root = col_root if col_root.is_absolute() else (root / col_root).resolve()
        asset_parts(item["asset_path"])
        pdf_path = resolved_root / item["asset_path"]
        if not pdf_path.is_file():
            issues.append({"asset_key": key, "reason": "missing_file", "path": str(pdf_path)})
            continue
        actual_bytes = pdf_path.stat().st_size
        if actual_bytes != item["bytes"]:
            issues.append({"asset_key": key, "reason": "byte_size_mismatch",
                           "expected": item["bytes"], "actual": actual_bytes})
        actual_sha = sha256(pdf_path.read_bytes()).hexdigest()
        if actual_sha != item["sha256"]:
            issues.append({"asset_key": key, "reason": "sha256_mismatch",
                           "expected": item["sha256"], "actual": actual_sha})
        if actual_sha in seen_shas:
            issues.append({"asset_key": key, "reason": "duplicate_eligible_sha256", "sha256": actual_sha})
        seen_shas.add(actual_sha)
        if item.get("asset_role") != "primary" and item.get("primary_asset_key") not in eligible_keys:
            issues.append({"asset_key": key, "reason": "orphaned_annex_missing_primary"})
        if check_pdf_pages and pymupdf is not None:
            with pymupdf.open(pdf_path) as doc:
                if doc.is_encrypted or doc.page_count != item["pages"]:
                    issues.append({"asset_key": key, "reason": "page_count_or_encryption_mismatch",
                                   "expected_pages": item["pages"], "actual_pages": doc.page_count,
                                   "encrypted": doc.is_encrypted})
        verified_bytes += actual_bytes
        verified_pages += item["pages"]

    return {
        "eligible_assets": len(eligible),
        "unique_sha256": len(seen_shas),
        "verified_bytes": verified_bytes,
        "verified_pages": verified_pages,
        "issues": issues,
        "ok": len(issues) == 0,
    }


def rebase_selection_manifest(
    manifest_path,
    output_path,
    collection_roots,
    *,
    root=ROOT,
    inventories=None,
    exclude_asset_keys=None,
    exclusion_reason="excluded_from_hpc_transfer",
):
    """Create a portable copy of an extraction selection manifest with rebased collection roots."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or "eligible_assets" not in manifest:
        raise ValueError("Invalid extraction selection manifest")

    exclude_set = set(exclude_asset_keys or ())
    new_roots = {col: str(collection_roots.get(col, path)) for col, path in manifest.get("collection_roots", {}).items()}
    for col, val in collection_roots.items():
        new_roots[col] = str(val)

    eligible = []
    excluded = [
        {**item, "collection_root": new_roots.get(item["collection"], item.get("collection_root", ""))}
        for item in manifest.get("excluded_assets", [])
    ]

    for item in manifest["eligible_assets"]:
        col = item["collection"]
        updated = {
            **item,
            "collection_root": new_roots.get(col, item["collection_root"]),
        }
        if inventories and col in inventories:
            updated["inventory_path"] = str(inventories[col])
        if item["asset_key"] in exclude_set or item["record_id"] in exclude_set:
            excluded.append({
                **{k: updated[k] for k in ("asset_key", "record_id", "collection", "collection_root", "asset_path",
                                            "asset_role", "attachment_type", "sha256", "bytes", "pages", "language",
                                            "v1_scope_bucket", "phase2_classification", "target_categories")},
                "exclusion_reasons": [exclusion_reason],
            })
        else:
            eligible.append(updated)

    # Maintain annex-bundle invariant if a primary was excluded.
    eligible_keys = {item["asset_key"] for item in eligible}
    orphaned = [item for item in eligible if item["asset_role"] != "primary" and item["primary_asset_key"] not in eligible_keys]
    if orphaned:
        orphan_keys = {item["asset_key"] for item in orphaned}
        eligible = [item for item in eligible if item["asset_key"] not in orphan_keys]
        for item in orphaned:
            excluded.append({
                **{k: item[k] for k in ("asset_key", "record_id", "collection", "collection_root", "asset_path",
                                         "asset_role", "attachment_type", "sha256", "bytes", "pages", "language",
                                         "v1_scope_bucket", "phase2_classification", "target_categories")},
                "exclusion_reasons": ["primary_asset_excluded"],
            })

    by_collection = {}
    for col in sorted(new_roots):
        col_el = [item for item in eligible if item["collection"] == col]
        col_ex = [item for item in excluded if item["collection"] == col]
        by_collection[col] = {
            "eligible_assets": len(col_el),
            "eligible_pages": sum(item["pages"] for item in col_el),
            "excluded_assets": len(col_ex),
            "excluded_pages": sum(item["pages"] for item in col_ex),
        }

    rebased = {
        **manifest,
        "collection_roots": new_roots,
        "inventories": dict(inventories) if inventories is not None else manifest.get("inventories", {}),
        "summary": {
            **manifest.get("summary", {}),
            "executable_eligible_assets": len(eligible),
            "executable_eligible_pages": sum(item["pages"] for item in eligible),
            "unique_eligible_sha256": len({item["sha256"] for item in eligible}),
            "excluded_or_held_assets": len(excluded),
            "excluded_or_held_pages": sum(item["pages"] for item in excluded),
            "by_collection": by_collection,
            "eligible_by_bucket": dict(Counter(item["v1_scope_bucket"] for item in eligible)),
            "eligible_by_language": dict(Counter(item["language"] for item in eligible)),
            "excluded_by_primary_reason": dict(Counter(item["exclusion_reasons"][0] for item in excluded)),
        },
        "eligible_assets": eligible,
        "excluded_assets": excluded,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rebased, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rebased
