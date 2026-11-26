"""Read-only orchestration for inspection, validation and reconciliation."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .intake import IntakeError, discover, observe_pdf
from .metadata import classify, validate
from .reconciliation import identity, reconcile, relation_matches, revision


def intake(root, collection, adapter, metadata, pdfinfo, *, workers=4, references=(), matching=False):
    issues = []
    records = []
    claimed = set()
    active_paths = set(discover(root))

    def issue(code, detail, *, severity="error", source=None, record_id=None, field=None):
        issues.append({key: value for key, value in {
            "code": code, "severity": severity, "detail": detail, "source": source,
            "record_id": record_id, "field": field,
        }.items() if value is not None})

    for source, raw in adapter(root, metadata, collection):
        if isinstance(raw, dict) and "_parse_error" in raw:
            issue("metadata_parse_error", raw["_parse_error"], source=source)
            records.append({"collection": collection, "source": source, "validation_state": "invalid"})
            continue
        normalized, diagnostics = validate(raw)
        record_id = raw.get("record_id") if isinstance(raw, dict) else None
        items = raw.get("assets") if isinstance(raw, dict) else None
        assets = [a for a in items if isinstance(a, dict) and isinstance(a.get("path"), str)] if isinstance(items, list) else []
        claimed.update(a["path"] for a in assets)
        for diagnostic in diagnostics:
            issue(**diagnostic, source=source, record_id=record_id)
        entry = {
            "collection": collection, "source": source, "original_metadata": raw,
            "validation_state": "invalid" if any(d["severity"] == "error" for d in diagnostics) else "valid",
            "observations": {}, "treatment_guidance_eligible": False, "clinical_review": "not_assessed",
            "classification": classify(normalized) if normalized else "unresolved",
            "identity_key": identity(normalized) if normalized else None,
        }
        entry["revision_key"] = revision(normalized, entry["identity_key"]) if normalized else None
        if normalized:
            entry["metadata"] = normalized
        records.append(entry)

    def probe(path):
        try:
            measurements, warnings = observe_pdf(root, path, pdfinfo)
            return path, {"observed": measurements, "warnings": warnings}
        except IntakeError as error:
            result = {"error": {"code": error.code, "detail": str(error)}}
            if error.observed:
                result["observed"] = error.observed
            return path, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        observations = dict(pool.map(probe, sorted(active_paths | claimed)))

    for path, value in observations.items():
        if "error" in value:
            issue(**value["error"], field=path)
        for warning in value.get("warnings", []):
            issue(warning, "PDF parser reported a structure warning; page count was readable.",
                  severity="warning", field=path)
        if adapter is not no_metadata and path in active_paths and path not in claimed:
            issue("unlisted_pdf", "Discovered PDF has no manifest record.", severity="warning", field=path)

    key_counts = Counter(entry["metadata"]["record_id"] for entry in records if "metadata" in entry)
    for entry in records:
        raw = entry.get("original_metadata")
        if not isinstance(raw, dict) or not isinstance(raw.get("assets"), list):
            continue
        record_id = raw.get("record_id")
        if isinstance(record_id, str) and key_counts[record_id] > 1:
            entry["validation_state"] = "invalid"
            issue("duplicate_record_id", "Record ID is repeated in the local collection.",
                  source=entry["source"], record_id=record_id)
        for asset in raw["assets"]:
            if not isinstance(asset, dict) or not isinstance(asset.get("path"), str):
                continue
            path = asset["path"]
            value = observations[path]
            if "error" in value:
                entry["validation_state"] = "invalid"
                issue(**value["error"], source=entry["source"], field=path, record_id=record_id)
            actual = value.get("observed")
            if not actual:
                continue
            entry["observations"][path] = actual
            expected = asset.get("observed")
            if isinstance(expected, dict):
                for name in sorted(expected.keys() & actual.keys()):
                    if expected[name] != actual[name]:
                        issue("observation_mismatch",
                              f"Expected {expected[name]!r}, measured {actual[name]!r}.",
                              source=entry["source"], record_id=record_id, field=f"{path}:{name}")
                        entry["validation_state"] = "invalid"

    candidates = [entry for entry in records if entry.get("metadata")]
    usable = [entry for entry in candidates if entry["validation_state"] == "valid"]
    local_entries = {id(entry) for entry in usable}
    for entry, _, code, action, kind, _ in relation_matches([*usable, *references]):
        if code not in {"declared_relationship", "verified_relationship"} and id(entry) in local_entries:
            entry["validation_state"] = "invalid"
            issue(code, action, source=entry["source"], record_id=entry["metadata"]["record_id"], field=kind)

    queue = reconcile([*candidates, *references]) if matching else []
    validated = [entry for entry in records if entry["validation_state"] == "valid"]
    return {
        "inventory": records,
        "files": [{"path": path, "discovered": path in active_paths, **value} for path, value in observations.items()],
        "issues": issues,
        "review_queue": queue,
        "summary": {
            "collection": collection, "records": len(records), "valid_records": len(validated),
            "invalid_records": len(records) - len(validated), "discovered_pdfs": len(active_paths),
            "readable_pdfs": sum("error" not in value for value in observations.values()),
            "page_counts_observed": sum("pages" in value.get("observed", {}) for value in observations.values()),
            "physical_pages": sum(value.get("observed", {}).get("pages", 0) for value in observations.values()),
            "classifications": dict(Counter(entry["classification"] for entry in validated)),
            "issues": dict(Counter(item["code"] for item in issues)),
            "error_count": sum(item["severity"] == "error" for item in issues),
            "review_queue": dict(Counter(item["kind"] for item in queue)),
        },
    }


def no_metadata(root, metadata, category):
    return ()
