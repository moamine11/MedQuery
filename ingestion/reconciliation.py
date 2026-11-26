"""Evidence-based identity suggestions and deterministic review entries."""

import hashlib
import json
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations

from .metadata import text_key


def fingerprint(*parts):
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verified_ids(record, *, include_primary_asset=False):
    items = list(record.get("identifiers", []))
    if include_primary_asset:
        primary = next(asset for asset in record["assets"] if asset["role"] == "primary")
        items.extend(primary.get("identifiers", []))
    return {(item.get("namespace") or record["issuer"].get("key") or
             text_key(record["issuer"]["name"]), item["scheme"], item["value"], item["scope"])
            for item in items if item.get("verified")}


def identity(record):
    stable = {entry for entry in verified_ids(record)
              if record["issuer"].get("key") or any(
                  item.get("namespace") == entry[0] and item.get("verified")
                  and (item["scheme"], item["value"], item["scope"]) == entry[1:]
                  for item in record.get("identifiers", [])
              )}
    work = next((entry for entry in sorted(stable) if entry[-1] == "document"), None)
    if work:
        return f"work:{fingerprint(*work, record['language'])}"
    if record.get("identity_key") and record.get("identity_evidence") and record["issuer"].get("key"):
        return f"work:{fingerprint(record['issuer']['key'], record['identity_key'], record['language'])}"
    return None


def revision(record, work):
    if not work:
        return None
    if record.get("revision_key") and record.get("revision_evidence"):
        return f"revision:{fingerprint(work, record['revision_key'])}"
    ids = sorted(entry for entry in verified_ids(record) if entry[-1] == "revision")
    return f"revision:{fingerprint(work, *ids[0])}" if ids else None


def relation_matches(entries):
    by_record = {f"{entry['collection']}:{entry['metadata']['record_id']}": entry for entry in entries}
    edges = {}
    for entry in entries:
        record = entry["metadata"]
        source = f"{entry['collection']}:{record['record_id']}"
        for relation in record.get("relations", []):
            target = relation["target"]
            if target not in by_record and f"{entry['collection']}:{target}" in by_record:
                target = f"{entry['collection']}:{target}"
            other = by_record.get(target)
            kind = relation["type"]
            if relation["state"] != "verified":
                yield entry, other, "declared_relationship", "Verify the relationship and target edition.", kind, relation["target"]
                continue
            invalid = (source == target or not other or
                       (kind == "translation_of" and record["language"] == other["metadata"]["language"]) or
                       (kind == "replaces" and (not entry["identity_key"] or
                        entry["identity_key"] != other["identity_key"] or
                        (entry["revision_key"] and entry["revision_key"] == other["revision_key"]))))
            if invalid:
                yield entry, other, "unresolved_verified_relationship", "Resolve the target and check language/edition identity.", kind, relation["target"]
            else:
                yield entry, other, "verified_relationship", "Retain the supplied relationship and its evidence.", kind, relation["target"]
                if kind in {"replaces", "translation_of", "adaptation_of"}:
                    edges.setdefault((source, kind), set()).add(target)
    for (source, kind), targets in edges.items():
        if len(targets) > 1:
            yield by_record[source], None, "conflicting_relationship_targets", "Select a single verified target.", kind, None
        stack = list(targets)
        seen = set()
        while stack:
            target = stack.pop()
            if target == source:
                yield by_record[source], None, "relationship_cycle", "Remove the contradictory cycle.", kind, None
                break
            if target not in seen:
                seen.add(target)
                stack.extend(edges.get((target, kind), ()))


def reconcile(entries):
    review = []
    ordered = sorted(entries, key=lambda entry: (entry["collection"], entry["metadata"]["record_id"], entry["source"]))
    def ref(entry):
        return {"collection": entry["collection"], "record_id": entry["metadata"]["record_id"], "source": entry["source"]}

    def add(kind, members, action, **evidence):
        review.append({"kind": kind, "members": members, "recommended_action": action, "evidence": evidence})

    hashes = defaultdict(list)
    primary_hash = {}
    for entry in ordered:
        record = entry["metadata"]
        for asset in record["assets"]:
            path = asset["path"]
            measured = entry["observations"].get(path)
            if not measured:
                continue
            member = {**ref(entry), "path": path, "role": asset["role"]}
            hashes[measured["sha256"]].append(member)
            if asset["role"] == "primary":
                primary_hash[(entry["collection"], record["record_id"], entry["source"])] = measured["sha256"]
            if asset["role"] != "primary":
                add("attached_asset", [member],
                    "Confirm the attachment scope and retain its own provenance and permissions.",
                    attachment_type=asset.get("attachment_type", asset["role"]))
    for digest, members in sorted(hashes.items()):
        if len(members) > 1:
            add("exact_bytes", members, "Reuse byte identity after checking edition, role, rights and publication claims.", sha256=digest)

    eligible = [entry for entry in ordered if entry["validation_state"] == "valid"]
    for entry, other, kind, action, relation, target in relation_matches(eligible):
        add(kind, [ref(entry)] + ([ref(other)] if other else []), action, relation=relation, target=target)

    for left, right in combinations(eligible, 2):
        a, b = left["metadata"], right["metadata"]
        a_title, b_title = (text_key(item.get("canonical_title", item["title"])) for item in (a, b))
        a_issuer, b_issuer = (item["issuer"].get("key") or text_key(item["issuer"]["name"]) for item in (a, b))
        same_issuer = a_issuer == b_issuer
        same_language = a["language"] == b["language"]
        same_title = bool(a_title and a_title == b_title)
        same_work = bool(left["identity_key"] and left["identity_key"] == right["identity_key"])
        a_bytes = primary_hash.get((left["collection"], a["record_id"], left["source"]))
        b_bytes = primary_hash.get((right["collection"], b["record_id"], right["source"]))
        same_bytes = bool(a_bytes and a_bytes == b_bytes)
        shared_ids = verified_ids(a, include_primary_asset=True) & verified_ids(b, include_primary_asset=True)
        same_revision = bool(left["revision_key"] and left["revision_key"] == right["revision_key"])
        a_edition, b_edition = (text_key(item.get("edition", "")) for item in (a, b))
        different_edition = bool(a_edition and b_edition and a_edition != b_edition)
        if same_bytes or shared_ids or same_work or (same_title and same_issuer):
            if (same_bytes and (not same_issuer or not same_language or different_edition or
                               a["document_type"] != b["document_type"] or a["authority"] != b["authority"])) or (
                               shared_ids and (not same_issuer or not same_language)):
                kind = "conflicting_identity_claims"
            elif not same_language:
                kind = "possible_translation"
            elif different_edition or (same_work and left["revision_key"] and right["revision_key"] and not same_revision):
                kind = "different_edition_candidate"
            elif not same_bytes and (same_revision or (same_title and same_issuer and a_edition and a_edition == b_edition)):
                kind = "alternate_rendition_candidate"
            elif same_bytes and same_issuer and same_title and set(a["categories"]) != set(b["categories"]):
                kind = "cross_category_candidate"
            else:
                kind = "bibliographic_match"
            add(kind, [ref(left), ref(right)], "Review publisher evidence and retain both local IDs before reconciling.",
                same_bytes=same_bytes, same_issuer=same_issuer, same_title=same_title,
                same_language=same_language, same_revision=same_revision,
                verified_identifiers=[list(value) for value in sorted(shared_ids)],
                candidate_categories=sorted(set(a["categories"]) | set(b["categories"])))
        elif same_language and (same_title or (
            same_issuer and min(len(a_title), len(b_title)) >= 25
            and SequenceMatcher(None, a_title, b_title, autojunk=False).ratio() >= 0.88
        )):
            add("similar_bibliography", [ref(left), ref(right)],
                "Compare issuer, identifiers, scope and edition before linking publications.",
                same_issuer=same_issuer, same_title=same_title)
    return review
