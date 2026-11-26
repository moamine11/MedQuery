"""Normalize metadata and validate the canonical and semantic contracts."""

import copy
import json
import re
import unicodedata
from pathlib import Path

import langcodes
import pycountry
from jsonschema import Draft202012Validator, FormatChecker

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs/ingestion/metadata.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())
CLINICAL_TYPES = {"clinical_guideline", "clinical_protocol", "recommendation_sheet", "consensus"}
REGULATORY_TYPES = {"ministerial_instruction", "regulatory_text", "administrative_text"}
CONTEXT_TYPES = {"policy_brief", "implementation_guide", "public_health_guidance", "technology_appraisal",
                 "executive_summary", "correction", "supplement", "formulary"}


def wording(value):
    return " ".join(unicodedata.normalize("NFC", value).split())


def text_key(value):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", value).casefold()))


def code(value):
    return "_".join(text_key(value).split())


def identifier_value(scheme, value):
    value = wording(value)
    if scheme == "doi":
        return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value, flags=re.I).lower()
    if scheme in {"isbn", "isbn10", "isbn13"}:
        return re.sub(r"[\s-]+", "", re.sub(r"^isbn(?:-1[03])?\s*:?\s*", "", value, flags=re.I)).upper()
    return value


def isbn_valid(value):
    if re.fullmatch(r"\d{9}[\dX]", value, flags=re.ASCII):
        return sum((10 - i) * (10 if c == "X" else int(c)) for i, c in enumerate(value)) % 11 == 0
    if re.fullmatch(r"97[89]\d{10}", value, flags=re.ASCII):
        return sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(value)) % 10 == 0
    return False


def classify(record):
    authority, doc_type = record["authority"], record["document_type"]
    issuer = record["issuer"]
    if authority == "educational" or doc_type == "educational_material":
        return "educational"
    if doc_type in REGULATORY_TYPES:
        return "regulatory_administrative"
    if authority == "contextual" or doc_type in CONTEXT_TYPES:
        return "contextual"
    if doc_type in CLINICAL_TYPES:
        if authority == "official_national" and issuer.get("country") == "DZ" and issuer["kind"] in {"government", "public_agency"}:
            return "official_algerian_clinical_candidate"
        if authority == "international_reference" and issuer["kind"] != "unknown":
            return "international_clinical_candidate"
    return "unresolved"


def validate(raw):
    issues = []
    def issue(code, field, detail, severity="error"):
        issues.append({"code": code, "severity": severity, "field": field, "detail": detail})

    for error in sorted(VALIDATOR.iter_errors(raw), key=lambda e: (list(map(str, e.absolute_path)), e.message)):
        issue("schema_validation", "/".join(map(str, error.absolute_path)), error.message)
    if issues:
        return None, issues

    record = copy.deepcopy(raw)
    for field in ("title", "canonical_title", "edition"):
        if field in record:
            record[field] = wording(record[field])
    record["issuer"]["name"] = wording(record["issuer"]["name"])
    record.setdefault("intake_review", "not_requested")
    for tag in [record["language"], *record.get("content_languages", [])]:
        if not langcodes.tag_is_valid(tag):
            issue("invalid_language", "language", f"Unregistered or invalid language tag: {tag}")
    scope = record.get("applicability", {})
    for country in [record["issuer"].get("country"), *scope.get("countries", [])]:
        if country and not pycountry.countries.get(alpha_2=country):
            issue("invalid_country", "applicability/issuer", f"Unregistered country code: {country}")
    for i, rule in enumerate(scope.get("rules", [])):
        age = rule.get("age_days", {})
        if age.get("min_inclusive", 0) >= age.get("max_exclusive", 2147483648):
            issue("invalid_age_range", f"applicability/rules/{i}", "Age bounds must define a nonempty range.")
    if record.get("effective_from") and record.get("effective_until"):
        if record["effective_until"] <= record["effective_from"]:
            issue("invalid_effective_window", "effective_until", "The exclusive end must follow the start.")
    if record.get("publication_date", "0001")[:4] == "0000":
        issue("invalid_publication_date", "publication_date", "Year zero is invalid.")
    paths = [asset["path"] for asset in record["assets"]]
    if len(paths) != len(set(paths)):
        issue("duplicate_asset_path", "assets", "The same path is listed more than once in this record.")
    for i, asset in enumerate(record["assets"]):
        attachment = asset.get("attachment_type")
        if attachment and (asset["role"] == "primary" or
                           (asset["role"] == "alternate") != (attachment == "alternate_rendition")):
            issue("incompatible_attachment_role", f"assets/{i}/attachment_type",
                  "Corrections and supplements attach as annexes; alternate renditions attach as alternates.")

    for prefix, node in [("", record), *((f"assets/{i}/", a) for i, a in enumerate(record["assets"]))]:
        reuse = node.setdefault("reuse", {})
        for status in ("use_status", "external_embedding_status"):
            reuse.setdefault(status, "unknown")
            if reuse.get(status) == "allowed" and not (reuse.get("evidence_url") or reuse.get("evidence_note")):
                issue("missing_permission_evidence", f"{prefix}reuse/{status}", "An allowed claim requires publisher permission evidence.")
        for i, identifier in enumerate(node.get("identifiers", [])):
            scheme = code(identifier["scheme"])
            value = identifier_value(scheme, identifier["value"])
            identifier.update(scheme=scheme, value=value)
            field = f"{prefix}identifiers/{i}"
            if (not scheme or not value or
                (scheme == "doi" and not re.fullmatch(r"10\.\d{4,9}/[^\s<>]+", value)) or
                (scheme in {"isbn", "isbn10", "isbn13"} and
                 (not isbn_valid(value) or (scheme != "isbn" and len(value) != int(scheme[4:]))))):
                issue("invalid_identifier", field, "Identifier value is invalid for its scheme.")
            if identifier.get("verified") and scheme == "as_recorded":
                issue("invalid_identifier", field, "Unparsed source values cannot be marked verified.")
        if prefix:
            acquisition = node.get("acquisition", {})
            if not (acquisition.get("requested_url") or acquisition.get("resolved_url")):
                issue("missing_acquisition_route", f"{prefix}acquisition",
                      "No documented acquisition URL supplied.", "warning")
    for i, relation in enumerate(record.get("relations", [])):
        if relation["target"] == record["record_id"]:
            issue("self_relation", f"relations/{i}", "Relationship points to the same record.")
        if relation["state"] == "verified" and not relation.get("note"):
            issue("missing_relation_evidence", f"relations/{i}", "A verified relation requires an evidence note.")
    for field in ("publication_date", "official_url"):
        if field not in record:
            issue("missing_metadata", field, "Value is absent.", "warning")
    if record["language"] == "und" or record["issuer"]["kind"] == "unknown" or classify(record) == "unresolved":
        issue("unresolved_classification", "authority", "Review language, issuer or source classification.", "warning")
    return record, issues
