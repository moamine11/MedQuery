"""Metadata adapters emit (source locator, canonical candidate) pairs."""

import csv
import io
import json
from collections import defaultdict
from typing import Iterable, Protocol

from .intake import IntakeError, open_local
from .metadata import code
from .treatment_protocols import treatment_protocols

CATEGORIES = {"clinical_protocols_ministry_directives", "treatment_protocols", "therapeutic_guidelines",
              "pharmaceutical_formularies", "who_guidelines"}
LANGUAGES = {"french": "fr", "français": "fr", "english": "en", "arabic": "ar", "العربية": "ar"}
TYPES = {"guideline": "clinical_guideline", "protocol": "clinical_protocol",
         "administrative": "administrative_text", "education": "educational_material"}


class Adapter(Protocol):
    def __call__(self, root, metadata, category) -> Iterable[tuple[str, dict]]:
        """Yield the source locator and candidate for each source row."""


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def canonical(root, metadata, category):
    with metadata.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield f"line:{number}", json.loads(line.decode("utf-8-sig" if number == 1 else "utf-8"),
                                                   object_pairs_hook=pairs, parse_constant=reject_constant)
            except ValueError as error:
                yield f"line:{number}", {"_parse_error": str(error)}


def reject_constant(value):
    raise ValueError(f"Invalid JSON number: {value}")


def csv_rows(stream, required):
    rows = csv.DictReader(stream)
    headers = rows.fieldnames or []
    if len(headers) != len(set(headers)) or not set(required).issubset(headers):
        raise ValueError("CSV needs unique headers including " + ", ".join(sorted(required)))
    for row in rows:
        extra = row.pop(None, None)
        if extra or any(value is None for value in row.values()):
            yield rows.line_num, {"_parse_error": "CSV row has the wrong number of fields."}
            continue
        yield rows.line_num, row


def language_tag(value):
    lowered = value.strip().casefold()
    return LANGUAGES.get(lowered, lowered.replace("_", "-")) or "und"


def generic_csv(root, metadata, category):
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        for number, raw in csv_rows(stream, {"record_id", "path", "title", "issuer"}):
            location = f"row:{number}"
            if "_parse_error" in raw:
                yield location, raw
                continue
            row = {key: value.strip() for key, value in raw.items()}
            get = lambda field: row.get(field, "")
            kind = code(get("document_type"))
            categories = [code(value) for value in get("categories").split("|") if value.strip()]
            record = {
                "schema_version": 1, "record_id": get("record_id"), "title": get("title"),
                "language": language_tag(get("language")), "categories": sorted(set(categories or [category])),
                "issuer": {"name": get("issuer"), "kind": code(get("issuer_kind")) or "unknown"},
                "document_type": TYPES.get(kind, kind or "unknown"),
                "authority": code(get("authority")) or "unknown",
                "assets": [{"path": get("path"), "role": "primary"}],
                "adapter_context": {"original_csv_row": raw},
            }
            if get("issuer_country"):
                record["issuer"]["country"] = get("issuer_country").upper()
            if get("content_languages"):
                record["content_languages"] = [language_tag(value) for value in get("content_languages").split("|")]
            for field in ("edition", "publication_date", "effective_from", "effective_until", "official_url"):
                if get(field):
                    record[field] = get(field)
            if get("acquisition_url"):
                record["assets"][0]["acquisition"] = {"resolved_url": get("acquisition_url")}
            if get("doi"):
                record["identifiers"] = [{"scheme": "doi", "value": get("doi"), "scope": "asset"}]
            if get("license_note"):
                record["assets"][0]["reuse"] = {"license_note": get("license_note")}
            yield location, record


def local_csv(root, name, required):
    with open_local(root, ["metadata", name]) as stream:
        yield from csv_rows(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""), required)


def optional_local_csv(root, name, required):
    try:
        yield from local_csv(root, name, required)
    except IntakeError as error:
        if error.code != "missing_file":
            raise


ISSUER_ALIASES = {
    "haute autorité de santé": ("Haute Autorité de Santé", "public_agency", "FR"),
    "haute autorité de santé (has)": ("Haute Autorité de Santé", "public_agency", "FR"),
    "national institute for health and care excellence": ("National Institute for Health and Care Excellence", "public_agency", "GB"),
    "institut national du cancer": ("Institut National du Cancer", "public_agency", "FR"),
    "centers for disease control and prevention (cdc), mmwr": ("Centers for Disease Control and Prevention (CDC)", "public_agency", "US"),
    "u.s. preventive services task force": ("U.S. Preventive Services Task Force", "public_agency", "US"),
    "u.s. department of veterans affairs and department of defense": ("U.S. Department of Veterans Affairs and Department of Defense", "government", "US"),
    "u.s. department of health and human services panel on antiretroviral guidelines for adults and adolescents (nih oarac working group)": (
        "U.S. Department of Health and Human Services (NIH OARAC)", "government", "US"),
    "world health organization": ("World Health Organization", "intergovernmental", ""),
    "world health organization, regional office for the eastern mediterranean": (
        "World Health Organization, Regional Office for the Eastern Mediterranean", "intergovernmental", ""),
    "ministère de la santé (algérie), dgpps": ("Ministère de la Santé (Algérie), DGPPS", "government", "DZ"),
    "ministère de la santé (algérie), dgssrh, with unicef algeria": (
        "Ministère de la Santé (Algérie), DGSSRH, with UNICEF Algeria", "government", "DZ"),
    "ministère de la santé / dgssrh (cover)": ("Ministère de la Santé (Algérie), DGSSRH", "government", "DZ"),
    "ministère de la santé (cover)": ("Ministère de la Santé (Algérie)", "government", "DZ"),
    "european society of cardiology": ("European Society of Cardiology", "professional_body", ""),
    "european society of cardiology (esc)": ("European Society of Cardiology", "professional_body", ""),
    "european society of human reproduction and embryology": ("European Society of Human Reproduction and Embryology", "professional_body", ""),
    "european thyroid association": ("European Thyroid Association", "professional_body", ""),
    "kidney disease: improving global outcomes": ("Kidney Disease: Improving Global Outcomes (KDIGO)", "professional_body", ""),
    "kidney disease: improving global outcomes (kdigo)": ("Kidney Disease: Improving Global Outcomes (KDIGO)", "professional_body", ""),
    "infectious diseases society of america": ("Infectious Diseases Society of America", "professional_body", "US"),
    "world federation of hemophilia": ("World Federation of Hemophilia", "professional_body", ""),
    "resuscitation council uk": ("Resuscitation Council UK", "professional_body", "GB"),
    "uk renal association": ("UK Renal Association", "professional_body", "GB"),
    "global initiative for asthma (gina)": ("Global Initiative for Asthma (GINA)", "professional_body", ""),
    "pituitary society": ("Pituitary Society", "professional_body", ""),
    "korean society of nephrology": ("Korean Society of Nephrology", "professional_body", "KR"),
    "spilf and splf": ("SPILF and SPLF", "professional_body", "FR"),
    "sfar and collaborating societies": ("SFAR and collaborating societies", "professional_body", "FR"),
    "ecmm; isham; asm": ("ECMM, ISHAM and ASM", "professional_body", ""),
    "ada, easd, jbds, aace and diabetes technology society": ("ADA, EASD, JBDS, AACE and Diabetes Technology Society", "professional_body", ""),
    "ada/easd; diabetologia, springer nature": ("ADA/EASD", "professional_body", ""),
    "société algérienne de pneumophtisiologie": ("Société Algérienne de Pneumophtisiologie", "professional_body", "DZ"),
}
TG_ROLES = {**TYPES, "translation": "clinical_guideline", "intervention_guide": "implementation_guide"}


def tg_asset(row, role, manual):
    asset = {"path": row["relative_path"], "role": role, "label": row["title"],
             "adapter_context": {"original_csv_row": row}}
    values = {"requested_url": "", "resolved_url": row.get("acquisition_url"),
              "method": row.get("origin"), "acquired_on": row.get("acquisition_date"),
              "evidence_note": row.get("access_status")}
    asset["acquisition"] = {key: value for key, value in values.items() if value}
    asset["reuse"] = {key: value for key, value in {
        "license_note": row.get("license"), "evidence_url": row.get("rights_evidence_url"),
    }.items() if value}
    if role == "annex":
        asset["attachment_type"] = code(row["role"])
    if manual:
        asset["adapter_context"]["manual_intake"] = manual
    if row.get("sha256") or row.get("byte_count") or row.get("page_count"):
        observed = {"media_type": "application/pdf"}
        for field, source in (("sha256", "sha256"), ("bytes", "byte_count"), ("pages", "page_count")):
            value = row.get(source, "")
            if value:
                observed[field] = int(value) if field != "sha256" and value.isascii() and value.isdecimal() else value
        asset["observed"] = observed
    if row.get("doi"):
        asset["identifiers"] = [{"scheme": "doi", "value": row["doi"], "scope": "asset"}]
    return asset


def tg_record(row, assets, topics, category):
    issuer = row["issuing_organization"]
    name, kind, country = ISSUER_ALIASES.get(issuer.casefold(), (issuer, "unknown", ""))
    role = code(row.get("role", ""))
    raw_type = TG_ROLES.get(role, role or "unknown")
    if row["title"].casefold().startswith("instruction"):
        raw_type = "ministerial_instruction"
    elif row["title"].casefold().startswith("protocoles"):
        raw_type = "clinical_protocol"
    authority = "unknown"
    if kind in {"public_agency", "intergovernmental"}:
        authority = "international_reference"
    elif kind == "government":
        authority = "official_national" if country == "DZ" else "international_reference"
    elif kind == "professional_body" and country != "DZ":
        authority = "international_reference"
    if raw_type in {"policy_brief", "implementation_guide"}:
        authority = "contextual"
    record = {
        "schema_version": 1, "record_id": row["document_id"], "title": row["title"],
        "categories": [category], "language": language_tag(row["language"]),
        "issuer": {"name": name, "kind": kind}, "document_type": raw_type, "authority": authority,
        "intake_review": "hold" if row.get("rag_review_state") == "permission_required" or row.get("origin") == "supplied_local" else "pending",
        "assets": assets, "applicability": {"geography": "unknown", "population": "unknown"},
        "evidence": [{"field": "catalogue_claims", "locator": f"metadata/documents.csv#{row['document_id']}"}],
    }
    if country:
        record["issuer"]["country"] = country
    for field, column in (("scope_text", "applicability"), ("population_text", "population")):
        if row.get(column):
            record["applicability"][field] = row[column]
    for field, column in (("edition", "edition"), ("publication_date", "publication_date"),
                          ("official_url", "official_source_url")):
        if row.get(column):
            record[field] = row[column]
    if row.get("identifiers"):
        record["identifiers"] = [{"scheme": "as_recorded", "value": row["identifiers"], "scope": "unresolved"}]
    if topics:
        record["topics"] = topics
    if row.get("related_id") and row.get("relation"):
        note = row["relation"]
        folded = note.casefold()
        relation = ("translation_of" if "translation of" in folded else
                    "summary_of" if "brief of" in folded or "summary of" in folded else "related_to")
        record["relations"] = [{"type": relation, "target": row["related_id"], "state": "candidate", "note": note}]
    return record


def therapeutic_guidelines(root, metadata, category):
    if metadata:
        raise ValueError("Use the Therapeutic Guidelines adapter with its local metadata/ directory.")
    entries = list(local_csv(root, "documents.csv", {"relative_path", "document_id", "title", "issuing_organization", "language"}))
    active = [(line, row) for line, row in entries if row.get("relative_path")]
    by_id = {row["document_id"]: row for _, row in active}
    manual = defaultdict(list)
    topics = defaultdict(list)
    for _, row in optional_local_csv(root, "manual_acquisitions.csv", {"document_id", "sha256"}):
        if "document_id" in row:
            manual[row["document_id"], row["sha256"]].append(row)
    labels = {r["topic_id"]: r["topic"] for _, r in optional_local_csv(root, "coverage.csv", {"topic_id", "topic"})
              if "topic_id" in r}
    for _, link in optional_local_csv(root, "topic_links.csv", {"document_id", "topic_id", "scope", "scope_evidence"}):
        if link.get("topic_id") in labels:
            topics[link["document_id"]].append({"vocabulary": "medquery_therapeutic_plan", "code": link["topic_id"],
                "label": labels[link["topic_id"]], "scope_note": f"{link['scope']}: {link['scope_evidence']}"})

    annexes = defaultdict(list)
    primary = []
    for line, row in active:
        parent = by_id.get(row.get("related_id"))
        same_work = parent and row.get("work_id") and row["work_id"] == parent.get("work_id")
        if (same_work and row["language"] == parent["language"] and
                code(row.get("role", "")) in {"correction", "supplement"} and
                code(parent.get("role", "")) not in {"correction", "supplement"}):
            annexes[parent["document_id"]].append(row)
        else:
            primary.append((line, row))

    for line, row in entries:
        if "_parse_error" in row:
            yield f"row:{line}", row
    for line, row in primary:
        key = row["document_id"]
        assets = [tg_asset(row, "primary", manual[key, row.get("sha256")])]
        assets.extend(tg_asset(annex, "annex", manual[annex["document_id"], annex.get("sha256")])
                      for annex in annexes[key])
        yield f"row:{line}", tg_record(row, assets, topics[key], category)


ADAPTERS: dict[str, Adapter] = {
    "csv": generic_csv,
    "therapeutic-guidelines": therapeutic_guidelines,
    "treatment-protocols": treatment_protocols,
}
