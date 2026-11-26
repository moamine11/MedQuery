"""Phase 2 using synthetic publications and PDF structures only."""

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from ingestion.adapters import ADAPTERS, canonical, generic_csv
from ingestion.metadata import SCHEMA, validate
from ingestion.pipeline import intake, no_metadata

CATEGORY = "treatment_protocols"
PDFINFO = shutil.which("pdfinfo")


def pdf_bytes(mark="a", newline=b"\n"):
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 72] /Marker ({mark}) >>".encode(),
    ]
    content = b"%PDF-1.4" + newline
    positions = []
    for number, body in enumerate(objects, 1):
        positions.append(len(content))
        content += newline.join([f"{number} 0 obj".encode(), body, b"endobj", b""])
    xref = len(content)
    return content + newline.join([
        b"xref", b"0 4", b"0000000000 65535 f ",
        *(f"{offset:010} 00000 n ".encode() for offset in positions),
        b"trailer", b"<< /Size 4 /Root 1 0 R >>", b"startxref", str(xref).encode(), b"%%EOF", b"",
    ])


def record(key, path=None, **fields):
    return {
        "schema_version": 1, "record_id": key, "title": "Synthetic clinical reference",
        "issuer": {"name": "Synthetic Issuer", "kind": "public_agency", "country": "DZ"},
        "language": "fr", "categories": [CATEGORY], "document_type": "clinical_protocol",
        "authority": "official_national", "assets": [{"path": path or f"{key}.pdf", "role": "primary"}],
        **fields,
    }


def work_identifier():
    return {"scheme": "work_code", "value": "SYN-001", "namespace": "synthetic",
            "scope": "document", "verified": True, "evidence": "Synthetic imprint"}


@unittest.skipUnless(PDFINFO, "Poppler pdfinfo must be installed")
class IntakeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "root"
        self.root.mkdir()
        self.manifest = self.root / "manifest.jsonl"

    def run_intake(self, records, **kwargs):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        return intake(self.root, CATEGORY, canonical, self.manifest, PDFINFO, **kwargs)

    def make_pdf(self, name, mark="a", newline=b"\n"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf_bytes(mark, newline))
        return path

    def test_discovery_exclusions_and_cr_pdf(self):
        active = self.make_pdf("pdfs/active.PDF", newline=b"\r")
        for name in ("archive/old.pdf", "reports/source.pdf", "pdfs/.hidden.pdf",
                     "pdfs/file.tmp.pdf", "Treatment_Protocols_DataCollection_Report.pdf"):
            self.make_pdf(name)
        initial = active.read_bytes()
        result = intake(self.root, CATEGORY, no_metadata, None, PDFINFO)
        self.assertEqual(result["summary"]["page_counts_observed"], 1)
        self.assertEqual([item["path"] for item in result["files"]], ["pdfs/active.PDF"])
        self.assertEqual(result["issues"], [])
        self.assertEqual(active.read_bytes(), initial)

    def test_paths_missing_corrupt_encrypted_and_independent_failures(self):
        outside = self.directory / "outside.pdf"
        outside.write_bytes(pdf_bytes())
        (self.root / "link.pdf").symlink_to(outside)
        (self.root / "dirlink").symlink_to(self.directory, target_is_directory=True)
        (self.root / "bad.pdf").write_bytes(b"%PDF-1.4\ninvalid\n%%EOF\n")
        shutil.copyfile(Path(__file__).with_name("fixtures") / "encrypted_test.pdf", self.root / "locked.pdf")
        self.make_pdf("valid.pdf")
        paths = ["../outside.pdf", str(outside), "link.pdf", "dirlink/outside.pdf",
                 "missing.pdf", "bad.pdf", "locked.pdf", "file.docx", "valid.pdf"]
        result = self.run_intake([record(str(i), path) for i, path in enumerate(paths)])
        self.assertEqual(result["summary"]["valid_records"], 1)
        self.assertEqual(result["summary"]["invalid_records"], 8)
        self.assertTrue({"missing_file", "corrupt_pdf", "encrypted_pdf", "invalid_path", "unsupported_file",
                         "unsafe_or_unreadable_path"} <= set(result["summary"]["issues"]))
        self.assertIn("sha256", next(file for file in result["files"] if file["path"] == "locked.pdf")["observed"])

    def test_invalid_manifest_and_metadata_mismatch(self):
        self.make_pdf("one.pdf")
        self.make_pdf("two.pdf")
        bad = record("one")
        bad["assets"][0]["observed"] = {"pages": 2, "sha256": "0" * 64}
        self.manifest.write_text(json.dumps(bad) + '\n{"title": "A", "title":"B"}\n'
                                 + json.dumps(record("two")) + "\n")
        result = intake(self.root, CATEGORY, canonical, self.manifest, PDFINFO)
        self.assertEqual(result["summary"]["valid_records"], 1)
        self.assertEqual(result["summary"]["issues"]["observation_mismatch"], 2)
        self.assertEqual(result["summary"]["issues"]["metadata_parse_error"], 1)

    def test_cross_category_duplicate_renditions_and_editions(self):
        self.make_pdf("same.pdf")
        self.make_pdf("second.pdf", "b")
        self.make_pdf("rendition.pdf", "c")
        base = {"identifiers": [work_identifier()], "revision_key": "2025", "revision_evidence": "Imprint"}
        rows = [
            record("same", edition="2025", **base),
            record("mirror", "same.pdf", categories=["therapeutic_guidelines"], edition="2025", **base),
            record("second", edition="2026", identifiers=base["identifiers"], revision_key="2026",
                   revision_evidence="Imprint", relations=[
                       {"type": "replaces", "target": "same", "state": "verified", "note": "Synthetic sequel"}]),
            record("rendition", edition="2025", **base),
            record("conflict", "same.pdf", language="en", edition="2024"),
        ]
        result = self.run_intake(rows, matching=True)
        self.assertEqual(result["summary"]["valid_records"], 5)
        self.assertEqual(len({entry["identity_key"] for entry in result["inventory"][:4]}), 1)
        queue = result["summary"]["review_queue"]
        self.assertTrue({"exact_bytes", "cross_category_candidate", "different_edition_candidate",
                         "alternate_rendition_candidate", "conflicting_identity_claims"} <= set(queue))
        self.assertEqual(result, self.run_intake(rows, matching=True, workers=1))

    def test_translation_correction_and_cycle(self):
        for key in ("source", "translation", "erratum", "bilingual"):
            self.make_pdf(f"{key}.pdf", key)
        rows = [
            record("source", identifiers=[work_identifier()], assets=[
                {"path": "source.pdf", "role": "primary"},
                {"path": "erratum.pdf", "role": "annex", "attachment_type": "correction"}]),
            record("translation", language="ar", identifiers=[work_identifier()], relations=[
                {"type": "translation_of", "target": "source", "state": "verified", "note": "Imprint"}]),
            record("bilingual", language="mul", content_languages=["ar", "fr"]),
        ]
        result = self.run_intake(rows, matching=True)
        self.assertEqual(result["summary"]["valid_records"], 3)
        self.assertIn("attached_asset", result["summary"]["review_queue"])
        self.assertNotEqual(result["inventory"][0]["identity_key"], result["inventory"][1]["identity_key"])
        rows[0]["relations"] = [{"type": "translation_of", "target": "translation", "state": "verified", "note": "Imprint"}]
        self.assertEqual(self.run_intake(rows)["summary"]["issues"]["relationship_cycle"], 2)

    def test_classifications_and_academic_approval_are_separate(self):
        for key in ("official", "school", "context", "law", "unknown", "foreign"):
            self.make_pdf(f"{key}.pdf", key)
        rows = [
            record("official", academic_review={"status": "approved_for_academic_work", "evidence": "Synthetic professor"}),
            record("school", document_type="educational_material"), record("context", authority="contextual"),
            record("law", document_type="administrative_text"),
            record("unknown", issuer={"name": "Unconfirmed", "kind": "unknown"}),
            record("foreign", authority="international_reference"),
        ]
        result = self.run_intake(rows)
        self.assertEqual([item["classification"] for item in result["inventory"]],
                         ["official_algerian_clinical_candidate", "educational", "contextual",
                          "regulatory_administrative", "unresolved", "international_clinical_candidate"])
        for item in result["inventory"]:
            self.assertFalse(item["treatment_guidance_eligible"])
            self.assertEqual(item["clinical_review"], "not_assessed")

    def test_csv_adapter_preserves_original_values(self):
        self.make_pdf("row.pdf")
        metadata = self.root / "metadata.csv"
        with metadata.open("w", newline="") as stream:
            csv.writer(stream).writerows([["record_id", "path", "title", "issuer", "language", "issuer_country", "doi"],
                                         ["r", "row.pdf", "Sample", "Issuer", " French ", "dz", "https://doi.org/10.1234/ABC"]])
        result = intake(self.root, CATEGORY, generic_csv, metadata, PDFINFO)
        item = result["inventory"][0]["metadata"]
        self.assertEqual(item["language"], "fr")
        self.assertEqual(item["issuer"]["country"], "DZ")
        self.assertNotIn("applicability", item)
        self.assertEqual(item["identifiers"][0]["value"], "10.1234/abc")
        self.assertEqual(item["adapter_context"]["original_csv_row"]["language"], " French ")

    def test_cli_outputs_dry_run_and_cross_collection_references(self):
        self.make_pdf("r.pdf")
        self.manifest.write_text(json.dumps(record("r")) + "\n")
        output = self.directory / "out"
        command = [sys.executable, "-m", "ingestion", "validate", "--root", str(self.root),
                   "--collection", CATEGORY, "--manifest", "manifest.jsonl", "--output", str(output)]
        dry = subprocess.run([*command, "--dry-run"], capture_output=True, check=True)
        self.assertFalse(output.exists())
        wet = subprocess.run(command, capture_output=True, check=True)
        self.assertEqual(dry.stdout, wet.stdout)
        first = {file.name: file.read_bytes() for file in output.iterdir()}
        subprocess.run(command, capture_output=True, check=True)
        self.assertEqual(first, {file.name: file.read_bytes() for file in output.iterdir()})
        command[3] = "reconcile"
        command[7] = "therapeutic_guidelines"
        command[-1] = str(self.directory / "compared")
        self.manifest.write_text(json.dumps(record("r", categories=["therapeutic_guidelines"])) + "\n")
        compared = subprocess.run([*command, "--reference-inventory", str(output / "inventory.jsonl")],
                                  capture_output=True, check=True)
        self.assertIn("cross_category_candidate", json.loads(compared.stdout)["review_queue"])

    def test_treatment_protocols_adapter_overrides_and_educational_hold(self):
        self.make_pdf("1_official_algeria/oncology/DZ_onco_colon-cancer-manual_na_fr.pdf", "cervix")
        self.make_pdf("3_educational_Algeria/cardiology/edu_cardio_hta_fr.pdf", "edu")
        result = intake(self.root, CATEGORY, ADAPTERS["treatment-protocols"], None, PDFINFO)
        self.assertEqual(result["summary"]["valid_records"], 2)
        by_id = {entry["metadata"]["record_id"]: entry for entry in result["inventory"]}
        cervix = by_id["MQ-TP-DZ-ONCO-CERVIX-2016"]
        self.assertIn("col de l’utérus", cervix["metadata"]["title"])
        self.assertEqual(cervix["metadata"]["publication_date"], "2016-02")
        self.assertEqual(cervix["classification"], "official_algerian_clinical_candidate")
        edu = by_id["MQ-TP-EDU-CARDIO-HTA"]
        self.assertEqual(edu["classification"], "educational")
        self.assertEqual(edu["metadata"]["intake_review"], "hold")
        self.assertTrue(edu["metadata"]["adapter_context"]["excluded_from_initial_scope"])

    def test_therapeutic_guidelines_issuer_distinctions(self):
        tg_root = self.root / "tg"
        (tg_root / "metadata").mkdir(parents=True)
        for name in ("esc.pdf", "sap.pdf", "creak.pdf"):
            (tg_root / "pdfs").mkdir(exist_ok=True)
            (tg_root / "pdfs" / name).write_bytes(pdf_bytes(name))
        with (tg_root / "metadata/documents.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows([
                ["document_id", "title", "issuing_organization", "language", "relative_path", "role", "publication_date"],
                ["ESC-1", "ESC Guideline", "European Society of Cardiology", "en", "pdfs/esc.pdf", "guideline", "2024"],
                ["SAP-1", "BPCO Algeria", "Société Algérienne de Pneumophtisiologie", "fr", "pdfs/sap.pdf", "guideline", "2025"],
                ["CREAK-1", "PNDS HAE", "Centre de référence des angioedèmes (CREAK)", "fr", "pdfs/creak.pdf", "guideline", "2024"],
            ])
        result = intake(tg_root, "therapeutic_guidelines", ADAPTERS["therapeutic-guidelines"], None, PDFINFO)
        by_id = {entry["metadata"]["record_id"]: entry for entry in result["inventory"]}
        self.assertEqual(by_id["ESC-1"]["metadata"]["issuer"]["kind"], "professional_body")
        self.assertEqual(by_id["ESC-1"]["classification"], "international_clinical_candidate")
        self.assertEqual(by_id["SAP-1"]["metadata"]["issuer"]["kind"], "professional_body")
        self.assertEqual(by_id["SAP-1"]["classification"], "unresolved")
        self.assertEqual(by_id["CREAK-1"]["classification"], "unresolved")

    def test_cross_collection_exact_duplicate_reconciliation(self):
        c1_root = self.root / "c1"
        c3_root = self.root / "c3"
        c1_root.mkdir()
        c3_root.mkdir()
        shared_pdf = pdf_bytes("shared")
        (c1_root / "pain.pdf").write_bytes(shared_pdf)
        (c3_root / "local-pain.pdf").write_bytes(shared_pdf)
        c3_manifest = self.root / "c3.jsonl"
        c3_manifest.write_text(json.dumps(record("LOCAL-PAIN", "local-pain.pdf", categories=["therapeutic_guidelines"])) + "\n")
        c3_res = intake(c3_root, "therapeutic_guidelines", canonical, c3_manifest, PDFINFO)
        c1_manifest = self.root / "c1.jsonl"
        c1_manifest.write_text(json.dumps(record("MQ-CP-PAIN", "pain.pdf", categories=["clinical_protocols_ministry_directives"])) + "\n")
        c1_res = intake(c1_root, "clinical_protocols_ministry_directives", canonical, c1_manifest, PDFINFO,
                        references=c3_res["inventory"], matching=True)
        self.assertIn("exact_bytes", c1_res["summary"]["review_queue"])
        self.assertIn("cross_category_candidate", c1_res["summary"]["review_queue"])


class MetadataTests(unittest.TestCase):
    def test_schema_dates_identifiers_and_missing_metadata(self):
        Draft202012Validator.check_schema(SCHEMA)
        self.assertIsNone(validate(record("x", publication_date="2025-02-29"))[0])
        self.assertIsNone(validate(record("x", title=""))[0])
        _, errors = validate(record("x", effective_from="2025-03-04", effective_until="2024-03-04",
                                    identifiers=[{"scheme": "doi", "scope": "asset", "value": "bad"},
                                                 {"scheme": "isbn", "scope": "asset", "value": "9789240000000"}]))
        self.assertEqual(sum(item["code"] == "invalid_identifier" for item in errors), 2)
        self.assertIn("invalid_effective_window", {item["code"] for item in errors})
        self.assertIsNone(validate(record("x", language="mul"))[0])
        _, errors = validate(record("x", language="xyz", issuer={"name": "Issuer", "kind": "unknown", "country": "XX"}))
        self.assertTrue({"invalid_language", "invalid_country"} <= {item["code"] for item in errors})
        self.assertIsNone(validate(record("x", clinical_review="approved"))[0])
