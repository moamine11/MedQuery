"""Phase 5 automated tests for extraction selection consistency and citation-aware chunking."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ingestion.chunking import (
    DEFAULT_TOKENIZER_PATH,
    TokenCounter,
    chunk_pages,
    read_jsonl,
    validate_chunk_citations,
)
from ingestion.extraction.assemble import finalize_asset
from ingestion.extraction.normalize import assemble_normalized
from ingestion.extraction.selection import build_selection_manifest, select_from_manifest
from ingestion.metadata import validate
from test_ingestion import record


def _make_page(
    asset_key,
    physical_page,
    blocks_spec,
    *,
    language="fr",
    status="extracted",
    printed_page_label=None,
    record_id=None,
):
    parts = []
    blocks = []
    offset = 0
    for idx, spec in enumerate(blocks_spec, 1):
        text = spec["text"]
        if parts:
            parts.append("\n\n")
            offset += 2
        char_start = offset
        char_end = offset + len(text)
        parts.append(text)
        offset = char_end
        blocks.append({
            "block_key": f"{asset_key}:{physical_page}:block:{idx}",
            "kind": spec.get("kind", "paragraph"),
            "status": spec.get("status", "extracted"),
            "char_start": char_start,
            "char_end": char_end,
            "bbox": spec.get("bbox", [0.1, min(0.85, 0.1 + idx * 0.08), 0.9, min(0.92, 0.15 + idx * 0.08)]),
            "text": text,
            "lines": spec.get("lines", []),
        })
    page = {
        "asset_key": asset_key,
        "record_id": record_id or asset_key,
        "asset_sha256": "a" * 64,
        "source_path": f"{asset_key}.pdf",
        "revision_key": "rev-1",
        "language": language,
        "physical_page": physical_page,
        "printed_page_label": printed_page_label,
        "status": status,
        "source_text": "".join(parts),
        "blocks": blocks,
        "tables": [],
        "figures": [],
    }
    assemble_normalized(page)
    return page


class SelectionConsistencyTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def _inv_row(self, collection, rec, sha_map, classification="official_algerian_clinical_candidate", valid=True):
        meta, _ = validate(rec)
        return {
            "collection": collection,
            "validation_state": "valid" if valid else "invalid",
            "classification": classification,
            "metadata": meta or rec,
            "original_metadata": rec,
            "observations": {
                a["path"]: {"bytes": 100, "pages": 5, "sha256": sha_map[a["path"]]}
                for a in rec["assets"]
            },
        }

    def test_selection_manifest_prevents_hold_circumvention_duplicates_and_unresolved(self):
        sha_pain = "1" * 64
        sha_has = "2" * 64
        sha_unres = "3" * 64
        sha_main = "4" * 64
        sha_corr = "5" * 64
        sha_supp = "6" * 64

        col1_inv = self.dir / "col1.jsonl"
        col2_inv = self.dir / "col2.jsonl"
        col3_inv = self.dir / "col3.jsonl"

        col1_rows = [
            self._inv_row("clinical_protocols_ministry_directives", record("PAIN-C1", "pain.pdf"), {"pain.pdf": sha_pain}),
        ]
        col2_rows = [
            self._inv_row("treatment_protocols", record("HAS-C2", "has.pdf"), {"has.pdf": sha_has},
                          classification="international_clinical_candidate"),
        ]
        col3_rows = [
            # Held copy of sha_pain in Col 3 must block PAIN-C1 too!
            self._inv_row("therapeutic_guidelines", record("PAIN-C3", "pain.pdf", intake_review="hold"), {"pain.pdf": sha_pain}),
            self._inv_row("therapeutic_guidelines", record("HAS-C3", "has.pdf"), {"has.pdf": sha_has},
                          classification="international_clinical_candidate"),
            self._inv_row("therapeutic_guidelines", record("UNRES-C3", "unres.pdf"), {"unres.pdf": sha_unres},
                          classification="unresolved"),
            self._inv_row("therapeutic_guidelines", record("MAIN-C3", "main.pdf", assets=[
                {"path": "main.pdf", "role": "primary"},
                {"path": "corr.pdf", "role": "annex", "attachment_type": "correction"},
                {"path": "supp.pdf", "role": "annex", "attachment_type": "supplement"},
            ]), {"main.pdf": sha_main, "corr.pdf": sha_corr, "supp.pdf": sha_supp},
                classification="international_clinical_candidate"),
        ]
        col1_inv.write_text("".join(json.dumps(r) + "\n" for r in col1_rows), encoding="utf-8")
        col2_inv.write_text("".join(json.dumps(r) + "\n" for r in col2_rows), encoding="utf-8")
        col3_inv.write_text("".join(json.dumps(r) + "\n" for r in col3_rows), encoding="utf-8")

        plan = {
            "collection_roots": {
                "clinical_protocols_ministry_directives": "col1_root",
                "treatment_protocols": "col2_root",
                "therapeutic_guidelines": "col3_root",
            },
            "summary": {"technical_extraction_eligible": 6, "technical_extraction_eligible_pages": 30},
            "assets": [
                {"collection": "clinical_protocols_ministry_directives", "record_id": "PAIN-C1", "asset_path": "pain.pdf",
                 "asset_role": "primary", "sha256": sha_pain, "bytes": 100, "pages": 5, "language": "fr",
                 "v1_scope_bucket": "duplicate_or_alternate_rendition", "phase2_classification": "official_algerian_clinical_candidate",
                 "technical_extraction_eligible": False, "exclusion_or_hold_reason": "exact_sha256_duplicate_extracted_via_therapeutic_guidelines"},
                {"collection": "therapeutic_guidelines", "record_id": "PAIN-C3", "asset_path": "pain.pdf",
                 "asset_role": "primary", "sha256": sha_pain, "bytes": 100, "pages": 5, "language": "fr",
                 "v1_scope_bucket": "primary_clinical_candidate", "phase2_classification": "official_algerian_clinical_candidate",
                 "technical_extraction_eligible": True, "exclusion_or_hold_reason": ""},
                {"collection": "treatment_protocols", "record_id": "HAS-C2", "asset_path": "has.pdf",
                 "asset_role": "primary", "sha256": sha_has, "bytes": 100, "pages": 5, "language": "fr",
                 "v1_scope_bucket": "duplicate_or_alternate_rendition", "phase2_classification": "international_clinical_candidate",
                 "technical_extraction_eligible": False, "exclusion_or_hold_reason": "exact_sha256_duplicate_extracted_via_therapeutic_guidelines"},
                {"collection": "therapeutic_guidelines", "record_id": "HAS-C3", "asset_path": "has.pdf",
                 "asset_role": "primary", "sha256": sha_has, "bytes": 100, "pages": 5, "language": "fr",
                 "v1_scope_bucket": "international_clinical_candidate", "phase2_classification": "international_clinical_candidate",
                 "technical_extraction_eligible": True, "exclusion_or_hold_reason": ""},
                {"collection": "therapeutic_guidelines", "record_id": "UNRES-C3", "asset_path": "unres.pdf",
                 "asset_role": "primary", "sha256": sha_unres, "bytes": 100, "pages": 5, "language": "fr",
                 "v1_scope_bucket": "held_or_unresolved", "phase2_classification": "unresolved",
                 "technical_extraction_eligible": False, "exclusion_or_hold_reason": "unresolved_issuer_or_society_classification"},
                {"collection": "therapeutic_guidelines", "record_id": "MAIN-C3", "asset_path": "main.pdf",
                 "asset_role": "primary", "sha256": sha_main, "bytes": 100, "pages": 5, "language": "fr",
                 "v1_scope_bucket": "international_clinical_candidate", "phase2_classification": "international_clinical_candidate",
                 "technical_extraction_eligible": True, "exclusion_or_hold_reason": ""},
                {"collection": "therapeutic_guidelines", "record_id": "MAIN-C3", "asset_path": "corr.pdf",
                 "asset_role": "annex", "sha256": sha_corr, "bytes": 100, "pages": 5, "language": "fr",
                 "v1_scope_bucket": "international_clinical_candidate", "phase2_classification": "international_clinical_candidate",
                 "technical_extraction_eligible": True, "exclusion_or_hold_reason": ""},
                {"collection": "therapeutic_guidelines", "record_id": "MAIN-C3", "asset_path": "supp.pdf",
                 "asset_role": "annex", "sha256": sha_supp, "bytes": 100, "pages": 5, "language": "fr",
                 "v1_scope_bucket": "international_clinical_candidate", "phase2_classification": "international_clinical_candidate",
                 "technical_extraction_eligible": True, "exclusion_or_hold_reason": ""},
            ],
        }
        plan_path = self.dir / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")

        manifest = build_selection_manifest(plan_path, {
            "clinical_protocols_ministry_directives": col1_inv,
            "treatment_protocols": col2_inv,
            "therapeutic_guidelines": col3_inv,
        }, root=self.dir)

        eligible_ids = {item["asset_key"] for item in manifest["eligible_assets"]}
        self.assertEqual(len(eligible_ids), 4)
        self.assertIn("HAS-C3", eligible_ids)
        self.assertNotIn("PAIN-C1", eligible_ids)
        self.assertNotIn("PAIN-C3", eligible_ids)
        self.assertNotIn("UNRES-C3", eligible_ids)
        # Category preservation on deduplicated HAS-C3
        has_entry = next(i for i in manifest["eligible_assets"] if i["asset_key"] == "HAS-C3")
        self.assertEqual(has_entry["target_categories"], ["therapeutic_guidelines", "treatment_protocols"])
        # Correction and supplement preserved
        self.assertEqual(manifest["summary"]["executable_eligible_assets"], 4)
        self.assertEqual(manifest["summary"]["executable_eligible_pages"], 20)

        manifest_path = self.dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        assets, _, roots = select_from_manifest(manifest_path, ["MAIN-C3"], ["MAIN-C3:1-3", "1-2"], root=self.dir)
        self.assertEqual(len(assets), 3)
        pages_by_key = {a.key: a.selected_pages for a in assets}
        self.assertEqual(pages_by_key["MAIN-C3"], (1, 2, 3))
        self.assertTrue(all(v == (1, 2) for k, v in pages_by_key.items() if k != "MAIN-C3"))
        self.assertEqual(roots["MAIN-C3"], (self.dir / "col3_root").resolve())
        with self.assertRaises(ValueError):
            select_from_manifest(manifest_path, ["PAIN-C1"], [], root=self.dir)


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.counter = TokenCounter(
            DEFAULT_TOKENIZER_PATH if DEFAULT_TOKENIZER_PATH.is_file() else None,
            fallback=not DEFAULT_TOKENIZER_PATH.is_file(),
        )

    def test_token_limits_and_medical_unit_preservation(self):
        long_sentences = " ".join(
            f"Administer metformin 500 mg twice daily and monitor eGFR ≥ 30 mL/min in patient {i}; ne pas dépasser 2000 mg/j."
            for i in range(45)
        )
        page = _make_page("DOC-LIMITS", 1, [
            {"kind": "heading", "text": "1. Dosing Limits"},
            {"kind": "paragraph", "text": long_sentences},
        ], language="FR")
        sections = finalize_asset([page])
        chunks, summary = chunk_pages([page], sections=sections, strategy="section_aware", counter=self.counter, max_tokens=200)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c["token_count"] <= 200 for c in chunks))
        self.assertTrue(all(c["language_tag"] == "fr" for c in chunks))
        self.assertEqual(validate_chunk_citations(chunks, [page]), [])
        # Ensure quantity-unit pairs remain intact inside chunk texts
        for chunk in chunks:
            self.assertNotIn("500\n", chunk["retrieval_text"])
            self.assertRegex(chunk["retrieval_text"], r"500 mg|2000 mg/j|30 mL/min")
        self.assertEqual(summary["total_chunks"], len(chunks))

    def test_source_offsets_nel_safety_and_citation_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            page = _make_page("DOC-NEL", 1, [
                {"kind": "paragraph", "text": "Posologie\u0085recommandée: 5,25\u00a0mg ≥ 2×/j, ne pas interrompre."},
                {"kind": "paragraph", "text": "Second paragraph with Café\u0301 and kg/m²."},
            ], printed_page_label="12")
            sections = finalize_asset([page])
            pages_file = tmp_dir / "pages.jsonl"
            pages_file.write_text(json.dumps(page, ensure_ascii=False) + "\n", encoding="utf-8")
            # Confirm standard .splitlines() would break on \x85 while read_jsonl succeeds
            self.assertGreater(len(pages_file.read_text(encoding="utf-8").splitlines()), 1)
            loaded = read_jsonl(pages_file)
            self.assertEqual(len(loaded), 1)
            chunks, _ = chunk_pages(loaded, sections=sections, strategy="section_aware", counter=self.counter)
            self.assertEqual(validate_chunk_citations(chunks, loaded), [])
            span = chunks[0]["spans"][0]
            cited = loaded[0]["source_text"][span["char_start"]:span["char_end"]]
            self.assertIn("5,25", cited)
            self.assertEqual(span["printed_page_label"], "12")

    def test_review_boundary_exclusion_and_unchunkable_reporting(self):
        mixed = _make_page("DOC-MIXED", 1, [
            {"kind": "heading", "text": "1. Clinical Guidance"},
            {"kind": "paragraph", "text": "Accepted clinical recommendation sentence."},
            {"kind": "table_text", "status": "needs_review", "text": "HELD TABLE ROW 999 mg"},
            {"kind": "figure_label", "status": "needs_review", "text": "HELD DIAGRAM ARROW"},
            {"kind": "paragraph", "status": "needs_review", "text": "UNCERTAIN OCR CONTENT"},
        ])
        held_only = _make_page("DOC-HELD", 1, [
            {"kind": "table_text", "status": "needs_review", "text": "Only held table"},
        ], status="needs_review")
        blank = _make_page("DOC-BLANK", 1, [], status="blank")
        pages = [mixed, held_only, blank]
        sections = finalize_asset([mixed])
        chunks, summary = chunk_pages(pages, sections=sections, strategy="section_aware", counter=self.counter)
        joined = " ".join(c["retrieval_text"] for c in chunks)
        self.assertIn("Accepted clinical recommendation sentence.", joined)
        self.assertNotIn("HELD TABLE", joined)
        self.assertNotIn("HELD DIAGRAM", joined)
        self.assertNotIn("UNCERTAIN OCR", joined)
        self.assertEqual(summary["unchunkable_documents_count"], 2)
        self.assertEqual({d["asset_key"] for d in summary["unchunkable_documents"]}, {"DOC-HELD", "DOC-BLANK"})

    def test_multilingual_preservation_without_nfkc_folding(self):
        ar_page = _make_page("DOC-AR", 1, [
            {"kind": "heading", "text": "١. التوصيات السريرية"},
            {"kind": "paragraph", "text": "يُوصى بجرعة ٥ ملغ ولا يُعطى أكثر من ١٫٥ كغ/م² يومياً."},
        ], language="ar")
        mul_page = _make_page("DOC-MUL", 1, [
            {"kind": "paragraph", "text": "Arrêté n° 11 — المادة ٧: لا يُطبق على المرضى دون 18 سنة."},
        ], language="MUL")
        sections = finalize_asset([ar_page]) + finalize_asset([mul_page])
        chunks, _ = chunk_pages([ar_page, mul_page], sections=sections, strategy="section_aware", counter=self.counter)
        self.assertEqual(len(chunks), 2)
        self.assertEqual({c["language_tag"] for c in chunks}, {"ar", "mul"})
        ar_chunk = next(c for c in chunks if c["language_tag"] == "ar")
        self.assertIn("١٫٥", ar_chunk["retrieval_text"])
        self.assertIn("م²", ar_chunk["retrieval_text"])
        self.assertEqual(validate_chunk_citations(chunks, [ar_page, mul_page]), [])

    def test_cross_page_continuity_and_discontinuity_barriers(self):
        # Pages 1 and 2 continue cleanly; Page 3 has a leading needs_review barrier; Page 5 skips Page 4.
        p1 = _make_page("DOC-CONT", 1, [
            {"kind": "heading", "text": "2. Protocol"},
            {"kind": "paragraph", "text": "First page prose continues cleanly."},
        ])
        p2 = _make_page("DOC-CONT", 2, [
            {"kind": "paragraph", "text": "Second page continuation under same section."},
        ])
        p3 = _make_page("DOC-CONT", 3, [
            {"kind": "table_text", "status": "needs_review", "text": "Held table barrier at top of page 3."},
            {"kind": "paragraph", "text": "Prose after barrier on page 3."},
        ])
        p5 = _make_page("DOC-CONT", 5, [
            {"kind": "paragraph", "text": "Prose after page gap on page 5."},
        ])
        pages = [p1, p2, p3, p5]
        sections = finalize_asset(pages)
        chunks, _ = chunk_pages(pages, sections=sections, strategy="section_aware", counter=self.counter)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["physical_pages"], [1, 2])
        self.assertEqual(chunks[1]["physical_pages"], [3])
        self.assertEqual(chunks[2]["physical_pages"], [5])
        self.assertEqual(validate_chunk_citations(chunks, pages), [])

    def test_deterministic_chunking_and_cli_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            in_dir = tmp_dir / "in"
            out1 = tmp_dir / "out1"
            out2 = tmp_dir / "out2"
            in_dir.mkdir()
            page = _make_page("DOC-DET", 1, [
                {"kind": "heading", "text": "1. Summary"},
                {"kind": "paragraph", "text": "Deterministic medical guidance with 10 mg dose."},
            ])
            sections = finalize_asset([page])
            (in_dir / "pages.jsonl").write_text(json.dumps(page) + "\n", encoding="utf-8")
            (in_dir / "sections.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sections), encoding="utf-8")
            (in_dir / "documents.json").write_text(json.dumps({"assets": [{"asset_key": "DOC-DET", "record_id": "DOC-DET"}]}))

            tok_args = ["--tokenizer", str(DEFAULT_TOKENIZER_PATH)] if DEFAULT_TOKENIZER_PATH.is_file() else []
            if not tok_args:
                self.skipTest("Cached Phase 5 tokenizer.json not available for CLI test")
            for out in (out1, out2):
                subprocess.run(
                    [sys.executable, "-m", "ingestion", "chunk", "--input", str(in_dir), "--output", str(out), *tok_args],
                    check=True,
                    capture_output=True,
                )
            for name in ("chunks.jsonl", "chunk_spans.jsonl", "chunk_summary.json"):
                self.assertEqual((out1 / name).read_bytes(), (out2 / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
