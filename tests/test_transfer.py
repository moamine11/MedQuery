"""Phase 9A automated tests for portable paths, transfer inventory, archives, and notebook validity."""

import ast
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from ingestion.chunking import chunk_pages, write_jsonl
from ingestion.embeddings import DEFAULT_MODEL, embed_chunks, get_model_spec
from ingestion.extraction.assemble import finalize_asset
from ingestion.extraction.config import ROOT, load_profile
from ingestion.extraction.runner import run
from ingestion.extraction.selection import (
    build_selection_manifest,
    rebase_selection_manifest,
    select_from_manifest,
    verify_manifest_assets,
)
from ingestion.metadata import validate
from ingestion.transfer import build_transfer_archives, verify_phase9_outputs
from test_embeddings import _FakeEncoder
from test_ingestion import record


class Phase9TransferAndPortabilityTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def _make_pdf(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 120), text)
            pdf.save(path)
        return path.stat().st_size, sha256(path.read_bytes()).hexdigest()

    def _build_fixture_corpus(self):
        col1_root = self.dir / "src_col1"
        col3_root = self.dir / "src_col3"
        b1, sha1 = self._make_pdf(col1_root / "doc1.pdf", "Official national clinical guidance for fever management.")
        b3, sha3 = self._make_pdf(col3_root / "pdfs/g3.pdf", "International guideline on hypertension treatment.")
        b_hold, sha_hold = self._make_pdf(col3_root / "pdfs/held.pdf", "Held permission required document.")

        rec1 = record("DOC-C1", "doc1.pdf")
        rec3 = record("DOC-C3", "pdfs/g3.pdf")
        rec_hold = record("DOC-HOLD", "pdfs/held.pdf", intake_review="hold")

        def inv_row(col, rec, sizes_shas, cls="official_algerian_clinical_candidate"):
            meta, _ = validate(rec)
            return {
                "collection": col,
                "validation_state": "valid",
                "classification": cls,
                "metadata": meta,
                "original_metadata": rec,
                "observations": {
                    p: {"bytes": sz, "pages": 1, "sha256": digest}
                    for p, (sz, digest) in sizes_shas.items()
                },
            }

        inv1 = self.dir / "inv1.jsonl"
        inv3 = self.dir / "inv3.jsonl"
        inv1.write_text(json.dumps(inv_row("clinical_protocols_ministry_directives", rec1, {"doc1.pdf": (b1, sha1)})) + "\n", encoding="utf-8")
        inv3.write_text(
            json.dumps(inv_row("therapeutic_guidelines", rec3, {"pdfs/g3.pdf": (b3, sha3)}, "international_clinical_candidate")) + "\n"
            + json.dumps(inv_row("therapeutic_guidelines", rec_hold, {"pdfs/held.pdf": (b_hold, sha_hold)}, "international_clinical_candidate")) + "\n",
            encoding="utf-8",
        )
        plan = {
            "collection_roots": {
                "clinical_protocols_ministry_directives": str(col1_root),
                "therapeutic_guidelines": str(col3_root),
            },
            "summary": {"technical_extraction_eligible": 2, "technical_extraction_eligible_pages": 2},
            "assets": [
                {"collection": "clinical_protocols_ministry_directives", "record_id": "DOC-C1", "asset_path": "doc1.pdf",
                 "asset_role": "primary", "sha256": sha1, "bytes": b1, "pages": 1, "language": "fr",
                 "v1_scope_bucket": "primary_clinical_candidate", "phase2_classification": "official_algerian_clinical_candidate",
                 "technical_extraction_eligible": True},
                {"collection": "therapeutic_guidelines", "record_id": "DOC-C3", "asset_path": "pdfs/g3.pdf",
                 "asset_role": "primary", "sha256": sha3, "bytes": b3, "pages": 1, "language": "en",
                 "v1_scope_bucket": "international_clinical_candidate", "phase2_classification": "international_clinical_candidate",
                 "technical_extraction_eligible": True},
                {"collection": "therapeutic_guidelines", "record_id": "DOC-HOLD", "asset_path": "pdfs/held.pdf",
                 "asset_role": "primary", "sha256": sha_hold, "bytes": b_hold, "pages": 1, "language": "en",
                 "v1_scope_bucket": "held_or_unresolved", "phase2_classification": "international_clinical_candidate",
                 "technical_extraction_eligible": False, "exclusion_or_hold_reason": "intake_review_hold"},
            ],
        }
        plan_path = self.dir / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        manifest = build_selection_manifest(
            plan_path,
            {"clinical_protocols_ministry_directives": inv1, "therapeutic_guidelines": inv3},
            root=self.dir,
        )
        manifest_path = self.dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, col1_root, col3_root

    def test_rebase_verify_and_portable_env_override(self):
        manifest_path, col1_root, col3_root = self._build_fixture_corpus()
        ver = verify_manifest_assets(manifest_path, root=self.dir, check_pdf_pages=True)
        self.assertTrue(ver["ok"])
        self.assertEqual(ver["eligible_assets"], 2)
        self.assertEqual(ver["verified_pages"], 2)

        # Move files to new portable roots and verify rebase + MEDQUERY_COLLECTION_ROOTS override.
        new_c1 = self.dir / "portable/c1"
        new_c3 = self.dir / "portable/c3"
        new_c1.mkdir(parents=True)
        (new_c3 / "pdfs").mkdir(parents=True)
        (new_c1 / "doc1.pdf").write_bytes((col1_root / "doc1.pdf").read_bytes())
        (new_c3 / "pdfs/g3.pdf").write_bytes((col3_root / "pdfs/g3.pdf").read_bytes())

        rebased_path = self.dir / "rebased.json"
        rebased = rebase_selection_manifest(
            manifest_path,
            rebased_path,
            {"clinical_protocols_ministry_directives": "portable/c1", "therapeutic_guidelines": "portable/c3"},
            root=self.dir,
            exclude_asset_keys={"DOC-C1"},
        )
        self.assertEqual(rebased["summary"]["executable_eligible_assets"], 1)
        self.assertEqual(rebased["summary"]["excluded_or_held_assets"], 2)
        self.assertTrue(verify_manifest_assets(rebased_path, root=self.dir)["ok"])

        with patch.dict("os.environ", {"MEDQUERY_COLLECTION_ROOTS": json.dumps({"therapeutic_guidelines": str(new_c3)})}):
            assets, _, roots = select_from_manifest(manifest_path, ["DOC-C3"], [], root=self.dir)
            self.assertEqual(len(assets), 1)
            self.assertEqual(roots["DOC-C3"], new_c3.resolve())

    def test_transfer_archive_excludes_held_and_verifies_end_to_end_outputs(self):
        manifest_path, _, _ = self._build_fixture_corpus()
        transfer_dir = self.dir / "transfer"
        report = build_transfer_archives(manifest_path, transfer_dir, root=ROOT)
        self.assertEqual(report["transfer_summary"]["hpc_authorized_assets"], 2)
        self.assertEqual(report["transfer_summary"]["excluded_or_held_assets"], 1)
        self.assertEqual(report["archives"]["input"]["pdf_count"], 2)
        self.assertTrue(report["archives"]["code"]["verified"])
        self.assertTrue(report["archives"]["input"]["verified"])
        import zipfile
        with zipfile.ZipFile(transfer_dir / "medquery_hpc_code.zip") as zf:
            self.assertIn("docs/ingestion/metadata.schema.json", zf.namelist())

        # Run extraction -> chunking -> embedding -> verify_phase9_outputs cleanly.
        assets, excluded, asset_roots = select_from_manifest(manifest_path, [], [], root=self.dir)
        profile = {**load_profile(workers=1), "docling_enabled": False, "paddle_enabled": False}
        tools = {"tesseract": None, "poppler": None}
        ext_dir = self.dir / "run/extract"
        chk_dir = self.dir / "run/chunk"
        emb_dir = self.dir / "run/embed"

        # Also exercise fallback when /usr/bin/time is absent.
        with patch("ingestion.extraction.runner.Path.is_file", return_value=False):
            run(self.dir, assets, excluded, ext_dir, profile, tools, {}, resume=True, progress=lambda _: None, asset_roots=asset_roots)

        from ingestion.chunking import read_jsonl
        pages = read_jsonl(ext_dir / "pages.jsonl")
        sections = read_jsonl(ext_dir / "sections.jsonl")
        docs = json.loads((ext_dir / "documents.json").read_text(encoding="utf-8"))["assets"]
        from ingestion.chunking import TokenCounter
        chunks, csum = chunk_pages(pages, sections=sections, documents=docs, counter=TokenCounter(fallback=True))
        chk_dir.mkdir(parents=True)
        write_jsonl(chk_dir / "chunks.jsonl", chunks)
        (chk_dir / "chunk_summary.json").write_text(json.dumps(csum), encoding="utf-8")

        spec = get_model_spec(DEFAULT_MODEL)
        embed_chunks(chunks, emb_dir, encoder=_FakeEncoder(spec), batch_size=2)

        ver = verify_phase9_outputs(manifest_path, ext_dir, chk_dir, emb_dir)
        self.assertTrue(ver["storage_compatible"])
        self.assertTrue(ver["all_pages_complete"])
        self.assertEqual(ver["held_leakage_keys"], [])
        self.assertEqual(ver["citation_issues_count"], 0)
        self.assertEqual(ver["missing_embeddings_count"], 0)

    def test_hpc_notebook_structure_and_python_syntax(self):
        nb_path = ROOT / "notebooks/full_ingestion_hpc.ipynb"
        self.assertTrue(nb_path.is_file())
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 9)
        for cell in code_cells:
            src = "".join(cell["source"])
            ast.parse(src)


if __name__ == "__main__":
    unittest.main()
