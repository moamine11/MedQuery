"""Phase 7 PostgreSQL + pgvector storage integration tests (rollback-only)."""

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from psycopg import errors

from database.connection import connect
from ingestion.chunking import read_jsonl, write_jsonl
from ingestion.embeddings import DEFAULT_MODEL, format_passage_input, get_model_spec, input_digest
from ingestion.storage import load_pipeline_outputs, research_retrieve


def _unit_vec(seed_idx, dims=384):
    vec = [0.01] * dims
    vec[seed_idx % dims] = 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


class StorageIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.db = connect()
        self.addCleanup(self.db.close)
        self.enterContext(self.db.transaction(force_rollback=True))
        self.key = f"p7-{uuid4().hex[:12]}"
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def _build_synthetic_bundle(self, *, nul_in_source=False, bad_dims=False, educational=False, name="bundle"):
        root = self.dir / name
        (root / "extract").mkdir(parents=True, exist_ok=True)
        (root / "chunk").mkdir(parents=True, exist_ok=True)
        (root / "embed").mkdir(parents=True, exist_ok=True)

        sha_pri = hashlib.sha256(f"{self.key}:pri".encode()).hexdigest()
        sha_corr = hashlib.sha256(f"{self.key}:corr".encode()).hexdigest()
        rec_id = f"REC-{self.key}"
        corr_key = f"{rec_id}@corr12345678"

        inv_path = root / "inv.jsonl"
        inv_entry = {
            "collection": "therapeutic_guidelines",
            "validation_state": "valid",
            "classification": "educational" if educational else "international_clinical_candidate",
            "metadata": {
                "schema_version": 1,
                "record_id": rec_id,
                "title": "Synthetic Guideline",
                "language": "fr",
                "content_languages": ["fr", "en"],
                "categories": ["therapeutic_guidelines", "treatment_protocols"],
                "document_type": "educational_material" if educational else "clinical_guideline",
                "authority": "educational" if educational else "international_reference",
                "edition": "2025-04",
                "publication_date": "2025-04-15",
                "issuer": {"name": f"Issuer {self.key}", "kind": "public_agency", "country": "FR"},
                "applicability": {
                    "geography": "country_list",
                    "countries": ["FR"],
                    "population": "restricted",
                    "rules": [{"effect": "include", "population_label": "Adults", "age_days": {"min_inclusive": 6570}}],
                },
                "topics": [{"vocabulary": "medquery_test", "code": f"T-{self.key}", "label": "Diabetes"}],
                "identifiers": [{"scheme": "doi", "value": f"10.9999/{self.key}", "namespace": "crossref",
                                 "scope": "document", "verified": True}],
                "reuse": {"use_status": "unknown", "external_embedding_status": "unknown"},
                "assets": [
                    {"path": "pdfs/main.pdf", "role": "primary", "reuse": {"use_status": "unknown"}},
                    {"path": "pdfs/corr.pdf", "role": "annex", "attachment_type": "correction",
                     "reuse": {"use_status": "unknown"}},
                ],
            },
            "observations": {
                "pdfs/main.pdf": {"bytes": 1024, "pages": 1, "sha256": sha_pri},
                "pdfs/corr.pdf": {"bytes": 512, "pages": 1, "sha256": sha_corr},
            },
        }
        inv_path.write_text(json.dumps(inv_entry) + "\n", encoding="utf-8")

        manifest_path = root / "manifest.json"
        manifest = {
            "schema_version": 1,
            "inventories": {"therapeutic_guidelines": str(inv_path)},
            "eligible_assets": [
                {
                    "asset_key": rec_id,
                    "record_id": rec_id,
                    "collection": "therapeutic_guidelines",
                    "asset_path": "pdfs/main.pdf",
                    "asset_role": "primary",
                    "attachment_type": None,
                    "sha256": sha_pri,
                    "bytes": 1024,
                    "pages": 1,
                    "language": "fr",
                    "target_categories": ["therapeutic_guidelines", "treatment_protocols"],
                    "acquisition_url": "https://example.invalid/main.pdf",
                    "document_use_status": "unknown",
                    "asset_use_status": "unknown",
                },
                {
                    "asset_key": corr_key,
                    "record_id": rec_id,
                    "collection": "therapeutic_guidelines",
                    "asset_path": "pdfs/corr.pdf",
                    "asset_role": "annex",
                    "attachment_type": "correction",
                    "sha256": sha_corr,
                    "bytes": 512,
                    "pages": 1,
                    "language": "fr",
                    "target_categories": ["therapeutic_guidelines", "treatment_protocols"],
                    "acquisition_url": "https://example.invalid/corr.pdf",
                    "document_use_status": "unknown",
                    "asset_use_status": "unknown",
                },
            ],
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        src_main = "Posologie\x00 500 mg." if nul_in_source else "Posologie metformine 500 mg par jour."
        src_corr = "Erratum: dose maximale 2000 mg."
        pages = [
            {
                "asset_key": rec_id,
                "record_id": rec_id,
                "asset_sha256": sha_pri,
                "source_path": "pdfs/main.pdf",
                "language": "fr",
                "physical_page": 1,
                "printed_page_label": "1",
                "status": "needs_review",
                "source_text": src_main + "\n\nHELD TABLE ROW",
                "blocks": [],
                "tables": [{"key": f"{rec_id}:1:table:1", "caption": "Held table", "grid": {"candidates": []}}],
                "figures": [],
            },
            {
                "asset_key": corr_key,
                "record_id": rec_id,
                "asset_sha256": sha_corr,
                "source_path": "pdfs/corr.pdf",
                "language": "fr",
                "physical_page": 1,
                "printed_page_label": "i",
                "status": "extracted",
                "source_text": src_corr,
                "blocks": [],
                "tables": [],
                "figures": [],
            },
        ]
        write_jsonl(root / "extract/pages.jsonl", pages)
        sections = [
            {
                "asset_key": rec_id,
                "section_key": f"{rec_id}:section:1",
                "ordinal": 1,
                "heading": "1. Posologie",
                "section_label": "1",
                "parent_section_key": None,
                "start_page": 1,
                "end_page": 1,
                "anchors": [{"physical_page": 1, "char_start": 0, "char_end": len(src_main)}],
            },
            {
                "asset_key": rec_id,
                "section_key": f"{rec_id}:section:2",
                "ordinal": 2,
                "heading": "1.1 Metformine",
                "section_label": "1.1",
                "parent_section_key": f"{rec_id}:section:1",
                "start_page": 1,
                "end_page": 1,
                "anchors": [{"physical_page": 1, "char_start": 0, "char_end": len(src_main)}],
            },
        ]
        write_jsonl(root / "extract/sections.jsonl", sections)
        write_jsonl(root / "extract/review_queue.jsonl", [
            {"asset_key": rec_id, "code": "unresolved_table_relationship", "physical_page": 1},
        ])

        chk1 = {
            "chunk_id": f"chk1-{self.key}",
            "ordinal": 1,
            "asset_key": rec_id,
            "record_id": rec_id,
            "asset_sha256": sha_pri,
            "language_tag": "fr",
            "strategy": "section_aware",
            "section_key": f"{rec_id}:section:2",
            "physical_pages": [1],
            "retrieval_text": src_main,
            "spans": [{
                "span_no": 1, "physical_page": 1, "char_start": 0, "char_end": len(src_main),
                "source_anchor": f"{rec_id}:1:block:1", "bbox": [0.1, 0.1, 0.8, 0.2],
            }],
        }
        chk2 = {
            "chunk_id": f"chk2-{self.key}",
            "ordinal": 2,
            "asset_key": corr_key,
            "record_id": rec_id,
            "asset_sha256": sha_corr,
            "language_tag": "fr",
            "strategy": "section_aware",
            "section_key": None,
            "physical_pages": [1],
            "retrieval_text": src_corr,
            "spans": [{
                "span_no": 1, "physical_page": 1, "char_start": 0, "char_end": len(src_corr),
                "source_anchor": f"{corr_key}:1:block:1", "bbox": [0.1, 0.3, 0.8, 0.4],
            }],
        }
        write_jsonl(root / "chunk/chunks.jsonl", [chk1, chk2])

        spec = get_model_spec(DEFAULT_MODEL)
        (root / "embed/embedding_model.json").write_text(json.dumps({
            k: spec[k] for k in (
                "provider", "model_name", "model_revision", "input_profile",
                "dimensions", "distance_metric", "is_external", "model_config_fingerprint",
            )
        }), encoding="utf-8")
        emb_rows = []
        for idx, chk in enumerate((chk1, chk2), 1):
            vec = _unit_vec(idx, dims=10 if bad_dims else 384)
            emb_rows.append({
                "chunk_id": chk["chunk_id"],
                "ordinal": chk["ordinal"],
                "model_name": spec["model_name"],
                "model_revision": spec["model_revision"],
                "dimensions": 10 if bad_dims else spec["dimensions"],
                "distance_metric": spec["distance_metric"],
                "input_sha256": input_digest(format_passage_input(chk["retrieval_text"], spec)),
                "embedding": vec,
            })
        write_jsonl(root / "embed/chunk_embeddings.jsonl", emb_rows)
        return manifest_path, root / "extract", root / "chunk", root / "embed", rec_id, src_main

    def test_load_idempotency_citations_and_safety_separation(self):
        manifest, ext, chk, emb, rec_id, src_main = self._build_synthetic_bundle()

        first = load_pipeline_outputs(self.db, manifest_path=manifest, extract_dir=ext, chunk_dir=chk, embed_dir=emb)
        self.assertEqual(first["versions_processed"], 1)
        self.assertEqual(first["extractions_created"], 1)
        self.assertEqual(first["pages_inserted"], 2)
        self.assertEqual(first["sections_inserted"], 2)
        self.assertEqual(first["tables_inserted"], 1)
        self.assertEqual(first["chunks_inserted"], 2)
        self.assertEqual(first["embeddings_inserted"], 2)

        # Verify cross-collection categories, portable URIs, parent section hierarchy, and staged gates.
        cats = self.db.execute(
            """
            SELECT dc.category_code FROM medical_rag.document_categories dc
            JOIN medical_rag.documents d ON d.id = dc.document_id
            WHERE d.identity_key LIKE %s ORDER BY dc.category_code
            """,
            (f"%{rec_id}%",),
        ).fetchall()
        self.assertEqual([r[0] for r in cats], ["therapeutic_guidelines", "treatment_protocols"])

        uris = self.db.execute(
            """
            SELECT a.role, a.storage_uri, a.use_status FROM medical_rag.document_assets a
            JOIN medical_rag.document_versions v ON v.id = a.version_id
            JOIN medical_rag.documents d ON d.id = v.document_id
            WHERE d.identity_key LIKE %s ORDER BY a.role DESC
            """,
            (f"%{rec_id}%",),
        ).fetchall()
        self.assertEqual(uris, [
            ("primary", "medquery://therapeutic_guidelines/pdfs/main.pdf", "unknown"),
            ("annex", "medquery://therapeutic_guidelines/pdfs/corr.pdf", "unknown"),
        ])

        # Authoritative clinical retrieval MUST return 0 rows for staged/unapproved content.
        self.assertEqual(self.db.execute("SELECT count(*) FROM medical_rag.retrievable_chunks").fetchone()[0], 0)
        self.assertEqual(self.db.execute("SELECT count(*) FROM medical_rag.retrievable_citations").fetchone()[0], 0)

        # Research retrieval requires explicit opt-in and returns exact cosine ranking + cited_text.
        with self.assertRaises(PermissionError):
            research_retrieve(self.db, _unit_vec(1))
        hits = research_retrieve(self.db, _unit_vec(1), allow_research_only=True, limit=2)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["retrieval_mode"], "research_only_unapproved")
        self.assertEqual(hits[0]["version_status"], "staged")
        self.assertEqual(hits[0]["spans"][0]["cited_text"], src_main)
        self.assertEqual(hits[0]["section_label"], "1.1")

        # Repeated ingestion is 100% idempotent without duplicating documents, chunks, or embeddings.
        attempt_meta = self.db.execute(
            "SELECT source_metadata FROM medical_rag.ingestion_attempts WHERE version_id = (SELECT id FROM medical_rag.document_versions ORDER BY id DESC LIMIT 1)"
        ).fetchone()[0]
        self.assertEqual(attempt_meta["extraction_completeness"], "complete")

        second = load_pipeline_outputs(self.db, manifest_path=manifest, extract_dir=ext, chunk_dir=chk, embed_dir=emb)
        self.assertEqual(second["extractions_created"], 0)
        self.assertEqual(second["extractions_reused"], 1)
        self.assertEqual(second["chunks_inserted"], 0)
        self.assertEqual(second["embeddings_inserted"], 0)

    def test_nul_detection_bad_embedding_and_rollback(self):
        before_docs = self.db.execute("SELECT count(*) FROM medical_rag.documents").fetchone()[0]
        manifest, ext, chk, emb, rec_id, _ = self._build_synthetic_bundle(nul_in_source=True)
        with self.assertRaisesRegex(ValueError, "NUL"), self.db.transaction():
            load_pipeline_outputs(self.db, manifest_path=manifest, extract_dir=ext, chunk_dir=chk, embed_dir=emb)
        self.assertEqual(self.db.execute("SELECT count(*) FROM medical_rag.documents").fetchone()[0], before_docs)
        self.assertEqual(
            self.db.execute("SELECT count(*) FROM medical_rag.documents WHERE identity_key LIKE %s", (f"%{rec_id}%",)).fetchone()[0],
            0,
        )

        manifest2, ext2, chk2, emb2, rec_id2, _ = self._build_synthetic_bundle(bad_dims=True, name="bundle2")
        with self.assertRaisesRegex(ValueError, "dimensions"), self.db.transaction():
            load_pipeline_outputs(self.db, manifest_path=manifest2, extract_dir=ext2, chunk_dir=chk2, embed_dir=emb2)
        self.assertEqual(self.db.execute("SELECT count(*) FROM medical_rag.documents").fetchone()[0], before_docs)
        self.assertEqual(
            self.db.execute("SELECT count(*) FROM medical_rag.documents WHERE identity_key LIKE %s", (f"%{rec_id2}%",)).fetchone()[0],
            0,
        )

    def test_incremental_staged_recovery(self):
        manifest, ext, chk, emb, _, _ = self._build_synthetic_bundle(name="recov")
        first = load_pipeline_outputs(self.db, manifest_path=manifest, extract_dir=ext, chunk_dir=chk, embed_dir=emb)
        self.assertEqual(first["extractions_created"], 1)
        # Simulate a second pipeline version that crashed while staged with partial pages.
        ver_id = self.db.execute("SELECT id FROM medical_rag.document_versions ORDER BY id DESC LIMIT 1").fetchone()[0]
        asset_id = self.db.execute("SELECT id FROM medical_rag.document_assets WHERE version_id = %s LIMIT 1", (ver_id,)).fetchone()[0]
        staged_id = self.db.execute(
            """
            INSERT INTO medical_rag.extractions (version_id, pipeline_name, pipeline_version, input_fingerprint, config_fingerprint)
            VALUES (%s, 'medquery_ingestion', 'phase7_v2', %s, %s) RETURNING id
            """,
            (ver_id, hashlib.sha256(b"in").digest(), hashlib.sha256(b"cfg").digest()),
        ).fetchone()[0]
        self.db.execute("INSERT INTO medical_rag.extraction_assets VALUES (%s, %s, %s)", (staged_id, asset_id, ver_id))
        # Now run load_pipeline_outputs for phase7_v1 again (reuses completed) and verify staged row doesn't break anything.
        again = load_pipeline_outputs(self.db, manifest_path=manifest, extract_dir=ext, chunk_dir=chk, embed_dir=emb)
        self.assertEqual(again["extractions_reused"], 1)

    def test_educational_exclusion_in_research_retrieval(self):
        manifest, ext, chk, emb, rec_id, src_main = self._build_synthetic_bundle(educational=True)
        load_pipeline_outputs(self.db, manifest_path=manifest, extract_dir=ext, chunk_dir=chk, embed_dir=emb)
        excluded_hits = research_retrieve(self.db, _unit_vec(1), allow_research_only=True, limit=10)
        self.assertFalse(any(h["phase2_classification"] == "educational" for h in excluded_hits))
        self.assertFalse(any(h["retrieval_text"] == src_main and rec_id in h["spans"][0]["source_anchor"] for h in excluded_hits))
        included_hits = research_retrieve(self.db, _unit_vec(1), allow_research_only=True, exclude_educational=False, limit=10)
        self.assertEqual(sum(1 for h in included_hits if h["phase2_classification"] == "educational"), 2)
        self.assertTrue(any(rec_id in h["spans"][0]["source_anchor"] for h in included_hits))

    def test_uncited_nul_sanitized_preserving_offsets(self):
        manifest, ext, chk, emb, rec_id, src_main = self._build_synthetic_bundle(name="uncited_nul")
        # Append a trailing quarantined needs_review block containing a NUL byte outside all chunk spans.
        pages = read_jsonl(ext / "pages.jsonl")
        pages[0]["source_text"] = pages[0]["source_text"] + "\n\nHeld ref\x00note."
        write_jsonl(ext / "pages.jsonl", pages)

        loaded = load_pipeline_outputs(self.db, manifest_path=manifest, extract_dir=ext, chunk_dir=chk, embed_dir=emb)
        self.assertEqual(loaded["sanitized_nul_pages"], [f"{rec_id}:1"])
        hits = research_retrieve(self.db, _unit_vec(1), allow_research_only=True, limit=2)
        self.assertEqual(hits[0]["spans"][0]["cited_text"], src_main)

    def test_unchunkable_staged_idempotent(self):
        manifest, ext, chk, emb, _, _ = self._build_synthetic_bundle(name="unchunkable")
        write_jsonl(chk / "chunks.jsonl", [])
        write_jsonl(emb / "chunk_embeddings.jsonl", [])
        first = load_pipeline_outputs(self.db, manifest_path=manifest, extract_dir=ext, chunk_dir=chk, embed_dir=emb)
        self.assertEqual(first["extractions_created"], 1)
        self.assertEqual(first["chunks_inserted"], 0)
        second = load_pipeline_outputs(self.db, manifest_path=manifest, extract_dir=ext, chunk_dir=chk, embed_dir=emb)
        self.assertEqual(second["extractions_reused"], 1)
        self.assertEqual(second["pages_inserted"], 0)
        self.assertEqual(second["attempts_recorded"], 0)

    def test_pilot_artifacts_load_and_cosine_retrieve_with_citations(self):
        manifest_path = Path("data/ingestion/phase5/extraction_selection_manifest.json")
        ext_dir = Path("data/ingestion/phase5/comparison_corpus")
        chk_dir = Path("data/ingestion/phase5/cli_chunk_run")
        emb_dir = Path("data/ingestion/phase6/cli_embed_run")
        if not (manifest_path.is_file() and (emb_dir / "chunk_embeddings.jsonl").is_file()):
            self.skipTest("Phase 5/6 pilot artifacts not present")

        test_pv = f"phase7_test_{self.key}"
        summary = load_pipeline_outputs(
            self.db,
            manifest_path=manifest_path,
            extract_dir=ext_dir,
            chunk_dir=chk_dir,
            embed_dir=emb_dir,
            pipeline_version=test_pv,
        )
        self.assertEqual(summary["versions_processed"], 8)
        self.assertEqual(summary["extractions_created"], 8)
        self.assertEqual(summary["pages_inserted"], 43)
        self.assertEqual(summary["chunks_inserted"], 106)
        self.assertEqual(summary["spans_inserted"], 503)
        self.assertEqual(summary["embeddings_inserted"], 106)
        # Authoritative clinical views must still remain empty!
        self.assertEqual(self.db.execute("SELECT count(*) FROM medical_rag.retrievable_chunks").fetchone()[0], 0)

        # Idempotent second pass on pilot artifacts
        again = load_pipeline_outputs(
            self.db,
            manifest_path=manifest_path,
            extract_dir=ext_dir,
            chunk_dir=chk_dir,
            embed_dir=emb_dir,
            pipeline_version=test_pv,
        )
        self.assertEqual(again["extractions_reused"], 8)
        self.assertEqual(again["chunks_inserted"], 0)

        # Verify exact cosine retrieval in PostgreSQL using a real chunk's embedding vector.
        sample_emb = read_jsonl(emb_dir / "chunk_embeddings.jsonl")[0]
        hits = research_retrieve(self.db, sample_emb["embedding"], allow_research_only=True, limit=3)
        self.assertGreaterEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0]["cosine_similarity"], 1.0, places=5)
        self.assertEqual(hits[0]["retrieval_text"], read_jsonl(chk_dir / "chunks.jsonl")[0]["retrieval_text"])
        self.assertTrue(all(sp["cited_text"] for sp in hits[0]["spans"]))


if __name__ == "__main__":
    unittest.main()
