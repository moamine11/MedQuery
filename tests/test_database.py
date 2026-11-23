"""Integration tests for the applied schema using rollback-only transactions."""

import hashlib
import unittest
from uuid import uuid4

from psycopg import errors

from database.connection import connect


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.db = connect()
        self.addCleanup(self.db.close)
        self.enterContext(self.db.transaction(force_rollback=True))
        self.db.execute("SET LOCAL search_path = medical_rag, public")
        self.key = f"db-test-{uuid4().hex}"

    def insert_id(self, query, params):
        return self.db.execute(query + " RETURNING id", params).fetchone()[0]

    def create_document(self):
        source = self.insert_id(
            "INSERT INTO sources (source_key, name, kind) VALUES (%s, %s, 'publisher')",
            (self.key, "Synthetic test publisher"),
        )
        document = self.insert_id(
            """
            INSERT INTO documents (issuer_id, identity_key, canonical_title, language_tag)
            VALUES (%s, %s, 'Synthetic document', 'en')
            """,
            (source, self.key),
        )
        return source, document

    def create_version(self, document, revision, replaces=None):
        return self.insert_id(
            """
            INSERT INTO document_versions
                (document_id, revision_key, title_as_published, replaces_version_id, use_status)
            VALUES (%s, %s, 'Synthetic edition', %s, 'allowed')
            """,
            (document, revision, replaces),
        )

    def create_extraction(self, version):
        asset_hash = hashlib.sha256(f"{self.key}:{version}".encode()).digest()
        asset = self.insert_id(
            """
            INSERT INTO document_assets
                (version_id, sha256, storage_uri, media_type, byte_size, role, use_status)
            VALUES (%s, %s, %s, 'application/pdf', 20, 'primary', 'allowed')
            """,
            (version, asset_hash, f"test://{self.key}/{version}"),
        )
        extraction = self.insert_id(
            """
            INSERT INTO extractions
                (version_id, pipeline_name, pipeline_version, input_fingerprint, config_fingerprint)
            VALUES (%s, 'test', '1', %s, %s)
            """,
            (version, asset_hash, hashlib.sha256(b"test configuration").digest()),
        )
        self.db.execute("INSERT INTO extraction_assets VALUES (%s, %s, %s)", (extraction, asset, version))
        page = self.insert_id(
            """
            INSERT INTO extracted_pages (extraction_id, asset_id, page_index, printed_page_label, source_text)
            VALUES (%s, %s, 3, 'ii', 'Résumé test')
            """,
            (extraction, asset),
        )
        return extraction, asset, page

    def create_chunk(self, extraction, page, ordinal=1):
        chunk = self.insert_id(
            """
            INSERT INTO chunks (extraction_id, ordinal, language_tag, retrieval_text)
            VALUES (%s, %s, 'en', 'Synthetic retrieval text')
            """,
            (extraction, ordinal),
        )
        self.db.execute(
            """
            INSERT INTO chunk_spans (chunk_id, span_no, extraction_id, page_id, char_start, char_end)
            VALUES (%s, 1, %s, %s, 0, 6)
            """,
            (chunk, extraction, page),
        )
        return chunk

    def create_model(self):
        return self.insert_id(
            """
            INSERT INTO embedding_models
                (provider, model_name, model_revision, input_profile, dimensions, distance_metric, is_external)
            VALUES ('synthetic', %s, '1', 'default', 2, 'l2', false)
            """,
            (self.key,),
        )

    def insert_embedding(self, chunk, model, vector):
        self.db.execute(
            """
            INSERT INTO chunk_embeddings (chunk_id, model_id, dimensions, distance_metric, embedding, input_sha256)
            VALUES (%s, %s, 2, 'l2', %s::vector, %s)
            """,
            (chunk, model, vector, hashlib.sha256(b"Synthetic retrieval text").digest()),
        )

    def complete(self, extraction):
        self.db.execute("UPDATE extractions SET status = 'completed' WHERE id = %s", (extraction,))

    def review(self, extraction, decision):
        self.db.execute(
            "INSERT INTO clinical_reviews (extraction_id, reviewer_ref, decision) VALUES (%s, %s, %s)",
            (extraction, self.key, decision),
        )

    def publish(self, extraction, model):
        self.db.execute("SELECT publish_extraction(%s, %s)", (extraction, model))

    def retrievable(self, document):
        return self.db.execute(
            "SELECT id FROM retrievable_chunks WHERE document_id = %s ORDER BY id", (document,)
        ).fetchall()

    def test_connection_and_pgvector(self):
        with connect() as connection:
            self.assertEqual(connection.execute("SELECT 1").fetchone(), (1,))
            version = connection.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
            self.assertIsNotNone(version)
            self.assertEqual(connection.execute("SELECT public.vector_dims('[1,2]'::public.vector)").fetchone(), (2,))
        self.assertTrue(connection.closed)

    def test_foreign_keys_reject_wrong_version_and_extraction(self):
        _, document = self.create_document()
        first = self.create_version(document, "first")
        second = self.create_version(document, "second", first)
        first_extraction, asset, first_page = self.create_extraction(first)
        second_extraction, _, second_page = self.create_extraction(second)
        chunk = self.create_chunk(first_extraction, first_page)

        with self.assertRaises(errors.ForeignKeyViolation), self.db.transaction():
            self.db.execute("INSERT INTO extraction_assets VALUES (%s, %s, %s)", (second_extraction, asset, second))
        with self.assertRaises(errors.ForeignKeyViolation), self.db.transaction():
            self.db.execute(
                """
                INSERT INTO chunk_spans (chunk_id, span_no, extraction_id, page_id, source_anchor)
                VALUES (%s, 2, %s, %s, 'wrong-page')
                """,
                (chunk, first_extraction, second_page),
            )

    def test_document_identity_and_version_duplicates(self):
        source, document = self.create_document()
        with self.assertRaises(errors.UniqueViolation), self.db.transaction():
            self.db.execute(
                """
                INSERT INTO documents (issuer_id, identity_key, canonical_title, language_tag)
                VALUES (%s, %s, 'A changed title', 'en')
                """,
                (source, self.key),
            )
        first = self.create_version(document, "first")
        with self.assertRaises(errors.UniqueViolation), self.db.transaction():
            self.create_version(document, "first")
        _, asset, _ = self.create_extraction(first)
        with self.assertRaises(errors.UniqueViolation), self.db.transaction():
            self.db.execute(
                """
                INSERT INTO document_assets (version_id, sha256, storage_uri, media_type, byte_size, role)
                SELECT version_id, sha256, %s, media_type, byte_size, 'alternate'
                FROM document_assets WHERE id = %s
                """,
                (f"test://{self.key}/duplicate", asset),
            )
        second = self.create_version(document, "second", first)
        self.assertEqual(
            self.db.execute("SELECT replaces_version_id FROM document_versions WHERE id = %s", (second,)).fetchone(),
            (first,),
        )
        self.db.execute("UPDATE document_versions SET status = 'current' WHERE id = %s", (first,))
        with self.assertRaises(errors.UniqueViolation), self.db.transaction():
            self.db.execute("UPDATE document_versions SET status = 'current' WHERE id = %s", (second,))

    def test_citation_traces_to_asset_and_acquisition(self):
        source, document = self.create_document()
        version = self.create_version(document, "first")
        extraction, asset, page = self.create_extraction(version)
        chunk = self.create_chunk(extraction, page)
        model = self.create_model()
        original_url = f"https://example.invalid/{self.key}"
        self.db.execute(
            """
            INSERT INTO ingestion_attempts
                (work_key, attempt_no, stage, status, source_id, requested_uri, version_id, asset_id, finished_at)
            VALUES (%s, 1, 'fetch', 'succeeded', %s, %s, %s, %s, now())
            """,
            (self.key, source, original_url, version, asset),
        )
        self.complete(extraction)
        self.insert_embedding(chunk, model, "[1,0]")
        self.review(extraction, "approved")
        self.publish(extraction, model)

        citation = self.db.execute(
            """
            SELECT issuer_name, title_as_published, asset_id, source_url, page_index, printed_page_label, cited_text
            FROM retrievable_citations WHERE chunk_id = %s
            """,
            (chunk,),
        ).fetchone()
        self.assertEqual(citation, ("Synthetic test publisher", "Synthetic edition", asset, original_url, 3, "ii", "Résumé"))

    def test_embeddings_validate_dimensions_and_rank_by_distance(self):
        _, document = self.create_document()
        extraction, _, page = self.create_extraction(self.create_version(document, "first"))
        farther = self.create_chunk(extraction, page)
        closer = self.create_chunk(extraction, page, ordinal=2)
        model = self.create_model()
        self.complete(extraction)
        with self.assertRaises(errors.CheckViolation), self.db.transaction():
            self.insert_embedding(closer, model, "[1,0,0]")
        self.insert_embedding(farther, model, "[0,1]")
        self.insert_embedding(closer, model, "[1,0]")
        self.review(extraction, "approved")
        self.publish(extraction, model)

        rows = self.db.execute(
            """
            SELECT chunk_id, embedding <-> %s::vector AS distance
            FROM retrievable_embeddings
            WHERE extraction_id = %s AND model_id = %s
            ORDER BY distance, chunk_id
            """,
            ("[1,0]", extraction, model),
        ).fetchall()
        self.assertEqual([row[0] for row in rows], [closer, farther])
        self.assertEqual(rows[0][1], 0)
        self.assertAlmostEqual(rows[1][1], 2**0.5)

    def test_retrieval_requires_current_approval_and_asset_permission(self):
        _, document = self.create_document()
        version = self.create_version(document, "first")
        extraction, asset, page = self.create_extraction(version)
        chunk = self.create_chunk(extraction, page)
        model = self.create_model()
        self.complete(extraction)
        self.insert_embedding(chunk, model, "[1,0]")

        self.assertEqual(self.retrievable(document), [])
        with self.assertRaisesRegex(errors.RaiseException, "current clinical approval"), self.db.transaction():
            self.publish(extraction, model)

        # Exercise the view's approval gate independently of the publication function.
        self.db.execute("UPDATE document_versions SET status = 'current' WHERE id = %s", (version,))
        self.db.execute("UPDATE extractions SET is_selected = true WHERE id = %s", (extraction,))
        self.assertEqual(self.retrievable(document), [])
        self.assertEqual(
            self.db.execute("SELECT chunk_id FROM retrievable_embeddings WHERE chunk_id = %s", (chunk,)).fetchall(),
            [],
        )

        self.review(extraction, "approved")
        self.publish(extraction, model)
        self.assertEqual(self.retrievable(document), [(chunk,)])
        self.db.execute("UPDATE document_assets SET use_status = 'restricted' WHERE id = %s", (asset,))
        self.assertEqual(self.retrievable(document), [])
        self.db.execute("UPDATE document_assets SET use_status = 'allowed' WHERE id = %s", (asset,))
        self.review(extraction, "revoked")
        self.assertEqual(self.retrievable(document), [])
