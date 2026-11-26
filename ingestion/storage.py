"""Idempotent, trigger-safe PostgreSQL + pgvector storage loader and research retrieval."""

from collections import defaultdict
from datetime import date
import json
from pathlib import Path
import re
from time import perf_counter

from database.connection import connect
from ingestion.chunking import read_jsonl
from ingestion.embeddings import DEFAULT_MODEL, get_model_spec, validate_vector
from ingestion.extraction.config import ROOT, canonical, fingerprint
from ingestion.extraction.layout import valid_box
from ingestion.metadata import text_key

ISO_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _check_no_nul(value, field_name):
    """Reject NUL bytes before PostgreSQL insertion without altering citation offsets."""
    if isinstance(value, str) and "\x00" in value:
        raise ValueError(f"PostgreSQL-incompatible NUL (0x00) byte detected in {field_name}; refusing silent offset shift")
    return value


def _sha256_bytes(hex_str, field_name="sha256"):
    if not isinstance(hex_str, str) or not SHA256_HEX_RE.fullmatch(hex_str.lower()):
        raise ValueError(f"Invalid SHA-256 hex digest for {field_name}: {hex_str!r}")
    return bytes.fromhex(hex_str)


def _full_date_or_none(value):
    if not isinstance(value, str) or not ISO_DAY_RE.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _portable_uri(collection, asset_path):
    clean = str(asset_path).lstrip("/")
    if Path(clean).is_absolute() or ".." in Path(clean).parts:
        raise ValueError(f"Asset path must be portable and relative: {asset_path!r}")
    return f"medquery://{collection}/{clean}"


def _derive_source_key(issuer):
    if issuer.get("key"):
        return issuer["key"]
    return "src:" + fingerprint([text_key(issuer["name"]), issuer.get("kind", "unknown"), issuer.get("country") or ""])[:16]


def _derive_identity_key(entry, collection, record_id):
    return entry.get("identity_key") or f"rec:{collection}:{record_id}"


def _derive_revision_key(entry, record_id, primary_sha256):
    return entry.get("revision_key") or f"rev:{fingerprint([record_id, primary_sha256])[:16]}"


def _vector_literal(vec):
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def load_inventory_records(inventory_paths):
    """Index validated inventory records by (collection, record_id) and (asset_key)."""
    by_rec = {}
    for path in inventory_paths:
        for entry in read_jsonl(path):
            meta = entry.get("metadata") or entry.get("original_metadata") or {}
            col = entry.get("collection")
            rec_id = meta.get("record_id")
            if col and rec_id:
                by_rec[(col, rec_id)] = entry
    return by_rec


def _resolve_inputs(manifest_path, inventory_paths, extract_dir, chunk_dir, embed_dir):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    eligible_by_key = {item["asset_key"]: item for item in manifest.get("eligible_assets", [])}
    inv_paths = list(inventory_paths or [])
    if not inv_paths and manifest.get("inventories"):
        inv_paths = [
            (ROOT / p if not Path(p).is_absolute() else Path(p))
            for p in manifest["inventories"].values()
        ]
    inv_by_rec = load_inventory_records(inv_paths)

    extract_dir = Path(extract_dir)
    chunk_dir = Path(chunk_dir)
    embed_dir = Path(embed_dir)

    pages = read_jsonl(extract_dir / "pages.jsonl")
    sections = read_jsonl(extract_dir / "sections.jsonl") if (extract_dir / "sections.jsonl").is_file() else []
    review_items = read_jsonl(extract_dir / "review_queue.jsonl") if (extract_dir / "review_queue.jsonl").is_file() else []
    chunks = read_jsonl(chunk_dir / "chunks.jsonl")
    emb_model = json.loads((embed_dir / "embedding_model.json").read_text(encoding="utf-8"))
    embeddings = read_jsonl(embed_dir / "chunk_embeddings.jsonl")
    return manifest, eligible_by_key, inv_by_rec, pages, sections, review_items, chunks, emb_model, embeddings


def _upsert_source(conn, *, source_key, name, kind, homepage_url=None, country_code=None):
    _check_no_nul(name, "sources.name")
    row = conn.execute(
        """
        INSERT INTO medical_rag.sources (source_key, name, kind, homepage_url, country_code)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (source_key) DO UPDATE
            SET name = EXCLUDED.name,
                kind = EXCLUDED.kind,
                homepage_url = COALESCE(EXCLUDED.homepage_url, medical_rag.sources.homepage_url),
                country_code = COALESCE(EXCLUDED.country_code, medical_rag.sources.country_code)
        RETURNING id
        """,
        (source_key, name, kind, homepage_url, country_code),
    ).fetchone()
    return row[0]


def _upsert_embedding_model(conn, model_meta):
    spec = get_model_spec(model_meta["model_name"], revision=model_meta.get("model_revision"))
    if model_meta.get("dimensions") != spec["dimensions"]:
        raise ValueError(f"Model dimensions mismatch: {model_meta.get('dimensions')} != {spec['dimensions']}")
    if model_meta.get("distance_metric") != spec["distance_metric"]:
        raise ValueError(f"Distance metric mismatch: {model_meta.get('distance_metric')} != {spec['distance_metric']}")
    if model_meta.get("is_external"):
        raise ValueError("Refusing to register external embedding model for unapproved medical content")
    row = conn.execute(
        """
        INSERT INTO medical_rag.embedding_models
            (provider, model_name, model_revision, input_profile, dimensions, distance_metric, is_external)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (provider, model_name, model_revision, input_profile, dimensions, distance_metric)
        DO UPDATE SET is_external = EXCLUDED.is_external
        RETURNING id
        """,
        (
            model_meta["provider"],
            model_meta["model_name"],
            model_meta["model_revision"],
            model_meta["input_profile"],
            model_meta["dimensions"],
            model_meta["distance_metric"],
            bool(model_meta.get("is_external", False)),
        ),
    ).fetchone()
    return row[0], spec


def _derive_section_text(sec, page_by_num):
    parts = []
    for anchor in sec.get("anchors", []):
        page = page_by_num.get(anchor["physical_page"])
        if not page:
            continue
        src = page.get("source_text", "")
        a, b = anchor.get("char_start"), anchor.get("char_end")
        if a is not None and b is not None and 0 <= a < b <= len(src):
            parts.append(src[a:b])
    return "\n\n".join(parts)


def _validate_page_and_chunk_invariants(pages, chunks, emb_by_chunk, spec):
    page_lookup = {(p["asset_key"], p["physical_page"]): p for p in pages}
    cited_ranges = defaultdict(list)
    for chunk in chunks:
        for span in chunk.get("spans", []):
            cited_ranges[(chunk["asset_key"], span["physical_page"])].append(
                (span.get("char_start"), span.get("char_end"))
            )

    sanitized_nul_pages = []
    for page in pages:
        src = page.get("source_text", "")
        if "\x00" in src:
            key = (page["asset_key"], page["physical_page"])
            ranges = cited_ranges.get(key, [])
            for idx, ch in enumerate(src):
                if ch == "\x00" and any(
                    a is not None and b is not None and a <= idx < b for a, b in ranges
                ):
                    raise ValueError(
                        f"PostgreSQL-incompatible NUL (0x00) byte detected in cited span of "
                        f"source_text({page['asset_key']}:{page['physical_page']}); refusing silent offset shift"
                    )
            # Uncited NULs occur only in quarantined needs_review blocks; 1:1 U+FFFD keeps all offsets exact.
            page["source_text"] = src.replace("\x00", "\ufffd")
            sanitized_nul_pages.append(f"{page['asset_key']}:{page['physical_page']}")
        _check_no_nul(page.get("printed_page_label"), f"printed_page_label({page['asset_key']}:{page['physical_page']})")

    for chunk in chunks:
        _check_no_nul(chunk["retrieval_text"], f"retrieval_text({chunk['chunk_id']})")
        _check_no_nul(chunk.get("heading"), f"heading({chunk['chunk_id']})")
        if not chunk["retrieval_text"].strip():
            raise ValueError(f"Empty retrieval_text on chunk {chunk['chunk_id']}")
        if chunk.get("table_id") is not None:
            raise ValueError(f"Held table_id unexpectedly attached to chunk {chunk['chunk_id']}")
        if not chunk.get("spans"):
            raise ValueError(f"Chunk {chunk['chunk_id']} has no citation spans")
        for span in chunk["spans"]:
            page = page_lookup.get((chunk["asset_key"], span["physical_page"]))
            if not page:
                raise ValueError(f"Span page missing for chunk {chunk['chunk_id']}: {span['physical_page']}")
            src = page["source_text"]
            a, b = span.get("char_start"), span.get("char_end")
            if a is None or b is None or a < 0 or b <= a or b > len(src):
                raise ValueError(f"Invalid span offsets [{a}, {b}) on chunk {chunk['chunk_id']}")
            if not valid_box(span.get("bbox")):
                raise ValueError(f"Invalid span bbox on chunk {chunk['chunk_id']}: {span.get('bbox')}")
        emb = emb_by_chunk.get(chunk["chunk_id"])
        if emb is None:
            raise ValueError(f"Missing embedding for chunk {chunk['chunk_id']}")
        if emb.get("model_name") != spec["model_name"] or emb.get("model_revision") != spec["model_revision"]:
            raise ValueError(f"Embedding model/revision mismatch on chunk {chunk['chunk_id']}")
        validate_vector(emb["embedding"], spec["dimensions"], distance_metric=spec["distance_metric"])
        _sha256_bytes(emb["input_sha256"], f"input_sha256({chunk['chunk_id']})")
    return sanitized_nul_pages


def _store_version_metadata(conn, col, rec_id, entry, manifest_items, source_id):
    meta = entry.get("metadata") or entry.get("original_metadata") or {}
    if entry.get("validation_state") != "valid" or meta.get("intake_review") == "hold":
        raise ValueError(f"Refusing to load held or invalid inventory record: {col}:{rec_id}")
    if entry.get("classification") == "unresolved":
        raise ValueError(f"Refusing to load unresolved inventory record: {col}:{rec_id}")

    title = meta["title"]
    _check_no_nul(title, "title")
    lang = meta["language"].lower()
    identity_key = _derive_identity_key(entry, col, rec_id)
    doc_id = conn.execute(
        """
        INSERT INTO medical_rag.documents (issuer_id, identity_key, canonical_title, language_tag)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (issuer_id, identity_key, language_tag) DO UPDATE
            SET canonical_title = EXCLUDED.canonical_title
        RETURNING id
        """,
        (source_id, identity_key, meta.get("canonical_title") or title, lang),
    ).fetchone()[0]

    # Preserve all cross-collection category associations.
    cats = set(meta.get("categories", [col]))
    for item in manifest_items:
        cats.update(item.get("target_categories", []))
    for cat in sorted(cats):
        conn.execute(
            "INSERT INTO medical_rag.document_categories (document_id, category_code) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (doc_id, cat),
        )

    # Verified document-scoped identifiers.
    for ident in meta.get("identifiers", []):
        if ident.get("verified") and ident.get("scope") == "document" and ident.get("scheme") != "as_recorded":
            ns = ident.get("namespace")
            ns_source_id = source_id if not ns else _upsert_source(
                conn, source_key=ns, name=ns, kind="registry", country_code=meta["issuer"].get("country")
            )
            conn.execute(
                """
                INSERT INTO medical_rag.document_identifiers
                    (source_id, identifier_scheme, identifier_value, document_id)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_id, identifier_scheme, identifier_value) DO UPDATE
                    SET document_id = EXCLUDED.document_id
                """,
                (ns_source_id, ident["scheme"], ident["value"], doc_id),
            )

    primary_item = next((i for i in manifest_items if i["asset_role"] == "primary"), manifest_items[0])
    primary_sha = primary_item["sha256"]
    revision_key = _derive_revision_key(entry, rec_id, primary_sha)
    scope = meta.get("applicability", {})
    geo_scope = scope.get("geography", "unknown")
    pop_scope = scope.get("population", "unknown")
    reuse = meta.get("reuse", {})
    use_status = reuse.get("use_status", "unknown")
    ext_status = reuse.get("external_embedding_status", "unknown")
    if use_status in {"restricted", "prohibited"}:
        raise ValueError(f"Refusing restricted/prohibited version {col}:{rec_id}")

    # Preserve non-privileged staging defaults; never fabricate approval or permission.
    ver_row = conn.execute(
        """
        INSERT INTO medical_rag.document_versions (
            document_id, revision_key, edition_label, title_as_published, official_url,
            published_on, effective_from, effective_until, geographic_scope, population_scope,
            license_code, license_url, license_note, use_status, external_embedding_status
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (document_id, revision_key) DO UPDATE
            SET edition_label = COALESCE(EXCLUDED.edition_label, medical_rag.document_versions.edition_label)
        RETURNING id
        """,
        (
            doc_id,
            revision_key,
            meta.get("edition"),
            title,
            meta.get("official_url"),
            _full_date_or_none(meta.get("publication_date")),
            _full_date_or_none(meta.get("effective_from")),
            _full_date_or_none(meta.get("effective_until")),
            geo_scope,
            pop_scope,
            reuse.get("license_code"),
            reuse.get("license_url"),
            reuse.get("license_note"),
            use_status,
            ext_status,
        ),
    ).fetchone()
    version_id = ver_row[0]

    if geo_scope == "country_list":
        for cc in scope.get("countries", []):
            conn.execute(
                "INSERT INTO medical_rag.version_countries (version_id, country_code) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (version_id, cc),
            )

    for topic in meta.get("topics", []):
        topic_id = conn.execute(
            """
            INSERT INTO medical_rag.medical_topics (vocabulary, code, label)
            VALUES (%s, %s, %s)
            ON CONFLICT (vocabulary, code) DO UPDATE SET label = EXCLUDED.label
            RETURNING id
            """,
            (topic["vocabulary"], topic["code"], topic["label"]),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO medical_rag.version_topics (version_id, topic_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (version_id, topic_id),
        )

    if scope.get("rules"):
        from psycopg.types.range import Range

    for rule_no, rule in enumerate(scope.get("rules", []), 1):
        age = rule.get("age_days")
        age_range = Range(age.get("min_inclusive", 0), age.get("max_exclusive"), "[)") if age else None
        conn.execute(
            """
            INSERT INTO medical_rag.version_applicability
                (version_id, rule_no, effect, population_label, age_days, sex, pregnancy_status, care_setting, criteria_text)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (version_id, rule_no) DO NOTHING
            """,
            (
                version_id,
                rule_no,
                rule["effect"],
                rule["population_label"],
                age_range,
                rule.get("sex"),
                rule.get("pregnancy_status"),
                rule.get("care_setting"),
                rule.get("criteria_text"),
            ),
        )

    # Register every eligible asset under this version (primary + correction/supplement annexes).
    asset_ids = {}
    asset_meta_by_path = {a["path"]: a for a in meta.get("assets", [])}
    for item in manifest_items:
        a_meta = asset_meta_by_path.get(item["asset_path"], {})
        a_reuse = a_meta.get("reuse", {})
        a_use = a_reuse.get("use_status", item.get("document_use_status", "unknown"))
        a_ext = a_reuse.get("external_embedding_status", "unknown")
        if a_use in {"restricted", "prohibited"}:
            raise ValueError(f"Refusing restricted/prohibited asset {item['asset_key']}")
        sha_bytes = _sha256_bytes(item["sha256"], item["asset_key"])
        uri = _portable_uri(item["collection"], item["asset_path"])
        aid = conn.execute(
            """
            INSERT INTO medical_rag.document_assets (
                version_id, sha256, storage_uri, media_type, byte_size, role,
                license_code, license_url, license_note, use_status, external_embedding_status
            )
            VALUES (%s, %s, %s, 'application/pdf', %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (version_id, sha256) DO UPDATE
                SET storage_uri = EXCLUDED.storage_uri
            RETURNING id
            """,
            (
                version_id,
                sha_bytes,
                uri,
                item["bytes"],
                item["asset_role"],
                a_reuse.get("license_code"),
                a_reuse.get("license_url"),
                a_reuse.get("license_note"),
                a_use,
                a_ext,
            ),
        ).fetchone()[0]
        asset_ids[item["asset_key"]] = aid

    return doc_id, version_id, asset_ids


def _record_attempt(conn, *, work_key, stage, status, source_id=None, requested_uri=None, resolved_uri=None,
                    source_metadata=None, observed_sha256=None, version_id=None, asset_id=None,
                    extraction_id=None, model_id=None, error_code=None, error_detail=None):
    next_no = conn.execute(
        "SELECT COALESCE(MAX(attempt_no), 0) + 1 FROM medical_rag.ingestion_attempts WHERE work_key = %s",
        (work_key,),
    ).fetchone()[0]
    conn.execute(
        """
        INSERT INTO medical_rag.ingestion_attempts (
            work_key, attempt_no, stage, status, source_id, requested_uri, resolved_uri,
            source_metadata, observed_sha256, version_id, asset_id, extraction_id, model_id,
            finished_at, error_code, error_detail
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, clock_timestamp(), %s, %s)
        """,
        (
            work_key,
            next_no,
            stage,
            status,
            source_id,
            requested_uri,
            resolved_uri,
            json.dumps(source_metadata, ensure_ascii=False) if source_metadata is not None else None,
            observed_sha256,
            version_id,
            asset_id,
            extraction_id,
            model_id,
            error_code,
            error_detail,
        ),
    )


def _clear_staged_extraction(conn, extraction_id):
    """Remove incomplete staged children before retrying an interrupted extraction."""
    conn.execute("DELETE FROM medical_rag.chunk_embeddings WHERE chunk_id IN (SELECT id FROM medical_rag.chunks WHERE extraction_id = %s)", (extraction_id,))
    conn.execute("DELETE FROM medical_rag.chunk_spans WHERE extraction_id = %s", (extraction_id,))
    conn.execute("DELETE FROM medical_rag.chunks WHERE extraction_id = %s", (extraction_id,))
    conn.execute("DELETE FROM medical_rag.extracted_tables WHERE extraction_id = %s", (extraction_id,))
    conn.execute("DELETE FROM medical_rag.sections WHERE extraction_id = %s", (extraction_id,))
    conn.execute("DELETE FROM medical_rag.extracted_pages WHERE extraction_id = %s", (extraction_id,))
    conn.execute(
        """
        DELETE FROM medical_rag.extraction_assets ea
        WHERE ea.extraction_id = %s
          AND NOT EXISTS (
              SELECT 1 FROM medical_rag.ingestion_attempts ia
              WHERE ia.extraction_id = ea.extraction_id AND ia.asset_id = ea.asset_id
          )
        """,
        (extraction_id,),
    )


def load_pipeline_outputs(
    conn,
    *,
    manifest_path,
    extract_dir,
    chunk_dir,
    embed_dir,
    inventory_paths=None,
    pipeline_name="medquery_ingestion",
    pipeline_version="phase7_v1",
):
    """Load validated metadata, extraction outputs, chunks, spans, and embeddings into PostgreSQL."""
    start = perf_counter()
    (
        manifest,
        eligible_by_key,
        inv_by_rec,
        pages,
        sections,
        review_items,
        chunks,
        emb_model,
        embeddings,
    ) = _resolve_inputs(manifest_path, inventory_paths, extract_dir, chunk_dir, embed_dir)

    spec = get_model_spec(emb_model["model_name"], revision=emb_model.get("model_revision"))
    emb_by_chunk = {row["chunk_id"]: row for row in embeddings}
    sanitized_nul_pages = _validate_page_and_chunk_invariants(pages, chunks, emb_by_chunk, spec)

    model_id, _ = _upsert_embedding_model(conn, emb_model)

    # Group loaded pages/sections/chunks/reviews by owning (collection, record_id) version.
    pages_by_ver = defaultdict(list)
    for page in pages:
        item = eligible_by_key.get(page["asset_key"])
        if not item:
            raise ValueError(f"Page asset {page['asset_key']} is not eligible in selection manifest")
        pages_by_ver[(item["collection"], item["record_id"])].append(page)

    sections_by_ver = defaultdict(list)
    for sec in sections:
        item = eligible_by_key[sec["asset_key"]]
        sections_by_ver[(item["collection"], item["record_id"])].append(sec)

    chunks_by_ver = defaultdict(list)
    for chk in chunks:
        item = eligible_by_key[chk["asset_key"]]
        chunks_by_ver[(item["collection"], item["record_id"])].append(chk)

    reviews_by_ver = defaultdict(list)
    for rev in review_items:
        item = eligible_by_key.get(rev["asset_key"])
        if item:
            reviews_by_ver[(item["collection"], item["record_id"])].append(rev)

    # Sort manifest items per version for bundle handling and completeness tracking.
    all_manifest_by_ver = defaultdict(list)
    for item in manifest.get("eligible_assets", []):
        all_manifest_by_ver[(item["collection"], item["record_id"])].append(item)

    manifest_by_ver = defaultdict(list)
    for key in sorted({p["asset_key"] for p in pages}):
        item = eligible_by_key[key]
        manifest_by_ver[(item["collection"], item["record_id"])].append(item)

    counts = {
        "versions_processed": 0,
        "extractions_created": 0,
        "extractions_reused": 0,
        "pages_inserted": 0,
        "sections_inserted": 0,
        "tables_inserted": 0,
        "chunks_inserted": 0,
        "spans_inserted": 0,
        "embeddings_inserted": 0,
        "attempts_recorded": 0,
    }
    doc_ids_by_rec = {}
    ver_ids_by_rec = {}

    for col, rec_id in sorted(pages_by_ver):
        entry = inv_by_rec.get((col, rec_id))
        if not entry:
            raise ValueError(f"Missing validated inventory record for {col}:{rec_id}")
        meta = entry.get("metadata") or entry.get("original_metadata") or {}
        issuer = meta["issuer"]
        source_id = _upsert_source(
            conn,
            source_key=_derive_source_key(issuer),
            name=issuer["name"],
            kind=issuer.get("kind", "unknown"),
            homepage_url=issuer.get("homepage"),
            country_code=issuer.get("country"),
        )
        m_items = sorted(manifest_by_ver[(col, rec_id)], key=lambda i: (i["asset_role"] != "primary", i["asset_key"]))
        doc_id, version_id, asset_ids = _store_version_metadata(conn, col, rec_id, entry, m_items, source_id)
        doc_ids_by_rec[(col, rec_id)] = doc_id
        ver_ids_by_rec[(col, rec_id)] = version_id
        counts["versions_processed"] += 1

        ver_pages = sorted(pages_by_ver[(col, rec_id)], key=lambda p: (p["asset_key"], p["physical_page"]))
        ver_sections = sorted(sections_by_ver[(col, rec_id)], key=lambda s: (s["asset_key"], s["ordinal"]))
        ver_chunks = sorted(chunks_by_ver[(col, rec_id)], key=lambda c: (c["ordinal"], c["chunk_id"]))

        input_fp = bytes.fromhex(fingerprint({
            "assets": [(i["asset_key"], i["sha256"], [p["physical_page"] for p in ver_pages if p["asset_key"] == i["asset_key"]]) for i in m_items],
        }))
        config_fp = bytes.fromhex(fingerprint({
            "pipeline": pipeline_version,
            "chunk_strategy": ver_chunks[0]["strategy"] if ver_chunks else "section_aware",
            "model_config": emb_model.get("model_config_fingerprint"),
        }))

        existing_ext = conn.execute(
            """
            SELECT id, status FROM medical_rag.extractions
            WHERE version_id = %s AND pipeline_name = %s AND pipeline_version = %s
              AND input_fingerprint = %s AND config_fingerprint = %s
            """,
            (version_id, pipeline_name, pipeline_version, input_fp, config_fp),
        ).fetchone()

        unchunkable_already_loaded = (
            existing_ext is not None
            and not ver_chunks
            and conn.execute(
                "SELECT count(*) FROM medical_rag.extracted_pages WHERE extraction_id = %s",
                (existing_ext[0],),
            ).fetchone()[0] == len(ver_pages)
            and conn.execute(
                "SELECT 1 FROM medical_rag.ingestion_attempts WHERE extraction_id = %s LIMIT 1",
                (existing_ext[0],),
            ).fetchone() is not None
        )

        if existing_ext and (existing_ext[1] == "completed" or unchunkable_already_loaded):
            extraction_id = existing_ext[0]
            counts["extractions_reused"] += 1
            if not ver_chunks:
                continue
            # Ensure any new embedding model rows for existing chunks are populated idempotently.
            db_chunks = conn.execute(
                "SELECT id, ordinal FROM medical_rag.chunks WHERE extraction_id = %s ORDER BY ordinal",
                (extraction_id,),
            ).fetchall()
            if len(db_chunks) != len(ver_chunks):
                raise ValueError(f"Chunk count mismatch on completed extraction {extraction_id}")
            for (db_chunk_id, _), chk in zip(db_chunks, ver_chunks):
                emb = emb_by_chunk[chk["chunk_id"]]
                res = conn.execute(
                    """
                    INSERT INTO medical_rag.chunk_embeddings
                        (chunk_id, model_id, dimensions, distance_metric, embedding, input_sha256)
                    VALUES (%s, %s, %s, %s, %s::public.vector, %s)
                    ON CONFLICT (chunk_id, model_id) DO NOTHING
                    """,
                    (
                        db_chunk_id,
                        model_id,
                        spec["dimensions"],
                        spec["distance_metric"],
                        _vector_literal(emb["embedding"]),
                        bytes.fromhex(emb["input_sha256"]),
                    ),
                )
                counts["embeddings_inserted"] += res.rowcount
            continue

        if existing_ext:
            extraction_id = existing_ext[0]
            _clear_staged_extraction(conn, extraction_id)
        else:
            extraction_id = conn.execute(
                """
                INSERT INTO medical_rag.extractions
                    (version_id, pipeline_name, pipeline_version, input_fingerprint, config_fingerprint)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (version_id, pipeline_name, pipeline_version, input_fp, config_fp),
            ).fetchone()[0]
            counts["extractions_created"] += 1

        for asset_key, aid in asset_ids.items():
            conn.execute(
                """
                INSERT INTO medical_rag.extraction_assets (extraction_id, asset_id, version_id)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (extraction_id, aid, version_id),
            )

        page_ids = {}
        page_by_asset_num = defaultdict(dict)
        for page in ver_pages:
            aid = asset_ids[page["asset_key"]]
            pid = conn.execute(
                """
                INSERT INTO medical_rag.extracted_pages
                    (extraction_id, asset_id, page_index, printed_page_label, source_text)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (extraction_id, aid, page["physical_page"], page.get("printed_page_label"), page.get("source_text", "")),
            ).fetchone()[0]
            page_ids[(page["asset_key"], page["physical_page"])] = pid
            page_by_asset_num[page["asset_key"]][page["physical_page"]] = page
            counts["pages_inserted"] += 1

        # Insert sections with global per-extraction ordinal and parent-before-child ordering.
        section_ids = {}
        for sec_ord, sec in enumerate(ver_sections, 1):
            ak = sec["asset_key"]
            aid = asset_ids[ak]
            start_pid = page_ids[(ak, sec["start_page"])]
            end_pid = page_ids[(ak, sec["end_page"])]
            parent_db_id = section_ids.get(sec.get("parent_section_key")) if sec.get("parent_section_key") else None
            text_content = _derive_section_text(sec, page_by_asset_num[ak])
            sid = conn.execute(
                """
                INSERT INTO medical_rag.sections (
                    extraction_id, parent_section_id, asset_id, ordinal, heading,
                    section_label, start_page_id, end_page_id, text_content, applicability_note
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    extraction_id,
                    parent_db_id,
                    aid,
                    sec_ord,
                    sec.get("heading"),
                    sec.get("section_label"),
                    start_pid,
                    end_pid,
                    text_content,
                    None,
                ),
            ).fetchone()[0]
            section_ids[sec["section_key"]] = sid
            counts["sections_inserted"] += 1

        # Store structured table candidates separately (never joined to chunks.table_id).
        table_ord = 0
        for page in ver_pages:
            ak = page["asset_key"]
            aid = asset_ids[ak]
            pid = page_ids[(ak, page["physical_page"])]
            for tbl in page.get("tables", []):
                table_ord += 1
                footnotes = "\n".join(str(x) for x in tbl.get("footnote_candidates", [])) or None
                conn.execute(
                    """
                    INSERT INTO medical_rag.extracted_tables (
                        extraction_id, section_id, asset_id, ordinal, source_label,
                        caption, start_page_id, end_page_id, grid, footnotes
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                    """,
                    (
                        extraction_id,
                        None,
                        aid,
                        table_ord,
                        tbl.get("key"),
                        tbl.get("caption"),
                        pid,
                        pid,
                        json.dumps(tbl.get("grid") or {"candidates": []}, ensure_ascii=False),
                        footnotes,
                    ),
                )
                counts["tables_inserted"] += 1

        # Insert chunks + spans + embeddings (re-indexing ordinal 1..N within this extraction).
        for chk_ord, chk in enumerate(ver_chunks, 1):
            sec_db_id = section_ids.get(chk.get("section_key")) if chk.get("section_key") else None
            db_chunk_id = conn.execute(
                """
                INSERT INTO medical_rag.chunks
                    (extraction_id, section_id, table_id, ordinal, language_tag, retrieval_text, applicability_note)
                VALUES (%s, %s, NULL, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    extraction_id,
                    sec_db_id,
                    chk_ord,
                    chk["language_tag"].lower(),
                    chk["retrieval_text"],
                    chk.get("applicability_note"),
                ),
            ).fetchone()[0]
            counts["chunks_inserted"] += 1

            for span in chk["spans"]:
                pid = page_ids[(chk["asset_key"], span["physical_page"])]
                conn.execute(
                    """
                    INSERT INTO medical_rag.chunk_spans (
                        chunk_id, span_no, extraction_id, page_id, char_start, char_end,
                        source_anchor, bbox, table_cells
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, NULL)
                    """,
                    (
                        db_chunk_id,
                        span["span_no"],
                        extraction_id,
                        pid,
                        span["char_start"],
                        span["char_end"],
                        span.get("source_anchor") or chk["chunk_id"],
                        json.dumps(span["bbox"]),
                    ),
                )
                counts["spans_inserted"] += 1

            emb = emb_by_chunk[chk["chunk_id"]]
            conn.execute(
                """
                INSERT INTO medical_rag.chunk_embeddings
                    (chunk_id, model_id, dimensions, distance_metric, embedding, input_sha256)
                VALUES (%s, %s, %s, %s, %s::public.vector, %s)
                """,
                (
                    db_chunk_id,
                    model_id,
                    spec["dimensions"],
                    spec["distance_metric"],
                    _vector_literal(emb["embedding"]),
                    bytes.fromhex(emb["input_sha256"]),
                ),
            )
            counts["embeddings_inserted"] += 1

        # Complete extraction only if chunks and spans exist (enforced by validate_extraction trigger).
        if ver_chunks:
            conn.execute("UPDATE medical_rag.extractions SET status = 'completed' WHERE id = %s", (extraction_id,))

        primary_item = m_items[0]
        expected_bundle = sorted(all_manifest_by_ver[(col, rec_id)], key=lambda i: i["asset_key"])
        expected_pages_by_asset = {i["asset_key"]: i["pages"] for i in expected_bundle}
        extracted_pages_by_asset = {
            i["asset_key"]: sorted(p["physical_page"] for p in ver_pages if p["asset_key"] == i["asset_key"])
            for i in m_items
        }
        is_complete = (
            set(expected_pages_by_asset) == set(extracted_pages_by_asset)
            and all(
                extracted_pages_by_asset[ak] == list(range(1, expected_pages_by_asset[ak] + 1))
                for ak in expected_pages_by_asset
            )
        )
        extended_meta = {
            "collection": col,
            "record_id": rec_id,
            "document_type": meta.get("document_type"),
            "authority": meta.get("authority"),
            "phase2_classification": entry.get("classification"),
            "content_languages": meta.get("content_languages", []),
            "publication_date_raw": meta.get("publication_date"),
            "relations": meta.get("relations", []),
            "target_categories": primary_item.get("target_categories", [col]),
            "review_items_count": len(reviews_by_ver[(col, rec_id)]),
            "review_codes": sorted({r.get("code") for r in reviews_by_ver[(col, rec_id)] if r.get("code")}),
            "selected_pages": [p["physical_page"] for p in ver_pages],
            "extraction_completeness": "complete" if is_complete else "partial",
            "expected_pages_by_asset": expected_pages_by_asset,
            "extracted_pages_by_asset": extracted_pages_by_asset,
            "metadata_digest": fingerprint(meta),
        }
        attempt_status = "succeeded" if ver_chunks and not reviews_by_ver[(col, rec_id)] else (
            "needs_review" if reviews_by_ver[(col, rec_id)] else "skipped"
        )
        _record_attempt(
            conn,
            work_key=f"load:{col}:{rec_id}",
            stage="embed" if ver_chunks else "extract",
            status=attempt_status,
            source_id=source_id,
            requested_uri=None,
            resolved_uri=primary_item.get("acquisition_url"),
            source_metadata=extended_meta,
            observed_sha256=_sha256_bytes(primary_item["sha256"]),
            version_id=version_id,
            asset_id=asset_ids[primary_item["asset_key"]],
            extraction_id=extraction_id,
            model_id=model_id,
        )
        counts["attempts_recorded"] += 1

    # Second pass: resolve verified translation_of_id / replaces_version_id when both endpoints exist.
    for col, rec_id in sorted(pages_by_ver):
        entry = inv_by_rec[(col, rec_id)]
        meta = entry.get("metadata") or {}
        doc_id = doc_ids_by_rec[(col, rec_id)]
        ver_id = ver_ids_by_rec[(col, rec_id)]
        for rel in meta.get("relations", []):
            if rel.get("state") != "verified":
                continue
            target = rel["target"]
            t_col, _, t_rec = target.partition(":")
            if not t_rec:
                t_col, t_rec = col, target
            if rel["type"] == "translation_of" and (t_col, t_rec) in doc_ids_by_rec:
                target_doc_id = doc_ids_by_rec[(t_col, t_rec)]
                if target_doc_id != doc_id:
                    conn.execute(
                        "UPDATE medical_rag.documents SET translation_of_id = %s WHERE id = %s AND translation_of_id IS NULL",
                        (target_doc_id, doc_id),
                    )
            elif rel["type"] == "replaces" and (t_col, t_rec) in ver_ids_by_rec:
                target_ver_id = ver_ids_by_rec[(t_col, t_rec)]
                if target_ver_id != ver_id:
                    conn.execute(
                        "UPDATE medical_rag.document_versions SET replaces_version_id = %s WHERE id = %s AND replaces_version_id IS NULL",
                        (target_ver_id, ver_id),
                    )

    counts["elapsed_seconds"] = round(perf_counter() - start, 4)
    counts["model_id"] = model_id
    counts["model_name"] = spec["model_name"]
    counts["model_revision"] = spec["model_revision"]
    counts["sanitized_nul_pages"] = sanitized_nul_pages
    return counts


def research_retrieve(
    conn,
    query_vector,
    *,
    model_name=DEFAULT_MODEL,
    model_revision=None,
    limit=5,
    language_tag=None,
    allow_research_only=False,
    exclude_educational=True,
):
    """Research-only exact cosine similarity query over staged local chunks with full source citations.

    Authoritative clinical serving MUST query medical_rag.retrievable_chunks / retrievable_citations instead.
    """
    if not allow_research_only:
        raise PermissionError(
            "research_retrieve accesses unapproved staged research data; pass allow_research_only=True explicitly."
        )
    spec = get_model_spec(model_name, revision=model_revision)
    validate_vector(query_vector, spec["dimensions"], distance_metric=spec["distance_metric"])
    vec_sql = _vector_literal(query_vector)

    rows = conn.execute(
        """
        WITH ranked AS (
            SELECT c.id AS chunk_id,
                   c.extraction_id,
                   c.ordinal,
                   c.language_tag,
                   c.retrieval_text,
                   v.id AS version_id,
                   v.status AS version_status,
                   v.use_status,
                   v.title_as_published,
                   v.edition_label,
                   d.id AS document_id,
                   issuer.name AS issuer_name,
                   sec.section_label,
                   sec.heading,
                   1.0 - (ce.embedding <=> %s::public.vector) AS cosine_similarity,
                   ia.source_metadata
            FROM medical_rag.chunk_embeddings ce
            JOIN medical_rag.embedding_models m ON m.id = ce.model_id
            JOIN medical_rag.chunks c ON c.id = ce.chunk_id
            JOIN medical_rag.extractions e ON e.id = c.extraction_id
            JOIN medical_rag.document_versions v ON v.id = e.version_id
            JOIN medical_rag.documents d ON d.id = v.document_id
            JOIN medical_rag.sources issuer ON issuer.id = d.issuer_id
            LEFT JOIN medical_rag.sections sec ON sec.id = c.section_id
            LEFT JOIN LATERAL (
                SELECT source_metadata
                FROM medical_rag.ingestion_attempts
                WHERE extraction_id = e.id
                ORDER BY attempt_no DESC, id DESC
                LIMIT 1
            ) ia ON true
            WHERE m.model_name = %s
              AND m.model_revision = %s
              AND e.status = 'completed'
              AND c.table_id IS NULL
              AND v.use_status NOT IN ('restricted', 'prohibited')
              AND (%s::text IS NULL OR c.language_tag = lower(%s::text))
              AND (NOT %s OR COALESCE(ia.source_metadata->>'phase2_classification', '') <> 'educational')
            ORDER BY ce.embedding <=> %s::public.vector, c.id
            LIMIT %s
        )
        SELECT r.chunk_id, r.ordinal, r.language_tag, r.retrieval_text, r.version_status,
               r.use_status, r.title_as_published, r.edition_label, r.issuer_name,
               r.section_label, r.heading, r.cosine_similarity, r.source_metadata,
               s.span_no, p.page_index, p.printed_page_label, a.storage_uri, a.sha256,
               s.char_start, s.char_end, s.source_anchor, s.bbox,
               substring(p.source_text FROM s.char_start + 1 FOR s.char_end - s.char_start) AS cited_text
        FROM ranked r
        JOIN medical_rag.chunk_spans s ON s.chunk_id = r.chunk_id
        JOIN medical_rag.extracted_pages p ON p.id = s.page_id
        JOIN medical_rag.document_assets a ON a.id = p.asset_id
        ORDER BY r.cosine_similarity DESC, r.chunk_id, s.span_no
        """,
        (vec_sql, spec["model_name"], spec["model_revision"], language_tag, language_tag, exclude_educational, vec_sql, limit),
    ).fetchall()

    grouped = {}
    for row in rows:
        cid = row[0]
        if cid not in grouped:
            meta = row[12] or {}
            grouped[cid] = {
                "chunk_id": cid,
                "ordinal": row[1],
                "language_tag": row[2],
                "retrieval_text": row[3],
                "retrieval_mode": "research_only_unapproved",
                "version_status": row[4],
                "use_status": row[5],
                "title_as_published": row[6],
                "edition_label": row[7],
                "issuer_name": row[8],
                "section_label": row[9],
                "heading": row[10],
                "cosine_similarity": round(float(row[11]), 6),
                "document_type": meta.get("document_type"),
                "authority": meta.get("authority"),
                "phase2_classification": meta.get("phase2_classification"),
                "spans": [],
            }
        grouped[cid]["spans"].append({
            "span_no": row[13],
            "page_index": row[14],
            "printed_page_label": row[15],
            "storage_uri": row[16],
            "asset_sha256": bytes(row[17]).hex(),
            "char_start": row[18],
            "char_end": row[19],
            "source_anchor": row[20],
            "bbox": row[21],
            "cited_text": row[22],
        })
    return list(grouped.values())


def add_command(commands):
    command = commands.add_parser("load", help="Load validated extraction, chunk, and embedding outputs into PostgreSQL.")
    command.add_argument("--manifest", type=Path, required=True, help="Executable extraction selection manifest JSON.")
    command.add_argument("--inventory", type=Path, action="append", default=[],
                         help="Optional validated inventory.jsonl override (defaults to manifest inventories).")
    command.add_argument("--extract-dir", type=Path, required=True, help="Directory with pages.jsonl and sections.jsonl.")
    command.add_argument("--chunk-dir", type=Path, required=True, help="Directory with chunks.jsonl.")
    command.add_argument("--embed-dir", type=Path, required=True,
                         help="Directory with embedding_model.json and chunk_embeddings.jsonl.")
    command.add_argument("--pipeline-name", default="medquery_ingestion")
    command.add_argument("--pipeline-version", default="phase7_v1")
    command.add_argument("--dry-run", action="store_true",
                         help="Execute all inserts/validations inside a rolled-back transaction without committing.")
    return command


def main(args):
    with connect() as conn:
        with conn.transaction(force_rollback=args.dry_run):
            summary = load_pipeline_outputs(
                conn,
                manifest_path=args.manifest.resolve(),
                inventory_paths=[p.resolve() for p in args.inventory] or None,
                extract_dir=args.extract_dir.resolve(),
                chunk_dir=args.chunk_dir.resolve(),
                embed_dir=args.embed_dir.resolve(),
                pipeline_name=args.pipeline_name,
                pipeline_version=args.pipeline_version,
            )
            summary["dry_run"] = bool(args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0
