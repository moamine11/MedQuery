\set ON_ERROR_STOP on
BEGIN;
SELECT pg_advisory_xact_lock(730170001);
SELECT NOT EXISTS (
    SELECT 1 FROM medical_rag.schema_migrations WHERE version = 2
) AS apply_migration \gset
\if :apply_migration
SET LOCAL search_path = medical_rag, public;

CREATE FUNCTION preserve_asset_identity() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog, medical_rag, public, pg_temp AS $$
BEGIN
    IF (NEW.id, NEW.version_id, NEW.sha256, NEW.storage_uri, NEW.media_type, NEW.byte_size,
        NEW.role, NEW.created_at) IS DISTINCT FROM
       (OLD.id, OLD.version_id, OLD.sha256, OLD.storage_uri, OLD.media_type, OLD.byte_size,
        OLD.role, OLD.created_at) THEN
        RAISE EXCEPTION 'Asset identity is immutable';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER document_assets_identity BEFORE UPDATE ON document_assets
FOR EACH ROW EXECUTE FUNCTION preserve_asset_identity();

CREATE FUNCTION update_version_context() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog, medical_rag, public, pg_temp AS $$
BEGIN
    IF (NEW.id, NEW.document_id, NEW.revision_key, NEW.created_at) IS DISTINCT FROM
       (OLD.id, OLD.document_id, OLD.revision_key, OLD.created_at) THEN
        RAISE EXCEPTION 'Publisher revision identity is immutable';
    END IF;
    IF (NEW.edition_label, NEW.title_as_published, NEW.official_url, NEW.published_on,
        NEW.effective_from, NEW.effective_until, NEW.replaces_version_id,
        NEW.geographic_scope, NEW.population_scope) IS DISTINCT FROM
       (OLD.edition_label, OLD.title_as_published, OLD.official_url, OLD.published_on,
        OLD.effective_from, OLD.effective_until, OLD.replaces_version_id,
        OLD.geographic_scope, OLD.population_scope) THEN
        NEW.clinical_context_revision := OLD.clinical_context_revision + 1;
    ELSIF NEW.clinical_context_revision NOT IN (OLD.clinical_context_revision, OLD.clinical_context_revision + 1) THEN
        RAISE EXCEPTION 'Invalid clinical context revision change';
    END IF;
    IF NEW.status IS DISTINCT FROM OLD.status THEN
        NEW.status_changed_at := clock_timestamp();
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER document_versions_context BEFORE UPDATE ON document_versions
FOR EACH ROW EXECUTE FUNCTION update_version_context();

CREATE FUNCTION bump_version_context() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog, medical_rag, public, pg_temp AS $$
BEGIN
    IF TG_OP = 'UPDATE' AND NEW.version_id <> OLD.version_id THEN
        RAISE EXCEPTION 'Move clinical metadata by removing and adding the row';
    END IF;
    IF TG_OP = 'UPDATE' AND NEW IS NOT DISTINCT FROM OLD THEN
        RETURN NEW;
    END IF;
    UPDATE document_versions SET clinical_context_revision = clinical_context_revision + 1
    WHERE id = CASE WHEN TG_OP = 'DELETE' THEN OLD.version_id ELSE NEW.version_id END;
    RETURN NULL;
END;
$$;
CREATE TRIGGER version_countries_context AFTER INSERT OR UPDATE OR DELETE ON version_countries
FOR EACH ROW EXECUTE FUNCTION bump_version_context();
CREATE TRIGGER version_topics_context AFTER INSERT OR UPDATE OR DELETE ON version_topics
FOR EACH ROW EXECUTE FUNCTION bump_version_context();
CREATE TRIGGER version_applicability_context AFTER INSERT OR UPDATE OR DELETE ON version_applicability
FOR EACH ROW EXECUTE FUNCTION bump_version_context();

CREATE FUNCTION protect_extraction_input() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog, medical_rag, public, pg_temp AS $$
DECLARE
    target_extraction bigint;
    extraction_status text;
BEGIN
    IF TG_OP = 'UPDATE' AND OLD.extraction_id <> NEW.extraction_id THEN
        RAISE EXCEPTION 'Cannot move a row between extractions';
    END IF;
    target_extraction := CASE WHEN TG_OP = 'DELETE' THEN OLD.extraction_id ELSE NEW.extraction_id END;
    SELECT status INTO extraction_status FROM extractions WHERE id = target_extraction FOR UPDATE;
    IF extraction_status = 'completed' THEN
        RAISE EXCEPTION 'Completed extraction % is immutable', target_extraction;
    END IF;
    IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER extraction_assets_protect BEFORE INSERT OR UPDATE OR DELETE ON extraction_assets
FOR EACH ROW EXECUTE FUNCTION protect_extraction_input();
CREATE TRIGGER extracted_pages_protect BEFORE INSERT OR UPDATE OR DELETE ON extracted_pages
FOR EACH ROW EXECUTE FUNCTION protect_extraction_input();
CREATE TRIGGER sections_protect BEFORE INSERT OR UPDATE OR DELETE ON sections
FOR EACH ROW EXECUTE FUNCTION protect_extraction_input();
CREATE TRIGGER extracted_tables_protect BEFORE INSERT OR UPDATE OR DELETE ON extracted_tables
FOR EACH ROW EXECUTE FUNCTION protect_extraction_input();
CREATE TRIGGER chunks_protect BEFORE INSERT OR UPDATE OR DELETE ON chunks
FOR EACH ROW EXECUTE FUNCTION protect_extraction_input();
CREATE TRIGGER chunk_spans_protect BEFORE INSERT OR UPDATE OR DELETE ON chunk_spans
FOR EACH ROW EXECUTE FUNCTION protect_extraction_input();

CREATE FUNCTION validate_extraction() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog, medical_rag, public, pg_temp AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF OLD.status = 'completed' THEN RAISE EXCEPTION 'Completed extractions are retained'; END IF;
        RETURN OLD;
    END IF;
    IF TG_OP = 'INSERT' THEN
        IF NEW.status <> 'staged' THEN RAISE EXCEPTION 'New extractions must be staged'; END IF;
        RETURN NEW;
    END IF;
    IF (NEW.id, NEW.version_id, NEW.pipeline_name, NEW.pipeline_version,
        NEW.input_fingerprint, NEW.config_fingerprint, NEW.created_at) IS DISTINCT FROM
       (OLD.id, OLD.version_id, OLD.pipeline_name, OLD.pipeline_version,
        OLD.input_fingerprint, OLD.config_fingerprint, OLD.created_at) OR
       (OLD.status = 'completed' AND
        (NEW.status, NEW.completed_at) IS DISTINCT FROM (OLD.status, OLD.completed_at)) THEN
        RAISE EXCEPTION 'Extraction identity and completed output are immutable';
    END IF;
    IF NEW.status = 'completed' AND OLD.status <> 'completed' THEN
        IF NOT EXISTS (SELECT 1 FROM chunks WHERE extraction_id = NEW.id) OR EXISTS (
            SELECT 1 FROM chunks c WHERE c.extraction_id = NEW.id
            AND NOT EXISTS (SELECT 1 FROM chunk_spans s WHERE s.chunk_id = c.id)
        ) THEN
            RAISE EXCEPTION 'Extraction % needs chunks and citations for each chunk', NEW.id;
        END IF;
        IF EXISTS (
            SELECT 1 FROM chunk_spans s JOIN extracted_pages p ON p.id = s.page_id
            JOIN chunks c ON c.id = s.chunk_id
            WHERE c.extraction_id = NEW.id
              AND (s.char_end > char_length(p.source_text) OR
                   (s.table_cells IS NOT NULL AND c.table_id IS NULL))
        ) THEN
            RAISE EXCEPTION 'Extraction % contains invalid citation offsets or table coordinates', NEW.id;
        END IF;
        IF EXISTS (
            SELECT 1 FROM sections s
            JOIN extracted_pages p1 ON p1.id = s.start_page_id
            JOIN extracted_pages p2 ON p2.id = s.end_page_id
            LEFT JOIN sections parent ON parent.id = s.parent_section_id
            WHERE s.extraction_id = NEW.id AND
                (p1.page_index > p2.page_index OR parent.ordinal >= s.ordinal)
        ) OR EXISTS (
            SELECT 1 FROM extracted_tables t
            JOIN extracted_pages p1 ON p1.id = t.start_page_id
            JOIN extracted_pages p2 ON p2.id = t.end_page_id
            WHERE t.extraction_id = NEW.id AND p1.page_index > p2.page_index
        ) THEN
            RAISE EXCEPTION 'Extraction % contains invalid page ranges or section hierarchy', NEW.id;
        END IF;
        NEW.completed_at := clock_timestamp();
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER extractions_validate BEFORE INSERT OR UPDATE OR DELETE ON extractions
FOR EACH ROW EXECUTE FUNCTION validate_extraction();

CREATE FUNCTION record_clinical_review() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog, medical_rag, public, pg_temp AS $$
DECLARE
    version_key bigint;
BEGIN
    IF TG_OP <> 'INSERT' THEN
        RAISE EXCEPTION 'Clinical review decisions are append-only';
    END IF;
    SELECT version_id INTO version_key FROM extractions
    WHERE id = NEW.extraction_id AND status = 'completed';
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Clinical reviews require a completed extraction';
    END IF;
    SELECT clinical_context_revision INTO NEW.clinical_context_revision
    FROM document_versions WHERE id = version_key FOR UPDATE;
    NEW.decided_at := clock_timestamp();
    RETURN NEW;
END;
$$;
CREATE TRIGGER clinical_reviews_record BEFORE INSERT OR UPDATE OR DELETE ON clinical_reviews
FOR EACH ROW EXECUTE FUNCTION record_clinical_review();

CREATE VIEW latest_clinical_reviews AS
SELECT DISTINCT ON (extraction_id) *
FROM clinical_reviews
ORDER BY extraction_id, decided_at DESC, id DESC;

CREATE VIEW extraction_permissions AS
SELECT e.id AS extraction_id, e.version_id,
       v.use_status = 'allowed' AND COALESCE(a.all_assets_allowed, false) AS retrieval_allowed,
       v.external_embedding_status = 'allowed' AND
           COALESCE(a.all_assets_external_allowed, false) AS external_embedding_allowed
FROM extractions e
JOIN document_versions v ON v.id = e.version_id
LEFT JOIN LATERAL (
    SELECT bool_and(asset.use_status = 'allowed') AS all_assets_allowed,
           bool_and(asset.external_embedding_status = 'allowed') AS all_assets_external_allowed
    FROM extraction_assets ea
    JOIN document_assets asset ON asset.id = ea.asset_id
    WHERE ea.extraction_id = e.id
) a ON true;

CREATE FUNCTION publish_extraction(target_extraction bigint, serving_model bigint)
RETURNS void LANGUAGE plpgsql SET search_path = pg_catalog, medical_rag, public, pg_temp AS $$
DECLARE
    target_version document_versions%ROWTYPE;
    approved_at timestamptz;
    retrieval_day date;
    model_external boolean;
BEGIN
    SELECT v.* INTO STRICT target_version FROM document_versions v
    JOIN extractions e ON e.version_id = v.id WHERE e.id = target_extraction;
    PERFORM 1 FROM documents WHERE id = target_version.document_id FOR UPDATE;
    SELECT * INTO STRICT target_version FROM document_versions WHERE id = target_version.id FOR UPDATE;
    PERFORM 1 FROM extractions WHERE id = target_extraction AND status = 'completed' FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'Publication requires a completed extraction'; END IF;
    PERFORM 1 FROM document_assets a JOIN extraction_assets ea ON ea.asset_id = a.id
    WHERE ea.extraction_id = target_extraction ORDER BY a.id FOR SHARE OF a;
    SELECT is_external INTO STRICT model_external FROM embedding_models WHERE id = serving_model FOR SHARE;

    approved_at := clock_timestamp();
    retrieval_day := approved_at::date;
    IF target_version.status NOT IN ('staged', 'current') OR
       COALESCE(target_version.effective_from > retrieval_day, false) OR
       COALESCE(target_version.effective_until <= retrieval_day, false) THEN
        RAISE EXCEPTION 'Version % is not publishable on %', target_version.id, retrieval_day;
    END IF;
    IF (target_version.geographic_scope = 'country_list') <> EXISTS (
        SELECT 1 FROM version_countries WHERE version_id = target_version.id
    ) OR (target_version.population_scope = 'restricted' AND NOT EXISTS (
        SELECT 1 FROM version_applicability WHERE version_id = target_version.id
    )) THEN
        RAISE EXCEPTION 'Version % has incomplete applicability metadata', target_version.id;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM extraction_permissions p WHERE p.extraction_id = target_extraction
          AND p.retrieval_allowed AND (NOT model_external OR p.external_embedding_allowed)
    ) THEN
        RAISE EXCEPTION 'Publication requires compatible reuse permissions for the version and all input assets';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM latest_clinical_reviews r WHERE r.extraction_id = target_extraction
          AND r.clinical_context_revision = target_version.clinical_context_revision
          AND r.decision = 'approved' AND (r.valid_until IS NULL OR r.valid_until > approved_at)
    ) THEN
        RAISE EXCEPTION 'Publication requires a current clinical approval';
    END IF;
    IF EXISTS (
        SELECT 1 FROM chunks c WHERE c.extraction_id = target_extraction
        AND NOT EXISTS (
            SELECT 1 FROM chunk_embeddings ce WHERE ce.chunk_id = c.id AND ce.model_id = serving_model
        )
    ) THEN
        RAISE EXCEPTION 'Publication requires complete embeddings for the serving model';
    END IF;

    UPDATE document_versions SET status = 'superseded'
    WHERE document_id = target_version.document_id AND status = 'current' AND id <> target_version.id;
    UPDATE extractions SET is_selected = false
    WHERE version_id = target_version.id AND is_selected AND id <> target_extraction;
    UPDATE extractions SET is_selected = true WHERE id = target_extraction AND NOT is_selected;
    UPDATE document_versions SET status = 'current' WHERE id = target_version.id AND status <> 'current';
END;
$$;

CREATE VIEW retrievable_chunks WITH (security_barrier = true) AS
SELECT c.*, v.id AS version_id, v.document_id, v.geographic_scope, v.population_scope
FROM chunks c
JOIN extractions e ON e.id = c.extraction_id
JOIN document_versions v ON v.id = e.version_id
JOIN extraction_permissions p ON p.extraction_id = e.id
JOIN latest_clinical_reviews r ON r.extraction_id = e.id
WHERE v.status = 'current' AND e.is_selected AND e.status = 'completed'
  AND (v.effective_from IS NULL OR v.effective_from <= statement_timestamp()::date)
  AND (v.effective_until IS NULL OR v.effective_until > statement_timestamp()::date)
  AND p.retrieval_allowed
  AND r.clinical_context_revision = v.clinical_context_revision AND r.decision = 'approved'
  AND (r.valid_until IS NULL OR r.valid_until > statement_timestamp());

CREATE VIEW retrievable_embeddings WITH (security_barrier = true) AS
SELECT ce.*, c.extraction_id, c.version_id, c.document_id
FROM chunk_embeddings ce
JOIN retrievable_chunks c ON c.id = ce.chunk_id
JOIN embedding_models m ON m.id = ce.model_id
JOIN extraction_permissions p ON p.extraction_id = c.extraction_id
WHERE NOT m.is_external OR p.external_embedding_allowed;

CREATE VIEW retrievable_citations WITH (security_barrier = true) AS
SELECT s.chunk_id, s.span_no, issuer.name AS issuer_name, v.title_as_published,
       v.edition_label, v.official_url, acquisition.source_url, p.asset_id, a.sha256 AS asset_sha256,
       p.page_index, p.printed_page_label, section.section_label, section.heading,
       t.source_label AS table_label, s.char_start, s.char_end, s.source_anchor, s.bbox, s.table_cells,
       CASE WHEN s.char_start IS NOT NULL THEN
           substring(p.source_text FROM s.char_start + 1 FOR s.char_end - s.char_start)
       END AS cited_text
FROM retrievable_chunks c
JOIN chunk_spans s ON s.chunk_id = c.id
JOIN extracted_pages p ON p.id = s.page_id
JOIN document_assets a ON a.id = p.asset_id
JOIN document_versions v ON v.id = c.version_id
JOIN documents d ON d.id = c.document_id
JOIN sources issuer ON issuer.id = d.issuer_id
LEFT JOIN LATERAL (
    SELECT COALESCE(i.resolved_uri, i.requested_uri) AS source_url
    FROM ingestion_attempts i
    WHERE i.asset_id = a.id AND i.status IN ('succeeded', 'skipped')
      AND i.stage IN ('fetch', 'identity')
      AND COALESCE(i.resolved_uri, i.requested_uri) IS NOT NULL
    ORDER BY (i.source_id IS DISTINCT FROM d.issuer_id), i.started_at, i.id
    LIMIT 1
) acquisition ON true
LEFT JOIN extracted_tables t ON t.id = c.table_id
LEFT JOIN sections section ON section.id = COALESCE(c.section_id, t.section_id);

INSERT INTO schema_migrations (version, name) VALUES (2, 'publication');
\endif
COMMIT;
