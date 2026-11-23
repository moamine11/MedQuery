\set ON_ERROR_STOP on
BEGIN;
SELECT pg_advisory_xact_lock(730170001);
CREATE SCHEMA IF NOT EXISTS medical_rag;
CREATE TABLE IF NOT EXISTS medical_rag.schema_migrations (
    version integer PRIMARY KEY,
    name text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
SELECT NOT EXISTS (
    SELECT 1 FROM medical_rag.schema_migrations WHERE version = 1
) AS apply_migration \gset
\if :apply_migration

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
SET LOCAL search_path = medical_rag, public;

CREATE DOMAIN sha256_digest AS bytea CHECK (octet_length(VALUE) = 32);
CREATE DOMAIN country_code AS text CHECK (VALUE ~ '^[A-Z]{2}$');
CREATE DOMAIN language_tag AS text CHECK (VALUE ~ '^[A-Za-z]{1,8}(-[A-Za-z0-9]{1,8})*$');
CREATE DOMAIN permission_status AS text
    CHECK (VALUE IN ('unknown', 'allowed', 'restricted', 'prohibited'));

CREATE TABLE sources (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_key text NOT NULL UNIQUE,
    name text NOT NULL,
    kind text NOT NULL,
    homepage_url text,
    country_code country_code
);

CREATE TABLE documents (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    issuer_id bigint NOT NULL REFERENCES sources,
    identity_key text NOT NULL,
    canonical_title text NOT NULL,
    language_tag language_tag NOT NULL CHECK (language_tag = lower(language_tag)),
    translation_of_id bigint REFERENCES documents,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (issuer_id, identity_key, language_tag),
    CHECK (translation_of_id <> id)
);

CREATE TABLE document_identifiers (
    source_id bigint NOT NULL REFERENCES sources,
    identifier_scheme text NOT NULL,
    identifier_value text NOT NULL,
    document_id bigint NOT NULL REFERENCES documents,
    PRIMARY KEY (source_id, identifier_scheme, identifier_value)
);

CREATE TABLE categories (
    code text PRIMARY KEY,
    name text NOT NULL UNIQUE
);

INSERT INTO categories (code, name) VALUES
    ('clinical_protocols_ministry_directives', 'Clinical Protocols & Ministry Directives'),
    ('treatment_protocols', 'Treatment Protocols'),
    ('therapeutic_guidelines', 'Therapeutic Guidelines'),
    ('pharmaceutical_formularies', 'Pharmaceutical Formularies'),
    ('who_guidelines', 'WHO Guidelines');

CREATE TABLE document_categories (
    document_id bigint NOT NULL REFERENCES documents,
    category_code text NOT NULL REFERENCES categories,
    PRIMARY KEY (document_id, category_code)
);

CREATE TABLE document_versions (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id bigint NOT NULL REFERENCES documents,
    revision_key text NOT NULL,
    edition_label text,
    title_as_published text NOT NULL,
    official_url text,
    published_on date,
    effective_from date,
    effective_until date,
    status text NOT NULL DEFAULT 'staged'
        CHECK (status IN ('staged', 'current', 'superseded', 'withdrawn')),
    status_changed_at timestamptz NOT NULL DEFAULT now(),
    replaces_version_id bigint,
    geographic_scope text NOT NULL DEFAULT 'unknown'
        CHECK (geographic_scope IN ('global', 'country_list', 'unknown')),
    population_scope text NOT NULL DEFAULT 'unknown'
        CHECK (population_scope IN ('general', 'restricted', 'unknown')),
    clinical_context_revision bigint NOT NULL DEFAULT 1 CHECK (clinical_context_revision > 0),
    license_code text,
    license_url text,
    license_note text,
    use_status permission_status NOT NULL DEFAULT 'unknown',
    external_embedding_status permission_status NOT NULL DEFAULT 'unknown',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, revision_key),
    UNIQUE (id, document_id),
    FOREIGN KEY (replaces_version_id, document_id) REFERENCES document_versions (id, document_id),
    CHECK (replaces_version_id <> id),
    CHECK (effective_until > effective_from)
);

CREATE UNIQUE INDEX document_versions_one_current ON document_versions (document_id)
    WHERE status = 'current';

CREATE TABLE version_countries (
    version_id bigint NOT NULL REFERENCES document_versions,
    country_code country_code NOT NULL,
    PRIMARY KEY (version_id, country_code)
);

CREATE TABLE medical_topics (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    vocabulary text NOT NULL,
    code text NOT NULL,
    label text NOT NULL,
    UNIQUE (vocabulary, code)
);

CREATE TABLE version_topics (
    version_id bigint NOT NULL REFERENCES document_versions,
    topic_id bigint NOT NULL REFERENCES medical_topics,
    PRIMARY KEY (version_id, topic_id)
);

CREATE TABLE version_applicability (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    version_id bigint NOT NULL REFERENCES document_versions,
    rule_no integer NOT NULL CHECK (rule_no > 0),
    effect text NOT NULL CHECK (effect IN ('include', 'exclude')),
    population_label text NOT NULL,
    age_days int4range,
    sex text,
    pregnancy_status text,
    care_setting text,
    criteria_text text,
    UNIQUE (version_id, rule_no),
    CHECK (age_days IS NULL OR (NOT isempty(age_days) AND age_days <@ '[0,)'::int4range))
);

CREATE TABLE document_assets (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    version_id bigint NOT NULL REFERENCES document_versions,
    sha256 sha256_digest NOT NULL,
    storage_uri text NOT NULL,
    media_type text NOT NULL,
    byte_size bigint NOT NULL CHECK (byte_size >= 0),
    role text NOT NULL CHECK (role IN ('primary', 'annex', 'alternate')),
    license_code text,
    license_url text,
    license_note text,
    use_status permission_status NOT NULL DEFAULT 'unknown',
    external_embedding_status permission_status NOT NULL DEFAULT 'unknown',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (version_id, sha256),
    UNIQUE (id, version_id)
);

CREATE TABLE extractions (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    version_id bigint NOT NULL REFERENCES document_versions,
    pipeline_name text NOT NULL,
    pipeline_version text NOT NULL,
    input_fingerprint sha256_digest NOT NULL,
    config_fingerprint sha256_digest NOT NULL,
    status text NOT NULL DEFAULT 'staged' CHECK (status IN ('staged', 'completed', 'failed')),
    is_selected boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    UNIQUE (version_id, pipeline_name, pipeline_version, input_fingerprint, config_fingerprint),
    UNIQUE (id, version_id),
    CHECK ((status = 'completed') = (completed_at IS NOT NULL)),
    CHECK (completed_at >= created_at),
    CHECK (NOT is_selected OR status = 'completed')
);

CREATE UNIQUE INDEX extractions_one_selected ON extractions (version_id) WHERE is_selected;

CREATE TABLE extraction_assets (
    extraction_id bigint NOT NULL,
    asset_id bigint NOT NULL,
    version_id bigint NOT NULL,
    PRIMARY KEY (extraction_id, asset_id),
    FOREIGN KEY (extraction_id, version_id) REFERENCES extractions (id, version_id),
    FOREIGN KEY (asset_id, version_id) REFERENCES document_assets (id, version_id)
);

CREATE TABLE extracted_pages (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    extraction_id bigint NOT NULL,
    asset_id bigint NOT NULL,
    page_index integer NOT NULL CHECK (page_index > 0),
    printed_page_label text,
    source_text text NOT NULL,
    UNIQUE (extraction_id, asset_id, page_index),
    UNIQUE (id, extraction_id),
    UNIQUE (id, extraction_id, asset_id),
    FOREIGN KEY (extraction_id, asset_id) REFERENCES extraction_assets
);

CREATE TABLE sections (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    extraction_id bigint NOT NULL REFERENCES extractions,
    parent_section_id bigint,
    asset_id bigint NOT NULL,
    ordinal integer NOT NULL CHECK (ordinal > 0),
    heading text,
    section_label text,
    start_page_id bigint NOT NULL,
    end_page_id bigint NOT NULL,
    text_content text NOT NULL,
    applicability_note text,
    UNIQUE (extraction_id, ordinal),
    UNIQUE (id, extraction_id),
    UNIQUE (id, extraction_id, asset_id),
    FOREIGN KEY (parent_section_id, extraction_id, asset_id)
        REFERENCES sections (id, extraction_id, asset_id),
    FOREIGN KEY (start_page_id, extraction_id, asset_id)
        REFERENCES extracted_pages (id, extraction_id, asset_id),
    FOREIGN KEY (end_page_id, extraction_id, asset_id)
        REFERENCES extracted_pages (id, extraction_id, asset_id),
    CHECK (parent_section_id <> id)
);

CREATE TABLE extracted_tables (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    extraction_id bigint NOT NULL REFERENCES extractions,
    section_id bigint,
    asset_id bigint NOT NULL,
    ordinal integer NOT NULL CHECK (ordinal > 0),
    source_label text,
    caption text,
    start_page_id bigint NOT NULL,
    end_page_id bigint NOT NULL,
    grid jsonb NOT NULL CHECK (jsonb_typeof(grid) = 'object'),
    footnotes text,
    UNIQUE (extraction_id, ordinal),
    UNIQUE (id, extraction_id),
    FOREIGN KEY (section_id, extraction_id, asset_id) REFERENCES sections (id, extraction_id, asset_id),
    FOREIGN KEY (start_page_id, extraction_id, asset_id) REFERENCES extracted_pages (id, extraction_id, asset_id),
    FOREIGN KEY (end_page_id, extraction_id, asset_id) REFERENCES extracted_pages (id, extraction_id, asset_id)
);

CREATE TABLE chunks (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    extraction_id bigint NOT NULL REFERENCES extractions,
    section_id bigint,
    table_id bigint,
    ordinal integer NOT NULL CHECK (ordinal > 0),
    language_tag language_tag NOT NULL CHECK (language_tag = lower(language_tag)),
    retrieval_text text NOT NULL CHECK (btrim(retrieval_text) <> ''),
    applicability_note text,
    UNIQUE (extraction_id, ordinal),
    UNIQUE (id, extraction_id),
    FOREIGN KEY (section_id, extraction_id) REFERENCES sections (id, extraction_id),
    FOREIGN KEY (table_id, extraction_id) REFERENCES extracted_tables (id, extraction_id)
);

CREATE TABLE chunk_spans (
    chunk_id bigint NOT NULL,
    span_no integer NOT NULL CHECK (span_no > 0),
    extraction_id bigint NOT NULL,
    page_id bigint NOT NULL,
    char_start integer,
    char_end integer,
    source_anchor text CHECK (btrim(source_anchor) <> ''),
    bbox jsonb,
    table_cells jsonb CHECK (jsonb_typeof(table_cells) = 'array'),
    PRIMARY KEY (chunk_id, span_no),
    FOREIGN KEY (chunk_id, extraction_id) REFERENCES chunks (id, extraction_id),
    FOREIGN KEY (page_id, extraction_id) REFERENCES extracted_pages (id, extraction_id),
    CHECK ((char_start IS NULL AND char_end IS NULL) OR
           (char_start IS NOT NULL AND char_end IS NOT NULL AND char_start >= 0 AND char_end > char_start)),
    CHECK (char_start IS NOT NULL OR source_anchor IS NOT NULL OR bbox IS NOT NULL),
    CHECK (CASE WHEN jsonb_typeof(bbox) = 'array' THEN
               jsonb_array_length(bbox) = 4 AND
               jsonb_typeof(bbox -> 0) = 'number' AND jsonb_typeof(bbox -> 1) = 'number' AND
               jsonb_typeof(bbox -> 2) = 'number' AND jsonb_typeof(bbox -> 3) = 'number' AND
               (bbox -> 0)::numeric >= 0 AND (bbox -> 1)::numeric >= 0 AND
               (bbox -> 2)::numeric <= 1 AND (bbox -> 3)::numeric <= 1 AND
               (bbox -> 0)::numeric < (bbox -> 2)::numeric AND
               (bbox -> 1)::numeric < (bbox -> 3)::numeric
           ELSE bbox IS NULL END)
);

CREATE TABLE embedding_models (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    provider text NOT NULL,
    model_name text NOT NULL,
    model_revision text NOT NULL,
    input_profile text NOT NULL,
    dimensions integer NOT NULL CHECK (dimensions BETWEEN 1 AND 16000),
    distance_metric text NOT NULL CHECK (distance_metric IN ('cosine', 'l2', 'inner_product', 'l1')),
    is_external boolean NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (provider, model_name, model_revision, input_profile, dimensions, distance_metric),
    UNIQUE (id, dimensions, distance_metric)
);

CREATE TABLE chunk_embeddings (
    chunk_id bigint NOT NULL REFERENCES chunks,
    model_id bigint NOT NULL,
    dimensions integer NOT NULL,
    distance_metric text NOT NULL,
    embedding vector NOT NULL,
    input_sha256 sha256_digest NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (chunk_id, model_id),
    FOREIGN KEY (model_id, dimensions, distance_metric)
        REFERENCES embedding_models (id, dimensions, distance_metric),
    CHECK (vector_dims(embedding) = dimensions),
    CHECK (distance_metric <> 'cosine' OR vector_norm(embedding) > 0)
);

CREATE TABLE ingestion_attempts (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    work_key text NOT NULL,
    attempt_no integer NOT NULL CHECK (attempt_no > 0),
    stage text NOT NULL CHECK (stage IN ('fetch', 'identity', 'extract', 'validate', 'embed')),
    status text NOT NULL CHECK (status IN ('running', 'succeeded', 'failed', 'needs_review', 'skipped')),
    source_id bigint REFERENCES sources,
    requested_uri text,
    resolved_uri text,
    source_metadata jsonb CHECK (jsonb_typeof(source_metadata) = 'object'),
    observed_sha256 sha256_digest,
    version_id bigint REFERENCES document_versions,
    asset_id bigint REFERENCES document_assets,
    extraction_id bigint REFERENCES extractions,
    model_id bigint REFERENCES embedding_models,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    error_code text,
    error_detail text,
    UNIQUE (work_key, attempt_no),
    FOREIGN KEY (asset_id, version_id) REFERENCES document_assets (id, version_id),
    FOREIGN KEY (extraction_id, version_id) REFERENCES extractions (id, version_id),
    FOREIGN KEY (extraction_id, asset_id) REFERENCES extraction_assets,
    CHECK (version_id IS NOT NULL OR (asset_id IS NULL AND extraction_id IS NULL)),
    CHECK ((status = 'running') = (finished_at IS NULL)),
    CHECK (finished_at >= started_at),
    CHECK (status <> 'failed' OR error_code IS NOT NULL)
);

CREATE TABLE clinical_reviews (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    extraction_id bigint NOT NULL REFERENCES extractions,
    clinical_context_revision bigint NOT NULL CHECK (clinical_context_revision > 0),
    reviewer_ref text NOT NULL CHECK (btrim(reviewer_ref) <> ''),
    decision text NOT NULL CHECK (decision IN ('approved', 'rejected', 'revoked')),
    decided_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    valid_until timestamptz,
    notes text,
    CHECK (valid_until IS NULL OR (decision = 'approved' AND valid_until > decided_at))
);

CREATE INDEX documents_translation_idx ON documents (translation_of_id);
CREATE INDEX document_identifiers_document_idx ON document_identifiers (document_id);
CREATE INDEX document_categories_category_idx ON document_categories (category_code, document_id);
CREATE INDEX document_versions_replaces_idx ON document_versions (replaces_version_id);
CREATE INDEX version_countries_country_idx ON version_countries (country_code, version_id);
CREATE INDEX version_topics_topic_idx ON version_topics (topic_id, version_id);
CREATE INDEX document_assets_sha256_idx ON document_assets (sha256);
CREATE INDEX extraction_assets_asset_idx ON extraction_assets (asset_id, version_id);
CREATE INDEX sections_parent_idx ON sections (parent_section_id);
CREATE INDEX extracted_tables_section_idx ON extracted_tables (section_id);
CREATE INDEX chunks_section_idx ON chunks (section_id);
CREATE INDEX chunks_table_idx ON chunks (table_id);
CREATE INDEX chunk_spans_page_idx ON chunk_spans (page_id);
CREATE INDEX chunk_embeddings_model_idx ON chunk_embeddings (model_id, chunk_id);
CREATE INDEX ingestion_attempts_source_idx ON ingestion_attempts (source_id, started_at DESC);
CREATE INDEX ingestion_attempts_asset_idx ON ingestion_attempts (asset_id, started_at DESC);
CREATE INDEX ingestion_attempts_version_idx ON ingestion_attempts (version_id, started_at DESC);
CREATE INDEX ingestion_attempts_extraction_idx ON ingestion_attempts (extraction_id);
CREATE INDEX ingestion_attempts_unresolved_idx ON ingestion_attempts (stage, started_at)
    WHERE status IN ('failed', 'needs_review');
CREATE INDEX clinical_reviews_latest_idx ON clinical_reviews (extraction_id, decided_at DESC, id DESC);

INSERT INTO schema_migrations (version, name) VALUES (1, 'core');
\endif
COMMIT;
