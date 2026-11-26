# Unified metadata v1

Use one UTF-8 JSON object per candidate publication revision in `manifest.jsonl`. Use forward-slash paths relative to the collection root. Supply one primary asset and attach verified same-edition annexes or alternate renditions as needed. Category adapters can emit the same record from existing manifests. Validate each object against [metadata.schema.json](metadata.schema.json), JSON Schema Draft 2020-12, with format checking enabled.

## Fields teammates prepare

| Group | Required for intake | Optional with evidence |
| --- | --- | --- |
| Manifest | `schema_version: 1`, unique local `record_id`, one or more seeded `categories` | Evidence locators and an adapter context object |
| Publication | Published `title`, primary `language`, `issuer.name` and `issuer.kind`, `document_type`, `authority` | Canonical title, issuer key and home jurisdiction, document identity and revision keys, scoped identifiers, edition, date of publication, effective dates, publisher status and relations |
| Access and scope | `assets` with one primary `path` and `role` | Official and acquisition URLs, evidence and acquisition date, geographic scope, countries, population and rules, topics, additional content languages, reuse at version and asset level, intake review state |

Minimal synthetic example, formatted as a single JSONL line.

```json
{"schema_version":1,"record_id":"example-001","categories":["treatment_protocols"],"title":"Exemple de protocole","language":"fr","issuer":{"name":"Example Issuing Body","kind":"unknown"},"document_type":"clinical_protocol","authority":"unknown","assets":[{"path":"pdfs/example.pdf","role":"primary"}]}
```

Omit absent optional fields. Use `unknown` for an unresolved classification, scope or permission, and `und` for an unknown primary language. Intake with no supportable title or issuer remains an unresolved discovery/attempt record until identified. Paths, placeholders, guesses and numeric file statistics cannot establish a publisher. Prefer the publisher imprint or authenticated register, followed by documented acquisition evidence, sidecars and filenames. Preserve conflicts with evidence locations. Use `YYYY`, `YYYY-MM` or `YYYY-MM-DD` as recorded. Only a full verified day maps to the SQL date. Effective dates require day precision, and the end is exclusive. An absent date or rights statement stays unresolved.

## Identity and categories

- A publication identity is keyed by verified issuer, stable work key and canonical lowercase BCP 47 primary language. Assign deterministic internal keys only after establishing evidence. Use `record_id` as a collection-local pointer. Carry issuer identifiers such as Official Journal issue and act numbers, an official reference or a verified work code as scoped identifiers. Record DOI, ISBN and document-number edition or rendition scope before using them for reconciliation.
- A revision represents an issued edition of that identity. Derive a stable revision key from verified publisher evidence, or allocate a reviewed key where the publisher supplies no revision token. A verified reissue can point to a revision of the same document with `replaces`. Different bytes alone may reflect a cover sheet, scan, mirror, or alternate. Different translations and jurisdictional adaptations get distinct identities. An Arabic/French publication issued as one unit can use `language: "mul"` and `content_languages: ["ar","fr"]`. Additional script in a logo or masthead can be recorded as evidence. Apply per-section and per-chunk language downstream.
- A relationship points to an in-manifest `record_id` or a qualified `collection:record_id` reference in the local reconciliation catalogue. `candidate` records hints. `verified` requires evidence and a resolvable target. For translation, resolve the target document. For replacement, resolve the target revision and check that both revisions belong to the same document. Check for self-links, conflicting or cyclic relationships, and ambiguous identifier assignments in semantic validation.
- Category membership is independent of identity, edition, issuer and authority. Add a reviewed category association to the resolved document when the same publication belongs in another category. Keep every distinct acquisition event. Compare archive filenames, publisher IDs, hashes and visible pages before classifying near duplicates, translations, summaries, corrections, annexes or competing guidelines.

## Authority, applicability and evidence

`issuer.kind` describes the organization. `document_type` describes the publication. `authority` describes the project's retrieval context. Use `official_national` for authenticated Algerian national material, `international_reference` for eligible foreign or international references, `contextual` for supporting regulatory, strategy or implementation material as appropriate, and `educational` for teaching content. Official national material can be legal or administrative. Apply document type and scope separately to determine where it can be cited. Drug nomenclature guidance belongs in the contextual lane when it supplies background. Classification alone confers no retrieval permission or approval.

Use registered BCP 47 tags and ISO 3166-1 alpha-2 countries. Store issuer country separately from clinical applicability. Use `global`, a verified `country_list` or `unknown` and keep original scope text, including regional limits the DB cannot express directly. Represent population as `general`, `restricted` or `unknown`, with source wording and structured inclusion/exclusion rules when supported. Conditions within a rule are conjunctive, an applicable exclusion takes precedence, and unknown scope supplies no positive applicability match. Record age in whole days with inclusive lower and exclusive upper bounds. Keep topic vocabulary, code, label and any scope evidence together.

A provenance review records who issued the specific revision, publisher landing URL when verified, actual acquisition URL when documented, original file identity and supporting locators. Acquisitions record requested and resolved URLs separately. Record a known mirror as a host. A local file with no known download URL keeps its real local path, method, hash and missing route, with no invented URL. Cite each extracted unit back to the original asset and physical page, plus printed page, heading/article, figure or table labels where available.

## Generated fields and states

- Discovery measures SHA-256, byte length, media type and page count. It derives an immutable storage URI. Treat supplied observations as expectations and recompute them before storage. The runner generates work keys, attempt numbers, fingerprints, timestamps, DB IDs and extraction/citation coordinates. Source PDFs stay read-only.
- Apply reuse decisions separately to the version and to each asset. Store evidence for the intended retrieval and external processing uses. Status is `unknown`, `allowed`, `restricted` or `prohibited`. A conditional licence remains unresolved until the project verifies scope, attribution, third-party terms and handling of the exact asset. Record conditions and evidence regardless of the decision.
- `publisher_status` and `intake_review` are intake evidence and workflow annotations. New database versions start staged. Clinical decisions are recorded separately by an authorized reviewer after extraction completion and are tied to the applicable clinical context revision. Intake manifests cannot grant clinical approval or select a current extraction. See [database mapping](METADATA_MAPPING.md).

## Phase 2 validation contract

The [JSON Schema](metadata.schema.json) is the canonical contract. The CLI reports schema errors, invalid registry codes, impossible dates, reversed effective windows or age bounds, duplicate record IDs and asset paths, invalid DOI syntax and ISBN checksums, unsupported or unavailable assets, and mismatched file observations. Supply any known SHA-256, byte count, media type or physical page count under an asset's `observed` object. Intake remeasures every listed PDF. An encrypted PDF is held even when its header and page count can be read. A structural pass checks the file and page tree and can leave deeper page-content defects undetected.

A caller can assert a verified identifier with `verified: true` and a source locator in `evidence`. Scope must be explicit. Use an issuer key or explicit identifier namespace for a collection-independent work key. If using a reviewed `identity_key` or `revision_key`, supply `identity_evidence` or `revision_evidence`. The resolver derives keys from those claims and leaves unresolved keys empty. Verify issuer aliases and identifiers against publication evidence before recording them. Reconciliation emits candidate matches, including verified relationships and unresolved targets, and preserves each local record. Matching hashes establish byte identity. Different publisher claims, language or edition still require review.

Use `attachment_type` on annexes and alternate renditions to distinguish corrections, supplements, annexes and alternate representations. Verified relations require a supporting note and a resolvable target. For cross-collection targets, pass a reference inventory to reconciliation and use `collection_code:record_id`. Translation edges require different language identities. Replacement edges require a shared evidenced work. Chains are checked for cycles. Use candidate relationships while any link is unresolved.

An optional `academic_review` records approval for academic project work with evidence. It has no effect on permission fields or intake classification. An `allowed` permission claim requires reuse evidence and remains a supplied claim for review. Authority and classification stay independent of retrieval eligibility. Phase 2 emits `clinical_review: "not_assessed"` and `treatment_guidance_eligible: false` for inspected records, including official clinical candidates.

### Generic CSV

Start with the [header template](metadata_template.csv). Fill `record_id`, `path`, `title`, and `issuer`. Leave unknown optional fields blank. The adapter supplies `und` for an unknown language, `unknown` for an absent classification, and the CLI collection code when the categories cell is blank. Separate category and content language lists with `|`. Each row has one primary PDF and can record one unverified DOI claim. Use canonical JSONL for multiple assets, structured applicability, rights decisions and sourced identity or version relationships. The adapter preserves the entire original CSV row.
