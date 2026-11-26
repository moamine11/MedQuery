# MedQuery extraction decision

**Proposed for review after Phase 3.** Use PyMuPDF for native page text and source geometry, Tesseract for selective OCR, and a separate layout review path for complex tables, journals and diagrams. Preserve the competing candidates when they disagree on a clinically sensitive field. The [experiment report](EXTRACTION_EXPERIMENTS.md) records the measured evidence and the [output specification](EXTRACTION_OUTPUT_SPECIFICATION.md) defines the handoff.

## Engine selection

| Component | Proposed use |
| --- | --- |
| PyMuPDF | Open the verified asset, extract native glyphs and word and block geometry, render the displayed CropBox, inspect scan overlays and expose ruled-table candidates. Retain native order as the starting point. |
| Poppler | Independent native comparison if PyMuPDF fails or its text or ordering needs corroboration. Use default text for its reading-order candidate and keep the layout and bounding-box streams as diagnostic evidence. Flag repairs to the geometry stream. |
| PyMuPDF table finder | Try it first on clearly ruled, native tables. Verify header, row and value associations against the original geometry before accepting structured cells. |
| Docling Heron plus TableFormer Fast | Use CPU-native layout and table candidates for borderless or complex layouts, and for locating tables and figures when the lighter attempt is insufficient. Keep its model and parser roles visible in audit output. |
| Tesseract 5.3.4 | Run locally with `fra+eng`, `ara+eng` or `eng` for the page or region. Preserve language choice, OCR confidence and geometry. Review mixed Arabic/French areas at the block level. |
| PaddleOCR mobile Arabic | Consider as a targeted second candidate for difficult Arabic lines. The tested word-box option failed and the working adapter supplies line polygons, which require source verification for word-level citations. |

<!-- DECISION_METRICS_START -->

PyMuPDF's annotated CER was 9.0%, with 98/104 box hits and a 0.15 s cold page median. Selective OCR improved CER to 3.3% and exact critical items to 50/57. It routed 4 of 40 pages to OCR and took 13.1 s across the pilot. Full-page OCR on all pages took 63.6 s.

Docling located 11/12 annotated table regions and recovered 10/17 tested row/header relationships. PyMuPDF located 6/12 regions and better preserved verified merged cells. Keep both as candidate tools. Docling's cold page median was 6.97 s, its peak RSS was 1170 MiB, and neither engine established marker-level footnote or diagram links.

On the same five Arabic pages, Tesseract had 15.1% CER and PaddleOCR had 7.9%. PaddleOCR had a 14.61 s median and 1,092 MiB peak RSS. Tesseract had a 3.27 s median and 107 MiB peak RSS. The five-page comparison includes two controls, leaving three content pages, and the adapters use different geometry granularities. Use Tesseract for routine selective OCR and review difficult Arabic with both candidates when justified.

<!-- DECISION_METRICS_END -->

## Routing and fallbacks

1. **Verify the original asset first.** Match the Phase 2 identity and SHA-256, respect holds, and keep all required supplements and corrections linked to the correct version. Retain the physical page independently of any inferred or printed label.
2. **Start with native evidence.** Preserve native text, text trace, image coverage, block coordinates and direction hints. Treat missing letters, Unicode replacement/private-use/control codes, implausible script mix, overlapping or hidden text, and parser disagreements as signals. A significant raster with a dominant hidden layer warrants independent pixel OCR even when the PDF contains substantial text.
3. **Triage blank and decorative pages.** Use text, ink and rendered edge signals together. The exploratory threshold skipped all five pilot controls. On weak-text pages with low detail, record blank or decoration only when evidence supports it. Send uncertain faint or cropped content to review with the source image. Classify branding and decoration independently of parser output.
4. **Try crop OCR where the figure or label region has reliable bounds.** Reuse native text around the crop, anchor OCR to the original physical page, and preserve overlapping candidates separately. A raster rectangle alone did not find a usable automatic crop in this pilot. Use a located layout candidate or reviewed figure bounds and check coverage outside the crop. Use full-page OCR for raster-dominant pages and for weak text with clinically relevant visual detail and uncertain regions.
5. **Route reading-order problems to layout first.** Check columns, full-width headings, captions, rotated table content and block boundaries. Compare native order with geometry and a second parser on uncertain pages. For Arabic prose, segment columns right-to-left, read paragraphs top-to-bottom within a column, and retain Unicode logical order within each line. Treat embedded Latin abbreviations, decimal values and numbers according to their own direction. Whole-page positional sorting and blind string reversal can corrupt both. Render verification or a second OCR candidate is warranted when the logical string or a critical token remains uncertain.
6. **Preserve unresolved structure.** Link table cells to the correct row, column, header, spanning cell, note marker and physical fragment. Retain the visible caption and boxes while a link is unresolved. For diagrams, store checked labels alongside their boxes and preserve the original diagram region. Any traversal requires visible arrows, endpoint identities and branch conditions. Send ambiguous alternatives, continuations and links to review.
7. **On tool failure, retain diagnostics and retry only a compatible local candidate.** Record the failing adapter and bounded attempt. A blank result over visible source content, inconsistent page geometry, or unresolvable reading order leaves that material in review. Keep reviewed source text even when an unrelated region is unresolved, with review scoped to the smallest owning component.

## Quality acceptance

Profile text accuracy and aligned completeness by language and layout. Use these rates for parser comparisons. Accept a clinically sensitive excerpt or structure only with source-local verification of all critical quantities, units and exponents, timing, negations, comparisons, conditions, row/header/footnote links and edition/page identity. Missing text, plausible substitutions and joins across columns can change applicability even when average CER is low. Low OCR confidence is a review signal. High confidence or agreement between engines alone cannot certify it.

Use the displayed-page normalized `[x0,y0,x1,y1]` representation and, when available, exact code-point offsets. Give every unresolved item the original asset hash, 1-based physical page, optional printed label and the smallest reliable region. Guard any review or future publication with the existing authorization and clinical-context policy. [Output coordinates and SQL mapping](EXTRACTION_OUTPUT_SPECIFICATION.md)

Keep the routing thresholds and structure fallback provisional until the team reviews Phase 3, with coverage gaps for separate Arabic and French image-only specimens, true cross-page joins, ambiguous multi-branch algorithms and rotated or clipped pages. Any implementation begins after approval.
