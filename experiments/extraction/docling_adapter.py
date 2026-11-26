"""Offline structured extraction with Docling 2.63, Heron and TableFormer Fast."""

from engines import normalize, result, word


def extract(path, info, model_path, config):
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import DocItemLabel, TableItem

    required = ("docling-project--docling-layout-heron/model.safetensors",
                "docling-project--docling-models/model_artifacts/tableformer/fast/tableformer_fast.safetensors")
    if any(not (model_path / item).is_file() for item in required):
        raise FileNotFoundError("Expected the offline Heron and TableFormer Fast weights")

    options = PdfPipelineOptions(
        artifacts_path=model_path, do_ocr=False, do_table_structure=True, generate_parsed_pages=True,
        enable_remote_services=False, accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU, num_threads=config["ocr_workers"],
        ),
    )
    options.table_structure_options.mode = TableFormerMode.FAST
    page_number = info["physical_page"]
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    converted = converter.convert(path, page_range=(page_number, page_number), raises_on_error=False)
    if converted.status not in {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}:
        raise RuntimeError(f"Docling conversion returned {converted.status.value}: {converted.errors}")
    [page] = converted.pages
    if page.page_no != page_number - 1:
        raise ValueError("Docling returned an unexpected physical page")
    width, height = page.size.width, page.size.height

    def box(value):
        bbox = value.to_top_left_origin(height)
        return normalize((bbox.l, bbox.t, bbox.r, bbox.b), width, height)

    parsed = page.parsed_page
    lines = [[word(cell.text, box(cell.to_bounding_box()))] for cell in parsed.word_cells if cell.text.strip()]
    headings, tables, figures = [], [], []
    for item, _ in converted.document.iterate_items():
        anchors = [p for p in getattr(item, "prov", []) if p.page_no == page_number]
        if not anchors:
            continue
        bbox = box(anchors[0].bbox)
        if item.label in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER):
            headings.append({"text": item.text, "bbox": bbox})
        if item.label == DocItemLabel.PICTURE:
            figures.append({"bbox": bbox})
        if isinstance(item, TableItem):
            cells = [{
                "row": cell.start_row_offset_idx, "col": cell.start_col_offset_idx,
                "text": cell.text, "bbox": box(cell.bbox) if cell.bbox else None,
                "row_span": cell.row_span, "col_span": cell.col_span,
                "column_header": cell.column_header, "row_header": cell.row_header,
            } for cell in item.data.table_cells]
            footnotes = []
            for ref in item.footnotes:
                note = ref.resolve(converted.document)
                for note_anchor in note.prov:
                    if note_anchor.page_no == page_number:
                        footnotes.append({"text": note.text, "bbox": box(note_anchor.bbox)})
            tables.append({
                "bbox": bbox, "row_count": item.data.num_rows, "col_count": item.data.num_cols, "cells": cells,
                "linked_footnotes": footnotes, "footnote_links": [], "continuation_links": [],
            })
    return result(lines, text=converted.document.export_to_text(), headings=headings, tables=tables, diagnostics={
        "status": converted.status.value, "page_number": page.page_no + 1, "figures": figures,
        "word_geometry_stream": "docling-parse word cells", "structured_errors": [str(e) for e in converted.errors],
    })
