"""Phase 4 regression tests with neutral local fixtures."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from experiments.extraction.engines import pdf_box, result, word
from ingestion.extraction.assemble import finalize_asset, margin_signature
from ingestion.extraction.config import file_hash, fingerprint, load_profile
from ingestion.extraction.layout import order_lines, valid_box
from ingestion.extraction.normalize import assemble_normalized, normalize
from ingestion.extraction.page import process_page
from ingestion.extraction.runner import checkpoint_read, checkpoint_write, process_job, run, snapshot
from ingestion.extraction.selection import Asset, page_numbers, select
from ingestion.extraction.structure import build_tables
from ingestion.intake import IntakeError


class NormalizationTests(unittest.TestCase):
    def test_safe_normalization_and_trace(self):
        source = "  Cafe\u0301   5,25\u00a0mg  ≥ 2×/j,\n ne pas   kg/m²\n\n لا ١٫٥ mmol/L  "
        value, spans = normalize(source, 7)
        self.assertEqual(value, "Café 5,25 mg ≥ 2×/j, ne pas kg/m²\n\nلا ١٫٥ mmol/L")
        coverage = []
        for span in spans:
            fragment = value[span["normalized_start"]:span["normalized_end"]]
            observed = source[span["source_start"]-7:span["source_end"]-7]
            if span["mode"] == "identity":
                self.assertEqual(fragment, observed)
            coverage.extend(range(span["normalized_start"], span["normalized_end"]))
        self.assertEqual(coverage, list(range(len(value))))

    def test_discretionary_wrap_only(self):
        self.assertEqual(normalize("exam\u00ad\nple  x-\ny  a-\n2")[0], "example x- y a- 2")
        self.assertEqual(normalize("Line A\nLine B", repair_wrap=False)[0], "Line A\nLine B")

    def test_block_mapping_and_review_barriers(self):
        page = {"source_text": "Keep\n\n42 mg\n\nKeep too", "blocks": [
            {"block_key": "a", "status": "extracted", "kind": "paragraph", "char_start": 0, "char_end": 4},
            {"block_key": "b", "status": "needs_review", "kind": "table_text", "char_start": 6, "char_end": 11},
            {"block_key": "c", "status": "extracted", "kind": "paragraph", "char_start": 13, "char_end": 21},
        ]}
        assemble_normalized(page)
        self.assertEqual(page["normalized_text"], "Keep\n\nKeep too")
        self.assertIsNone(page["blocks"][1]["normalized_range"])
        self.assertEqual([b["continuity_group"] for b in page["blocks"]], [1, None, 2])

    def test_repeat_margin_detection_preserves_source(self):
        pages = []
        for n in range(1, 4):
            pages.append({"physical_page": n, "asset_key": "A", "status": "extracted",
                          "source_text": "Publisher\n\nParagraph", "blocks": [
                              {"text": "Publisher", "bbox": [.1, .03, .3, .05], "kind": "paragraph", "status": "extracted",
                               "block_key": str(n), "char_start": 0, "char_end": 9},
                              {"text": "Paragraph", "bbox": [.1, .2, .3, .3], "kind": "paragraph", "status": "extracted",
                               "block_key": f"p{n}", "char_start": 11, "char_end": 20},
                          ]})
        finalize_asset(pages)
        self.assertTrue(all(p["normalized_text"] == "Paragraph" and p["source_text"].startswith("Publisher") for p in pages))


    def test_repeated_margin_candidates_exclude_negations_and_recommendations(self):
        block = {"text": "Do not repeat", "bbox": [.1, .03, .3, .05], "kind": "paragraph"}
        page = {"status": "extracted"}
        self.assertIsNone(margin_signature(block, page))
        self.assertIsNone(margin_signature({**block, "text": "توصية مشروطة"}, page))



class GeometryAndRoutingTests(unittest.TestCase):
    def test_cropbox_rotated_transform(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            page.set_cropbox(pymupdf.Rect(50, 70, 550, 770))
            page.set_rotation(90)
            box = pdf_box(page, (50, 70, 150, 170))
        self.assertTrue(valid_box(box))
        for actual, expected in zip(box, (530/700, .1, .9, .3)):
            self.assertAlmostEqual(actual, expected)

    def test_rtl_columns_preserve_latin_abbreviation_and_value(self):
        source = [{"text": "يسار", "bbox": [.1, .2, .4, .23], "style": None},
                  {"text": "يمين ABC 2.5 mg", "bbox": [.56, .24, .9, .28], "style": None}]
        ordered, ambiguous = order_lines(source, [.5], rtl=True)
        self.assertFalse(ambiguous)
        self.assertEqual([x["text"] for x in ordered], ["يمين ABC 2.5 mg", "يسار"])

    def test_missing_gutter_preserves_parser_sequence(self):
        source = [{"text": "left column ends", "bbox": [.10, .70, .48, .73]},
                  {"text": "right column begins", "bbox": [.51, .10, .9, .13]}]
        ordered, ambiguous = order_lines(source, [], rtl=False)
        self.assertFalse(ambiguous)
        self.assertEqual(ordered, source)

    def test_native_failure_uses_poppler_before_ocr(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((70, 100), "A visually present simple paragraph for the local extraction test.")
            box = pdf_box(page, page.get_text("words")[0][:4])
            engine = type("FakeEngines", (), {})()
            engine.page, engine.profile, engine.tools = page, load_profile(), {"poppler": "fake"}
            called = []
            def adapter(name, language, region):
                called.append(name)
                if name == "native":
                    raise OSError("simulated parse error")
                return result([[word("Fallback text", box)]])
            engine.extract = adapter
            data = process_page(engine, {"asset_key": "X", "asset_sha256": "a"*64, "physical_page": 1, "language": "en"})
        self.assertEqual(data["selected_candidate"], "poppler")
        self.assertEqual(called, ["native", "poppler"])
        self.assertEqual(data["source_text"], "Fallback text")

    def test_failed_ocr_holds_suspect_overlay(self):
        from PIL import Image
        from io import BytesIO
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            img = BytesIO()
            Image.new("RGB", (800, 1000), "white").save(img, format="PNG")
            page.insert_image(page.rect, stream=img.getvalue())
            line = "This extraction test contains an incorrect invisible text layer."
            page.insert_text((70, 100), line, render_mode=3)
            from ingestion.extraction.engines import PageEngines
            engine = PageEngines(Path("unused"), 1, page, load_profile(), {"tesseract": None, "poppler": None})
            data = process_page(engine, {"asset_key": "X", "asset_sha256": "a"*64, "physical_page": 1, "language": "en"})
        self.assertEqual(data["routing"]["route"], "suspect_overlay")
        self.assertIn("expected_pixel_text_unavailable", {x["code"] for x in data["diagnostics"]})
        self.assertTrue(all(b["status"] == "needs_review" for b in data["blocks"]))

    def test_multilingual_image_only_routes_to_mixed_ocr(self):
        from PIL import Image, ImageDraw
        from io import BytesIO
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            img = BytesIO()
            canvas = Image.new("RGB", (800, 1000), "white")
            draw = ImageDraw.Draw(canvas)
            for y in range(80, 920, 40):
                draw.rectangle([80, y, 720, y + 10], fill="black")
            canvas.save(img, format="PNG")
            page.insert_image(page.rect, stream=img.getvalue())
            engine = type("FakeEngines", (), {})()
            engine.page, engine.profile, engine.tools = page, load_profile(), {"tesseract": "fake", "poppler": None}
            seen = []
            def adapter(name, language, region):
                if name == "native":
                    return result([])
                seen.append((name, language))
                return result([[word("النص العربي", [.1, .1, .5, .15])]], diagnostics={"ocr_mean_word_confidence": 85})
            engine.extract = adapter
            data = process_page(engine, {"asset_key": "X", "asset_sha256": "a"*64, "physical_page": 1, "language": "mul"})
        self.assertEqual(seen, [("tesseract", "mixed")])
        self.assertEqual(data["selected_candidate"], "tesseract")

    def test_grids_preserve_candidates_and_forbid_stitching(self):
        box = [.1, .2, .8, .7]
        features = {"table_captions": [], "panels": [], "grid": False}
        table = build_tables({"docling": {"tables": [{"bbox": box, "cells": [
            {"row": 0, "col": 0, "row_span": 2, "text": "Q*", "bbox": box}
        ]}]}}, features, [], "asset:3")[0]
        self.assertEqual(table["grid"]["candidates"][0]["cells"][0]["row_span"], 2)
        self.assertEqual(table["status"], "needs_review")
        self.assertEqual(table["continuation"]["linked_fragments"], [])


    def test_continued_caption_and_terminal_marker_hold_full_visible_extent(self):
        lines = [
            {"text": "Table 1 Continued.", "bbox": [.10, .12, .3, .14]},
            {"text": "Well-spaced source text in an unresolved table.", "bbox": [.1, .4, .85, .42]},
            {"text": "(Continued)", "bbox": [.7, .84, .9, .86]},
            {"text": "Unrelated footer", "bbox": [.1, .96, .5, .97]},
        ]
        features = {"table_captions": [lines[0]["bbox"]], "panels": [], "grid": False}
        tools = {"docling": {"tables": [{"bbox": [.1, .94, .22, .98], "cells": []}]}}
        tables = build_tables(tools, features, lines, "asset:2")
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["bbox"], [.1, .12, .9, .86])
        self.assertEqual(tables[0]["status"], "needs_review")
        self.assertEqual(tables[0]["continuation"]["linked_fragments"], [])



class CheckpointTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.directory = Path(work.name)

    def test_corrupt_checkpoint_recovers_and_reuses(self):
        checkpoint = self.directory / "page.json"
        checkpoint_write(checkpoint, "old", {"text": "old"})
        self.assertIsNone(checkpoint_read(checkpoint, "new"))
        checkpoint.write_text('{"bad":true}')
        with patch("ingestion.extraction.runner.one_attempt", return_value=(
            {"text": "new"}, {"outcome": "ok"}
        )) as worker:
            req = {"profile": {"retries": 0}, "context": {}}
            process_job(req, "new", checkpoint, self.directory, True)
            page, events = process_job(req, "new", checkpoint, self.directory, True)
        self.assertEqual(worker.call_count, 1)
        self.assertEqual(page["text"], "new")
        self.assertEqual(events["outcome"], "reused")

    def test_retries_failure_and_recovery(self):
        checkpoint = self.directory / "page.json"
        failed = (None, {"outcome": "timeout", "error": "page_deadline_exceeded"})
        req = {"profile": {"retries": 1}, "context": {}}
        with patch("ingestion.extraction.runner.one_attempt", side_effect=(failed, failed)) as worker:
            page, output = process_job(req, "key", checkpoint, self.directory, True)
        self.assertEqual(worker.call_count, 2)
        self.assertEqual(page["status"], "failed")
        self.assertEqual(len(output["attempts"]), 2)
        self.assertFalse(checkpoint.exists())

    def test_batch_isolates_bad_asset_and_resumes_good(self):
        root, out = self.directory / "source", self.directory / "run"
        root.mkdir()
        path = root / "ok.pdf"
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 120), "This page contains enough plain synthetic text for native extraction.")
            pdf.save(path)
        good = Asset("ok", "ok", "ok.pdf", file_hash(path), path.stat().st_size, 1, "en", "primary",
                     None, None, None, "unknown", "unknown", (1,))
        bad = replace(good, key="bad", record_id="bad", path="missing.pdf")
        profile = {**load_profile(workers=1), "docling_enabled": False, "paddle_enabled": False}
        tools = {"tesseract": None, "poppler": None}
        metrics = run(root, [bad, good], [], out, profile, tools, {}, resume=False, progress=lambda _: None)
        pages = (out / "pages.jsonl").read_bytes()
        again = run(root, [bad, good], [], out, profile, tools, {}, resume=True, progress=lambda _: None)
        self.assertEqual(metrics["page_statuses"], {"failed": 1, "extracted": 1})
        self.assertEqual(again["reused_pages"], 1)
        self.assertEqual(pages, (out / "pages.jsonl").read_bytes())
        self.assertEqual(file_hash(path), good.sha256)

    def test_hash_change_refuses_local_source(self):
        (self.directory / "changed.pdf").write_text("changed")
        asset = Asset("A", "A", "changed.pdf", "0"*64, 7, 1, "en", "primary", None, None, None,
                      "unknown", "unknown", (1,))
        with self.assertRaisesRegex(IntakeError, "differ"):
            with snapshot(self.directory, asset, self.directory, 100):
                self.fail("Unverified source yielded a snapshot")

    def test_selection_enforces_holds_and_permissions(self):
        from test_ingestion import record
        from ingestion.metadata import validate
        inventory = self.directory / "inventory.jsonl"
        inputs = [record("hold", intake_review="hold"), record("blocked", reuse={"use_status": "prohibited"}),
                  record("ok")]
        rows = [{"collection": "treatment_protocols", "validation_state": "valid", "metadata": validate(r)[0],
                 "observations": {r["assets"][0]["path"]: {"bytes": 10, "pages": 3, "sha256": "a"*64}}}
                for r in inputs]
        inventory.write_text("".join(json.dumps(row) + "\n" for row in rows))
        assets, excluded = select(inventory, None, ["hold", "blocked", "ok"], ["1-2"])
        self.assertEqual([a.selected_pages for a in assets], [(1,2)])
        self.assertEqual({x["asset_key"] for x in excluded}, {"hold", "blocked"})
        with self.assertRaises(ValueError):
            page_numbers("2-4", 3)


if __name__ == "__main__":
    unittest.main()
