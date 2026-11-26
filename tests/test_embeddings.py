"""Phase 6 automated tests for local CPU embeddings, caching, and invariants."""

import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ingestion.chunking import read_jsonl
from ingestion.embeddings import (
    DEFAULT_MODEL,
    LocalE5Encoder,
    embed_chunks,
    format_passage_input,
    get_model_spec,
    input_digest,
    resolve_device,
    validate_vector,
)


class _FakeEncoder:
    """Deterministic unit-test encoder matching E5 dimensions and normalization."""

    def __init__(self, spec):
        self.spec = spec
        self.load_seconds = 0.01
        self.calls = []

    def encode_formatted(self, formatted_texts, *, batch_size=16):
        self.calls.append(list(formatted_texts))
        dims = self.spec["dimensions"]
        vectors, lengths = [], []
        for text in formatted_texts:
            tokens = text.split()
            if len(tokens) + 2 > self.spec["max_seq_length"]:
                raise ValueError("Exceeded max_seq_length")
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = [((digest[i % len(digest)] / 255.0) - 0.5) or 0.1 for i in range(dims)]
            norm = math.sqrt(sum(x * x for x in raw))
            vectors.append([x / norm for x in raw])
            lengths.append(len(tokens) + 2)
        return vectors, lengths


class EmbeddingTests(unittest.TestCase):
    def _sample_chunks(self):
        return [
            {
                "chunk_id": "chk-1",
                "ordinal": 1,
                "asset_key": "DOC-1",
                "record_id": "DOC-1",
                "asset_sha256": "a" * 64,
                "language_tag": "fr",
                "section_key": "DOC-1:section:1",
                "physical_pages": [1],
                "retrieval_text": "Metformine 500 mg deux fois par jour.",
                "spans": [{"span_no": 1, "physical_page": 1, "char_start": 0, "char_end": 37, "bbox": [0.1, 0.1, 0.9, 0.2]}],
            },
            {
                "chunk_id": "chk-2",
                "ordinal": 2,
                "asset_key": "DOC-2",
                "record_id": "DOC-2",
                "asset_sha256": "b" * 64,
                "language_tag": "ar",
                "section_key": "DOC-2:section:1",
                "physical_pages": [2],
                "retrieval_text": "يُوصى بجرعة ٥ ملغ يومياً.",
                "spans": [{"span_no": 1, "physical_page": 2, "char_start": 0, "char_end": 25, "bbox": [0.1, 0.2, 0.8, 0.3]}],
            },
        ]

    def test_model_spec_and_vector_validation(self):
        small = get_model_spec("intfloat/multilingual-e5-small")
        base = get_model_spec("intfloat/multilingual-e5-base")
        self.assertEqual(small["dimensions"], 384)
        self.assertEqual(base["dimensions"], 768)
        self.assertFalse(small["is_external"])
        self.assertEqual(small["query_prefix"], "query: ")
        self.assertEqual(small["passage_prefix"], "passage: ")

        with self.assertRaises(ValueError):
            get_model_spec("unknown/model")
        with self.assertRaises(ValueError):
            validate_vector([1.0, 2.0], 3)
        with self.assertRaises(ValueError):
            validate_vector([0.0, 0.0, 0.0], 3, distance_metric="cosine")
        with self.assertRaises(ValueError):
            validate_vector([1.0, float("nan"), 0.0], 3)

    def test_no_silent_truncation_check(self):
        spec = get_model_spec(DEFAULT_MODEL)
        fake = type("TokCheck", (), {})()
        fake.spec = spec
        fake.tokenizer = lambda texts, **_: {"input_ids": [[1] * 513 for _ in texts]}
        with self.assertRaisesRegex(ValueError, "exceeding max_seq_length=512"):
            LocalE5Encoder.check_lengths(fake, ["passage: too long"])

    def test_embed_chunks_deterministic_caching_and_resume(self):
        spec = get_model_spec(DEFAULT_MODEL)
        chunks = self._sample_chunks()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "emb"
            enc1 = _FakeEncoder(spec)
            rows1, sum1 = embed_chunks(chunks, out, model_name=DEFAULT_MODEL, batch_size=1, encoder=enc1)
            self.assertEqual(sum1["encoded_chunks"], 2)
            self.assertEqual(sum1["reused_cached_chunks"], 0)
            self.assertEqual(len(enc1.calls), 2)

            # Second run with same inputs reuses 100% from cache with zero encoder calls.
            enc2 = _FakeEncoder(spec)
            rows2, sum2 = embed_chunks(chunks, out, model_name=DEFAULT_MODEL, batch_size=2, encoder=enc2)
            self.assertEqual(sum2["encoded_chunks"], 0)
            self.assertEqual(sum2["reused_cached_chunks"], 2)
            self.assertEqual(enc2.calls, [])
            self.assertEqual(rows1, rows2)

            # Verify PostgreSQL-compatible model and embedding outputs.
            model_meta = json.loads((out / "embedding_model.json").read_text(encoding="utf-8"))
            self.assertEqual(model_meta["dimensions"], 384)
            self.assertEqual(model_meta["distance_metric"], "cosine")
            self.assertFalse(model_meta["is_external"])
            loaded_rows = read_jsonl(out / "chunk_embeddings.jsonl")
            self.assertEqual([r["chunk_id"] for r in loaded_rows], ["chk-1", "chk-2"])
            self.assertEqual(loaded_rows[0]["spans"], chunks[0]["spans"])
            expected_sha = input_digest(format_passage_input(chunks[0]["retrieval_text"], spec))
            self.assertEqual(loaded_rows[0]["input_sha256"], expected_sha)

            # Modifying one chunk re-encodes only the modified chunk.
            modified = [dict(chunks[0]), {**chunks[1], "retrieval_text": "نص معدل."}]
            enc3 = _FakeEncoder(spec)
            _, sum3 = embed_chunks(modified, out, model_name=DEFAULT_MODEL, batch_size=2, encoder=enc3)
            self.assertEqual(sum3["encoded_chunks"], 1)
            self.assertEqual(sum3["reused_cached_chunks"], 1)

    def test_embed_cli_help(self):
        proc = subprocess.run(
            [sys.executable, "-m", "ingestion", "embed", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--model", proc.stdout)
        self.assertIn("--batch-size", proc.stdout)
        self.assertIn("--device", proc.stdout)
        self.assertIn("--gpu-batch-size", proc.stdout)
        self.assertIn("--no-cpu-fallback", proc.stdout)

    def test_resolve_device_selection(self):
        import torch

        self.assertEqual(resolve_device("cpu"), "cpu")
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), "cpu")
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                resolve_device("cuda")
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            torch.cuda, "device_count", return_value=1
        ):
            self.assertEqual(resolve_device("auto"), "cuda")
            self.assertEqual(resolve_device("cuda"), "cuda")
            self.assertEqual(resolve_device("cuda:0"), "cuda:0")
            with self.assertRaises(ValueError):
                resolve_device("cuda:1")
        with self.assertRaises(ValueError):
            resolve_device("mps")

    def test_cuda_oom_backoff_and_cpu_fallback(self):
        import torch

        spec = get_model_spec(DEFAULT_MODEL)
        enc = object.__new__(LocalE5Encoder)
        enc.spec = spec
        enc._torch = torch
        enc.device = "cuda"
        enc.requested_device = "cuda"
        enc.fallback_to_cpu_on_oom = True
        enc.oom_backoff_events = 0
        enc.cpu_fallback_used = False
        enc.min_batch_size_used = None
        enc.peak_gpu_memory_mib = 0.0
        enc.load_seconds = 0.01
        enc.check_lengths = lambda texts: [8] * len(texts)

        moved_to = []
        enc.model = type("FakeModel", (), {"to": lambda _self, d: moved_to.append(d)})()

        seen_calls = []

        def fake_encode_batch(batch):
            seen_calls.append((enc.device, len(batch)))
            if enc.device == "cuda":
                raise RuntimeError("CUDA out of memory. Tried to allocate 256 MiB")
            return [[1.0 / math.sqrt(384)] * 384 for _ in batch]

        enc._encode_batch = fake_encode_batch

        with patch.object(torch.cuda, "is_available", return_value=False):
            vectors, lengths = enc.encode_formatted(["passage: a", "passage: b", "passage: c", "passage: d"], batch_size=4)
        self.assertEqual(len(vectors), 4)
        self.assertEqual(lengths, [8, 8, 8, 8])
        # Should halve batch 4 -> 2 -> 1 on CUDA, then fall back to CPU.
        self.assertEqual(seen_calls, [("cuda", 4), ("cuda", 2), ("cuda", 1), ("cpu", 1), ("cpu", 1), ("cpu", 1), ("cpu", 1)])
        self.assertTrue(enc.cpu_fallback_used)
        self.assertEqual(enc.device, "cpu")
        self.assertEqual(moved_to, ["cpu"])
        self.assertEqual(enc.oom_backoff_events, 3)

        # Resume after fallback keeps working with embed_chunks summary fields.
        with tempfile.TemporaryDirectory() as tmp:
            _, summary = embed_chunks(
                self._sample_chunks(),
                Path(tmp) / "emb",
                batch_size=4,
                encoder=enc,
            )
            self.assertEqual(summary["device"], "cpu")
            self.assertTrue(summary["cpu_fallback_used"])
            self.assertEqual(summary["oom_backoff_events"], 3)


if __name__ == "__main__":
    unittest.main()
