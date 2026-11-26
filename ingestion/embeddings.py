"""Local CPU/CUDA embedding generation with pinned revisions, caching, OOM recovery, and PostgreSQL compatibility."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import resource
import tempfile
from time import perf_counter

from ingestion.chunking import read_jsonl
from ingestion.extraction.config import fingerprint

DEFAULT_MODEL = "intfloat/multilingual-e5-small"

SUPPORTED_MODELS = {
    "intfloat/multilingual-e5-small": {
        "provider": "huggingface",
        "model_name": "intfloat/multilingual-e5-small",
        "model_revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "input_profile": "e5_passage_v1",
        "dimensions": 384,
        "distance_metric": "cosine",
        "is_external": False,
        "max_seq_length": 512,
        "pooling": "mean",
        "normalize_embeddings": True,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "intfloat/multilingual-e5-base": {
        "provider": "huggingface",
        "model_name": "intfloat/multilingual-e5-base",
        "model_revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
        "input_profile": "e5_passage_v1",
        "dimensions": 768,
        "distance_metric": "cosine",
        "is_external": False,
        "max_seq_length": 512,
        "pooling": "mean",
        "normalize_embeddings": True,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}


def get_model_spec(model_name=DEFAULT_MODEL, *, revision=None):
    """Return pinned model metadata and config fingerprint."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {model_name!r}; supported: {sorted(SUPPORTED_MODELS)}")
    spec = dict(SUPPORTED_MODELS[model_name])
    if revision:
        spec["model_revision"] = revision
    if spec["is_external"]:
        raise ValueError("External embedding models are disabled; local CPU inference only.")
    spec["model_config_fingerprint"] = fingerprint({
        k: spec[k]
        for k in (
            "provider",
            "model_name",
            "model_revision",
            "input_profile",
            "dimensions",
            "distance_metric",
            "is_external",
            "max_seq_length",
            "pooling",
            "normalize_embeddings",
            "query_prefix",
            "passage_prefix",
        )
    })
    return spec


def format_passage_input(text, spec):
    return f"{spec['passage_prefix']}{text}"


def format_query_input(text, spec):
    return f"{spec['query_prefix']}{text}"


def input_digest(formatted_text):
    return sha256(formatted_text.encode("utf-8")).hexdigest()


def validate_vector(vector, expected_dims, *, distance_metric="cosine"):
    """Verify exact dimensions, finite floats, and positive norm."""
    if not isinstance(vector, list) or len(vector) != expected_dims:
        raise ValueError(f"Expected {expected_dims} dimensions, got {len(vector) if isinstance(vector, list) else type(vector)}")
    sq_sum = 0.0
    for val in vector:
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError("Embedding vector contains non-numeric or non-finite values")
        sq_sum += float(val) * float(val)
    norm = math.sqrt(sq_sum)
    if distance_metric == "cosine" and norm <= 0.0:
        raise ValueError("Cosine embedding vector must have strictly positive norm")
    return norm


def resolve_device(device="auto"):
    """Resolve 'auto', 'cpu', or 'cuda[:index]' into an explicit PyTorch device string."""
    normalized = "auto" if device is None else str(device).strip().lower()
    if normalized == "cpu":
        return "cpu"
    import torch

    cuda_ok = bool(torch.cuda.is_available())
    if normalized == "auto":
        return "cuda" if cuda_ok else "cpu"
    if re.fullmatch(r"cuda(?::\d+)?", normalized):
        if not cuda_ok:
            raise RuntimeError(f"Requested device {device!r}, but CUDA is unavailable")
        if ":" in normalized:
            index = int(normalized.split(":", 1)[1])
            count = torch.cuda.device_count()
            if index >= count:
                raise ValueError(f"Requested CUDA device index {index}, but only {count} device(s) available")
        return normalized
    raise ValueError(f"Unsupported device {device!r}; use 'auto', 'cpu', or 'cuda'")


def _is_cuda_oom(exc, torch_mod):
    if isinstance(exc, getattr(torch_mod.cuda, "OutOfMemoryError", ())):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


@dataclass
class LocalE5Encoder:
    """Local CPU/CUDA HuggingFace E5 encoder with non-truncating length validation and OOM recovery."""

    spec: dict
    local_files_only: bool = False
    device: str = "auto"
    fallback_to_cpu_on_oom: bool = True

    def __post_init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.requested_device = self.device
        self.device = resolve_device(self.device)
        self.oom_backoff_events = 0
        self.cpu_fallback_used = False
        self.min_batch_size_used = None
        self.peak_gpu_memory_mib = 0.0

        load_start = perf_counter()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec["model_name"],
            revision=self.spec["model_revision"],
            local_files_only=self.local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            self.spec["model_name"],
            revision=self.spec["model_revision"],
            local_files_only=self.local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()
        self.load_seconds = round(perf_counter() - load_start, 4)

    def check_lengths(self, formatted_texts):
        """Measure full input token length (prefix + text + special tokens) and forbid silent truncation."""
        encoded = self.tokenizer(
            formatted_texts,
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        lengths = [len(ids) for ids in encoded["input_ids"]]
        limit = self.spec["max_seq_length"]
        for idx, length in enumerate(lengths):
            if length > limit:
                raise ValueError(
                    f"Input {idx} has {length} tokens including prefix and special tokens, exceeding max_seq_length={limit}"
                )
        return lengths

    def _encode_batch(self, batch):
        torch = self._torch
        tokens = self.tokenizer(
            batch,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        if self.device != "cpu":
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = self.model(**tokens)
            hidden = outputs.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            if self.spec["normalize_embeddings"]:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            rows = pooled.detach().cpu().tolist()
        for row in rows:
            validate_vector(row, self.spec["dimensions"], distance_metric=self.spec["distance_metric"])
        return rows

    def encode_formatted(self, formatted_texts, *, batch_size=16, fallback_to_cpu=None):
        """Encode formatted inputs in CPU/CUDA batches with mean pooling, L2 normalization, and OOM backoff."""
        if not formatted_texts:
            return [], []
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        lengths = self.check_lengths(formatted_texts)
        torch = self._torch
        allow_cpu_fallback = self.fallback_to_cpu_on_oom if fallback_to_cpu is None else bool(fallback_to_cpu)

        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        vectors = []
        active_batch = batch_size
        pos = 0
        while pos < len(formatted_texts):
            batch = formatted_texts[pos:pos + active_batch]
            try:
                rows = self._encode_batch(batch)
            except RuntimeError as exc:
                if not (self.device.startswith("cuda") and _is_cuda_oom(exc, torch)):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if active_batch > 1:
                    active_batch = max(1, active_batch // 2)
                    self.oom_backoff_events += 1
                    continue
                if allow_cpu_fallback:
                    self.model.to("cpu")
                    self.device = "cpu"
                    self.cpu_fallback_used = True
                    self.oom_backoff_events += 1
                    continue
                raise
            if self.min_batch_size_used is None or active_batch < self.min_batch_size_used:
                self.min_batch_size_used = active_batch
            vectors.extend(rows)
            pos += len(batch)

        if self.device.startswith("cuda") and torch.cuda.is_available():
            peak_mib = round(torch.cuda.max_memory_allocated() / 1024**2, 3)
            self.peak_gpu_memory_mib = max(self.peak_gpu_memory_mib, peak_mib)
        return vectors, lengths

    def encode_passages(self, texts, *, batch_size=16):
        formatted = [format_passage_input(t, self.spec) for t in texts]
        return self.encode_formatted(formatted, batch_size=batch_size)

    def encode_queries(self, queries, *, batch_size=16):
        formatted = [format_query_input(q, self.spec) for q in queries]
        return self.encode_formatted(formatted, batch_size=batch_size)


def verify_cpu_gpu_compatibility(
    texts,
    *,
    model_name=DEFAULT_MODEL,
    revision=None,
    batch_size=4,
    local_files_only=False,
    min_cosine=0.9999,
):
    """Compare CPU vs CUDA embeddings on a small smoke sample without requiring bitwise equality."""
    spec = get_model_spec(model_name, revision=revision)
    cpu_enc = LocalE5Encoder(spec, local_files_only=local_files_only, device="cpu")
    cpu_vecs, lengths = cpu_enc.encode_passages(texts, batch_size=batch_size)

    import torch

    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "count": len(texts),
            "dimensions": spec["dimensions"],
            "token_lengths": lengths,
            "compatible": True,
        }

    cuda_enc = LocalE5Encoder(spec, local_files_only=local_files_only, device="cuda")
    cuda_vecs, _ = cuda_enc.encode_passages(texts, batch_size=batch_size)
    cosines = []
    max_abs_diff = 0.0
    for cv, gv in zip(cpu_vecs, cuda_vecs, strict=True):
        cosines.append(sum(a * b for a, b in zip(cv, gv, strict=True)))
        max_abs_diff = max(max_abs_diff, *(abs(a - b) for a, b in zip(cv, gv, strict=True)))
    min_observed = min(cosines, default=1.0)
    if min_observed < min_cosine:
        raise ValueError(f"CPU/CUDA embedding cosine {min_observed:.6f} fell below {min_cosine}")
    return {
        "cuda_available": True,
        "count": len(texts),
        "dimensions": spec["dimensions"],
        "token_lengths": lengths,
        "min_cosine_similarity": round(min_observed, 8),
        "max_abs_diff": round(max_abs_diff, 8),
        "compatible": True,
    }


def _atomic_write_text(target, content):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)


def _atomic_write_jsonl(target, rows):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)


def _load_valid_cache(cache_path, spec):
    """Load cached chunk embeddings matching the pinned model config and vector invariants."""
    cached = {}
    path = Path(cache_path)
    if not path.is_file():
        return cached
    for row in read_jsonl(path):
        if row.get("model_config_fingerprint") != spec["model_config_fingerprint"]:
            continue
        cid = row.get("chunk_id")
        sha = row.get("input_sha256")
        vec = row.get("embedding")
        if not cid or not sha or not isinstance(vec, list):
            continue
        try:
            validate_vector(vec, spec["dimensions"], distance_metric=spec["distance_metric"])
        except ValueError:
            continue
        cached[(cid, sha)] = row
    return cached


def _peak_rss_mib():
    # Linux ru_maxrss is in KiB.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3)


def embed_chunks(
    chunks,
    output_dir,
    *,
    model_name=DEFAULT_MODEL,
    revision=None,
    batch_size=None,
    gpu_batch_size=64,
    device="auto",
    fallback_to_cpu=True,
    resume=True,
    encoder=None,
    local_files_only=False,
):
    """Embed chunk records locally (CPU or CUDA) with deterministic caching and PostgreSQL-compatible output."""
    start = perf_counter()
    spec = get_model_spec(model_name, revision=revision)
    if encoder is not None:
        resolved_device = getattr(encoder, "device", "cpu")
        effective_batch = batch_size if batch_size is not None else 16
    else:
        resolved_device = resolve_device(device)
        effective_batch = batch_size if batch_size is not None else (
            gpu_batch_size if resolved_device.startswith("cuda") else 16
        )
    if effective_batch < 1:
        raise ValueError("batch_size must be >= 1")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoints" / f"{spec['model_config_fingerprint']}.jsonl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Reject empty chunk sets or unauthorized external configurations.
    if not chunks:
        raise ValueError("No chunks supplied for embedding")

    cache = _load_valid_cache(checkpoint_path, spec) if resume else {}
    existing_out = out / "chunk_embeddings.jsonl"
    if resume and existing_out.is_file():
        cache.update(_load_valid_cache(existing_out, spec))

    ordered_chunks = sorted(chunks, key=lambda c: (c["ordinal"], c["chunk_id"]))
    pending = []
    results_by_key = {}

    for chunk in ordered_chunks:
        text = chunk["retrieval_text"]
        if not text.strip():
            raise ValueError(f"Chunk {chunk['chunk_id']} has empty retrieval_text")
        formatted = format_passage_input(text, spec)
        digest = input_digest(formatted)
        key = (chunk["chunk_id"], digest)
        if key in cache:
            cached_row = dict(cache[key])
            # Refresh preserved chunk/citation metadata while keeping cached embedding.
            cached_row.update({
                "ordinal": chunk["ordinal"],
                "asset_key": chunk["asset_key"],
                "record_id": chunk["record_id"],
                "asset_sha256": chunk["asset_sha256"],
                "language_tag": chunk["language_tag"],
                "section_key": chunk.get("section_key"),
                "physical_pages": chunk.get("physical_pages", []),
                "spans": chunk.get("spans", []),
            })
            results_by_key[key] = cached_row
        else:
            pending.append((chunk, formatted, digest))

    model_load_seconds = 0.0
    inference_seconds = 0.0
    active_encoder = encoder
    if pending:
        active_encoder = encoder or LocalE5Encoder(
            spec,
            local_files_only=local_files_only,
            device=resolved_device,
            fallback_to_cpu_on_oom=fallback_to_cpu,
        )
        model_load_seconds = getattr(active_encoder, "load_seconds", 0.0)
        inf_start = perf_counter()
        with checkpoint_path.open("a", encoding="utf-8") as ckpt_stream:
            pos = 0
            step_batch = effective_batch
            while pos < len(pending):
                batch_items = pending[pos:pos + step_batch]
                texts = [item[1] for item in batch_items]
                vectors, lengths = active_encoder.encode_formatted(texts, batch_size=step_batch)
                min_used = getattr(active_encoder, "min_batch_size_used", None)
                if min_used and min_used < step_batch:
                    step_batch = min_used
                for (chunk, _, digest), vec, tok_len in zip(batch_items, vectors, lengths):
                    norm = validate_vector(vec, spec["dimensions"], distance_metric=spec["distance_metric"])
                    row = {
                        "chunk_id": chunk["chunk_id"],
                        "ordinal": chunk["ordinal"],
                        "asset_key": chunk["asset_key"],
                        "record_id": chunk["record_id"],
                        "asset_sha256": chunk["asset_sha256"],
                        "language_tag": chunk["language_tag"],
                        "section_key": chunk.get("section_key"),
                        "physical_pages": chunk.get("physical_pages", []),
                        "spans": chunk.get("spans", []),
                        "provider": spec["provider"],
                        "model_name": spec["model_name"],
                        "model_revision": spec["model_revision"],
                        "input_profile": spec["input_profile"],
                        "model_config_fingerprint": spec["model_config_fingerprint"],
                        "dimensions": spec["dimensions"],
                        "distance_metric": spec["distance_metric"],
                        "input_sha256": digest,
                        "input_token_count": tok_len,
                        "l2_norm": round(norm, 6),
                        "embedding": vec,
                    }
                    results_by_key[(chunk["chunk_id"], digest)] = row
                    ckpt_stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                ckpt_stream.flush()
                os.fsync(ckpt_stream.fileno())
                pos += len(batch_items)
        inference_seconds = round(perf_counter() - inf_start, 4)

    ordered_rows = [
        results_by_key[(c["chunk_id"], input_digest(format_passage_input(c["retrieval_text"], spec)))]
        for c in ordered_chunks
    ]
    model_row = {
        "provider": spec["provider"],
        "model_name": spec["model_name"],
        "model_revision": spec["model_revision"],
        "input_profile": spec["input_profile"],
        "dimensions": spec["dimensions"],
        "distance_metric": spec["distance_metric"],
        "is_external": spec["is_external"],
        "pooling": spec["pooling"],
        "normalize_embeddings": spec["normalize_embeddings"],
        "query_prefix": spec["query_prefix"],
        "passage_prefix": spec["passage_prefix"],
        "max_seq_length": spec["max_seq_length"],
        "model_config_fingerprint": spec["model_config_fingerprint"],
    }
    _atomic_write_text(out / "embedding_model.json", json.dumps(model_row, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    _atomic_write_jsonl(out / "chunk_embeddings.jsonl", ordered_rows)

    max_input_tokens = max((r.get("input_token_count", 0) for r in ordered_rows), default=0)
    summary = {
        "model_name": spec["model_name"],
        "model_revision": spec["model_revision"],
        "input_profile": spec["input_profile"],
        "model_config_fingerprint": spec["model_config_fingerprint"],
        "dimensions": spec["dimensions"],
        "distance_metric": spec["distance_metric"],
        "is_external": spec["is_external"],
        "requested_device": device,
        "device": getattr(active_encoder, "device", resolved_device) if active_encoder is not None else resolved_device,
        "batch_size": effective_batch,
        "min_effective_batch_size": (getattr(active_encoder, "min_batch_size_used", None) or effective_batch) if active_encoder is not None else effective_batch,
        "oom_backoff_events": getattr(active_encoder, "oom_backoff_events", 0) if active_encoder is not None else 0,
        "cpu_fallback_used": getattr(active_encoder, "cpu_fallback_used", False) if active_encoder is not None else False,
        "peak_gpu_memory_mib": getattr(active_encoder, "peak_gpu_memory_mib", 0.0) if active_encoder is not None else 0.0,
        "total_chunks": len(ordered_rows),
        "encoded_chunks": len(pending),
        "reused_cached_chunks": len(ordered_rows) - len(pending),
        "max_input_tokens": max_input_tokens,
        "max_seq_length": spec["max_seq_length"],
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "elapsed_seconds": round(perf_counter() - start, 4),
        "peak_rss_mib": _peak_rss_mib(),
        "validation_errors": 0,
    }
    _atomic_write_text(out / "embedding_summary.json", json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return ordered_rows, summary


def add_command(commands):
    command = commands.add_parser("embed", help="Generate deterministic CPU/CUDA embeddings for chunk JSONL outputs.")
    command.add_argument("--input", type=Path, required=True,
                         help="Chunk run directory (containing chunks.jsonl) or explicit chunks.jsonl file.")
    command.add_argument("--output", type=Path, required=True, help="Output directory for embeddings and model metadata.")
    command.add_argument("--model", choices=sorted(SUPPORTED_MODELS), default=DEFAULT_MODEL)
    command.add_argument("--revision", help="Override pinned HuggingFace commit SHA.")
    command.add_argument("--device", default="auto", help="Execution device: auto, cpu, cuda, or cuda:<index>.")
    command.add_argument("--batch-size", type=int, default=None,
                         help="Explicit inference batch size (defaults to 16 on CPU or --gpu-batch-size on CUDA).")
    command.add_argument("--gpu-batch-size", type=int, default=64,
                         help="Default batch size when running on CUDA (default 64).")
    command.add_argument("--no-cpu-fallback", action="store_true",
                         help="Fail instead of falling back to CPU if CUDA OOM persists at batch size 1.")
    command.add_argument("--no-resume", action="store_true", help="Re-encode all chunks without reusing cache.")
    command.add_argument("--local-files-only", action="store_true", help="Load model strictly from local HuggingFace cache.")
    return command


def main(args):
    input_path = args.input.resolve()
    chunks_path = input_path / "chunks.jsonl" if input_path.is_dir() else input_path
    chunks = read_jsonl(chunks_path)
    _, summary = embed_chunks(
        chunks,
        args.output.resolve(),
        model_name=args.model,
        revision=args.revision,
        batch_size=args.batch_size,
        gpu_batch_size=args.gpu_batch_size,
        device=args.device,
        fallback_to_cpu=not args.no_cpu_fallback,
        resume=not args.no_resume,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0
