#!/usr/bin/env python3
"""Reproducible Phase 6 retrieval evaluation for E5-small vs E5-base and chunking strategies."""

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.chunking import TokenCounter, chunk_pages, read_jsonl, validate_chunk_citations, write_jsonl
from ingestion.embeddings import LocalE5Encoder, embed_chunks, get_model_spec


def _matches_question(question, chunk):
    if chunk["asset_key"] != question["expected_asset_key"]:
        return False
    if not (set(chunk["physical_pages"]) & set(question["expected_pages"])):
        return False
    text = chunk["retrieval_text"].replace("\u00ad", "")
    subs = question["required_substrings"]
    if question.get("match_any_substring"):
        return any(s in text for s in subs)
    return all(s in text for s in subs)


def _cosine_score(vec_a, vec_b):
    return sum(a * b for a, b in zip(vec_a, vec_b))


def evaluate_single(
    chunks_path,
    questions_path,
    output_dir,
    *,
    model_name,
    batch_size=16,
    pages_path=ROOT / "data/ingestion/phase5/comparison_corpus/pages.jsonl",
):
    chunks = read_jsonl(chunks_path)
    pages = read_jsonl(pages_path)
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    spec = get_model_spec(model_name)

    encoder = LocalE5Encoder(spec)
    emb_rows, emb_summary = embed_chunks(
        chunks,
        output_dir,
        model_name=model_name,
        batch_size=batch_size,
        resume=False,
        encoder=encoder,
    )
    vectors = [row["embedding"] for row in emb_rows]
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    q_start = perf_counter()
    q_vectors, q_lengths = encoder.encode_queries([q["query"] for q in questions], batch_size=batch_size)
    query_seconds = round(perf_counter() - q_start, 4)

    citation_errors = len(validate_chunk_citations(chunks, pages))
    per_question = []
    for q, q_vec, q_len in zip(questions, q_vectors, q_lengths):
        scored = []
        for idx, row in enumerate(emb_rows):
            scored.append((_cosine_score(q_vec, vectors[idx]), idx, row))
        scored.sort(key=lambda item: (-item[0], item[1]))
        top5 = scored[:5]

        first_match_rank = None
        for rank, (_, _, row) in enumerate(scored, 1):
            if _matches_question(q, chunk_by_id[row["chunk_id"]]):
                first_match_rank = rank
                break

        hit_at_5 = first_match_rank is not None and first_match_rank <= 5
        page_hit_at_5 = any(
            row["asset_key"] == q["expected_asset_key"]
            and (set(row["physical_pages"]) & set(q["expected_pages"]))
            for _, _, row in top5
        )
        top1_chunk = chunk_by_id[top5[0][2]["chunk_id"]]
        per_question.append({
            "id": q["id"],
            "query_language": q["query_language"],
            "target_language": q["target_language"],
            "cross_lingual": q["cross_lingual"],
            "query_tokens": q_len,
            "hit_at_5": hit_at_5,
            "page_hit_at_5": page_hit_at_5,
            "first_match_rank": first_match_rank,
            "rr_at_5": round(1.0 / first_match_rank, 4) if hit_at_5 else 0.0,
            "rr_full": round(1.0 / first_match_rank, 4) if first_match_rank else 0.0,
            "top1_asset_key": top1_chunk["asset_key"],
            "top1_pages": top1_chunk["physical_pages"],
            "top1_score": round(top5[0][0], 4),
            "top5_assets": [(r["asset_key"], r["physical_pages"], round(s, 4)) for s, _, r in top5],
        })

    def _agg(items):
        n = len(items)
        if not n:
            return {"count": 0, "recall_at_5": 0.0, "page_recall_at_5": 0.0, "mrr_at_5": 0.0, "mrr_full": 0.0}
        return {
            "count": n,
            "recall_at_5": round(sum(x["hit_at_5"] for x in items) / n, 4),
            "page_recall_at_5": round(sum(x["page_hit_at_5"] for x in items) / n, 4),
            "mrr_at_5": round(sum(x["rr_at_5"] for x in items) / n, 4),
            "mrr_full": round(sum(x["rr_full"] for x in items) / n, 4),
        }

    by_bucket = {
        "overall": _agg(per_question),
        "fr_monolingual": _agg([x for x in per_question if not x["cross_lingual"] and x["query_language"] == "fr"]),
        "en_monolingual": _agg([x for x in per_question if not x["cross_lingual"] and x["query_language"] == "en"]),
        "ar_monolingual": _agg([x for x in per_question if not x["cross_lingual"] and x["query_language"] == "ar"]),
        "cross_lingual": _agg([x for x in per_question if x["cross_lingual"]]),
    }

    dims = spec["dimensions"]
    bytes_per_vec = dims * 4
    n_chunks = len(emb_rows)
    peak_rss_mib = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3)

    result = {
        "model_name": spec["model_name"],
        "model_revision": spec["model_revision"],
        "dimensions": dims,
        "chunk_count": n_chunks,
        "citation_validation_errors": citation_errors,
        "max_passage_input_tokens": emb_summary["max_input_tokens"],
        "max_query_input_tokens": max(q_lengths),
        "max_seq_length": spec["max_seq_length"],
        "timing": {
            "model_load_seconds": encoder.load_seconds,
            "embedding_generation_seconds": emb_summary["inference_seconds"],
            "query_encoding_seconds": query_seconds,
            "query_ms_per_query": round(1000 * query_seconds / len(questions), 2),
        },
        "peak_rss_mib": peak_rss_mib,
        "storage": {
            "float32_bytes_per_vector": bytes_per_vec,
            "pilot_float32_kib": round((n_chunks * bytes_per_vec) / 1024, 2),
            "per_10k_chunks_float32_mib": round((10000 * bytes_per_vec) / (1024 * 1024), 2),
            "projected_6884_pages_float32_mib": round(((n_chunks / 43.0) * 6884 * bytes_per_vec) / (1024 * 1024), 2),
        },
        "metrics": by_bucket,
        "per_question": per_question,
    }
    Path(output_dir, "eval_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--chunks", type=Path)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/ingestion/phase6/eval_questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/ingestion/phase6")
    parser.add_argument("--model")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.single:
        res = evaluate_single(args.chunks, args.questions, args.output, model_name=args.model, batch_size=args.batch_size)
        print(json.dumps({"model": res["model_name"], "metrics": res["metrics"], "peak_rss_mib": res["peak_rss_mib"]}))
        return

    out_root = args.output.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    corpus_dir = ROOT / "data/ingestion/phase5/comparison_corpus"
    pages = read_jsonl(corpus_dir / "pages.jsonl")
    sections = read_jsonl(corpus_dir / "sections.jsonl")
    docs_meta = json.loads((corpus_dir / "documents.json").read_text(encoding="utf-8"))["assets"]
    tok = TokenCounter()

    # Reuse existing Phase 5 section_aware chunks and materialize fixed baselines for E5-small comparison.
    chunk_sets = {
        "section_aware": ROOT / "data/ingestion/phase5/cli_chunk_run/chunks.jsonl",
    }
    for strat in ("fixed_256_32", "fixed_512_64"):
        strat_chunks, _ = chunk_pages(pages, sections=sections, documents=docs_meta, strategy=strat, counter=tok)
        strat_path = out_root / f"chunks_{strat}.jsonl"
        write_jsonl(strat_path, strat_chunks)
        chunk_sets[strat] = strat_path

    runs = [
        ("e5_small_section_aware", "intfloat/multilingual-e5-small", chunk_sets["section_aware"]),
        ("e5_base_section_aware", "intfloat/multilingual-e5-base", chunk_sets["section_aware"]),
        ("e5_small_fixed_256_32", "intfloat/multilingual-e5-small", chunk_sets["fixed_256_32"]),
        ("e5_small_fixed_512_64", "intfloat/multilingual-e5-small", chunk_sets["fixed_512_64"]),
    ]

    results = {}
    for run_key, model_name, chunks_path in runs:
        run_out = out_root / run_key
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--single",
                "--chunks",
                str(chunks_path),
                "--questions",
                str(args.questions),
                "--output",
                str(run_out),
                "--model",
                model_name,
                "--batch-size",
                str(args.batch_size),
            ],
            check=True,
        )
        results[run_key] = json.loads((run_out / "eval_result.json").read_text(encoding="utf-8"))

    summary = {
        "note": "Preliminary source-grounded evaluation on 24 questions across 43 contiguous pages (not clinician-validated).",
        "runs": {
            k: {
                "model_name": v["model_name"],
                "model_revision": v["model_revision"],
                "dimensions": v["dimensions"],
                "chunk_count": v["chunk_count"],
                "citation_validation_errors": v["citation_validation_errors"],
                "max_passage_input_tokens": v["max_passage_input_tokens"],
                "max_query_input_tokens": v["max_query_input_tokens"],
                "timing": v["timing"],
                "peak_rss_mib": v["peak_rss_mib"],
                "storage": v["storage"],
                "metrics": v["metrics"],
            }
            for k, v in results.items()
        },
    }
    (out_root / "evaluation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
