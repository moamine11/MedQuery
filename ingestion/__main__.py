"""Command-line entry point for local MedQuery ingestion."""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from .adapters import ADAPTERS, CATEGORIES, canonical
from .intake import IntakeError
from .metadata import validate
from .pipeline import intake, no_metadata
from .reconciliation import identity, revision


def arguments():
    parser = argparse.ArgumentParser(description="Inspect and reconcile local MedQuery collections without database writes.")
    commands = parser.add_subparsers(dest="command", required=True)
    descriptions = {"inspect": "Discover PDFs and inspect file structure and optional metadata.",
                    "validate": "Validate supplied metadata and compare PDF observations.",
                    "reconcile": "Validate inputs and build a review queue with identity candidates."}
    for name, description in descriptions.items():
        sub = commands.add_parser(name, help=description, description=description)
        sub.add_argument("--root", type=Path, required=True, help="Read-only collection directory.")
        sub.add_argument("--collection", choices=sorted(CATEGORIES), required=True)
        source = sub.add_mutually_exclusive_group(required=name != "inspect")
        source.add_argument("--manifest", type=Path, help="Canonical UTF-8 JSONL manifest; relative to the collection root.")
        source.add_argument("--adapter", choices=sorted(ADAPTERS), help="Convert collection metadata to canonical JSONL records.")
        sub.add_argument("--metadata", type=Path, help="CSV file for the generic adapter; relative to the collection root.")
        sub.add_argument("--output", type=Path, help="Output directory outside the collection root.")
        sub.add_argument("--dry-run", action="store_true", help="Print the summary without writing output files.")
        sub.add_argument("--workers", type=int, default=4, help="Concurrent structural probes (1–16, default 4).")
        if name == "reconcile":
            sub.add_argument("--reference-inventory", type=Path, action="append", default=[],
                             help="Trusted local inventory.jsonl from a different collection. Repeatable.")
    from .chunking import add_command as add_chunk_command
    from .embeddings import add_command as add_embed_command
    from .extraction.cli import add_command
    from .storage import add_command as add_load_command

    add_command(commands)
    add_chunk_command(commands)
    add_embed_command(commands)
    add_load_command(commands)
    return parser


def load_references(paths, collection):
    entries = []
    seen = set()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                if item.get("validation_state") != "valid":
                    continue
                normalized, issues = validate(item["metadata"])
                if any(issue["severity"] == "error" for issue in issues):
                    raise ValueError(f"Invalid metadata in reference inventory: {path}")
                observations = item.get("observations", {})
                if item["collection"] not in CATEGORIES or item["collection"] == collection:
                    raise ValueError("Reference inventory must use a different known collection code.")
                for asset in normalized["assets"]:
                    observed = observations.get(asset["path"], {})
                    digest = observed.get("sha256", "")
                    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                        raise ValueError(f"Missing observed SHA-256 in reference inventory: {path}")
                key = (item["collection"], normalized["record_id"])
                if key in seen:
                    raise ValueError("Reference inventories repeat a collection/record ID.")
                seen.add(key)
                work = identity(normalized)
                entries.append({**item, "metadata": normalized, "identity_key": work,
                                "revision_key": revision(normalized, work)})
    return entries


def dump_outputs(output, result):
    output.mkdir(parents=True, exist_ok=True)
    for key, value in result.items():
        jsonl = key != "summary"
        target = output / (key + (".jsonl" if jsonl else ".json"))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                if jsonl:
                    for row in value:
                        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                else:
                    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
                stream.flush()
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def main():
    parser = arguments()
    args = parser.parse_args()
    if args.command == "extract":
        from .extraction.cli import main as extract

        return extract(args)
    if args.command == "chunk":
        from .chunking import main as chunk

        return chunk(args)
    if args.command == "embed":
        from .embeddings import main as embed

        return embed(args)
    if args.command == "load":
        from .storage import main as load

        return load(args)
    root = args.root.resolve()
    output = args.output.resolve() if args.output else None
    if not root.is_dir():
        parser.error("--root must be a readable directory.")
    if not args.dry_run and output is None:
        parser.error("--output is required unless --dry-run is used.")
    if output and output.is_relative_to(root):
        parser.error("--output must be outside the source tree.")
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16.")
    if bool(args.metadata) != (args.adapter == "csv"):
        parser.error("--metadata is required with --adapter csv and is only valid with that adapter.")
    if args.adapter == "therapeutic-guidelines" and args.collection != "therapeutic_guidelines":
        parser.error("The Therapeutic Guidelines adapter requires its own collection code.")
    if args.adapter == "treatment-protocols" and args.collection != "treatment_protocols":
        parser.error("The Treatment Protocols adapter requires its own collection code.")

    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        parser.error("Install Poppler to provide pdfinfo.")
    metadata = args.manifest or args.metadata
    if metadata and not metadata.is_absolute():
        metadata = root / metadata
    if output and metadata and metadata.resolve().parent == output and metadata.name in {
        "inventory.jsonl", "files.jsonl", "issues.jsonl", "review_queue.jsonl", "summary.json"
    }:
        parser.error("--output overlaps a supplied input file.")
    adapter = canonical if args.manifest else ADAPTERS.get(args.adapter, no_metadata)
    ref_paths = getattr(args, "reference_inventory", [])
    if output and any(p.resolve() == output / "inventory.jsonl" for p in ref_paths):
        parser.error("--output overlaps a reference inventory.")
    try:
        references = load_references(ref_paths, args.collection)
        result = intake(root, args.collection, adapter, metadata, pdfinfo, workers=args.workers,
                        references=references, matching=args.command == "reconcile")
        if not args.dry_run:
            dump_outputs(output, result)
    except (OSError, ValueError, TypeError, KeyError, IntakeError) as error:
        print(f"Intake failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 1 if result["summary"]["error_count"] else 0


if __name__ == "__main__":
    sys.exit(main())
