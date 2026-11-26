"""The local extraction command for the MedQuery ingestion CLI."""

import json
from pathlib import Path
import sys

from ingestion.intake import IntakeError

from .config import ROOT, load_profile, local_tools
from .selection import select, select_from_manifest


def add_command(commands):
    command = commands.add_parser("extract", help="Extract selected, validated PDFs using local engines.")
    command.add_argument("--root", type=Path, help="Read-only collection root (required with --inventory).")
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument("--inventory", type=Path, help="Phase 2 validated inventory.jsonl.")
    source.add_argument("--selection-manifest", type=Path,
                        help="Executable extraction selection manifest JSON combining unified plan and eligibility rules.")
    command.add_argument("--pilot", type=Path, help="Pilot CSV restricting asset identity and default physical pages.")
    command.add_argument("--document", action="append", default=[], help="Record ID or pilot asset ID. Repeatable.")
    command.add_argument("--pages", action="append", default=[],
                         help="Physical pages such as 1,3-5 or ASSET:1,3-5. Repeatable.")
    command.add_argument("--output", type=Path, required=True, help="Run directory under this repository's ignored data tree.")
    command.add_argument("--profile", type=Path, help="JSON profile overriding the built-in settings.")
    command.add_argument("--workers", type=int, help="Concurrent page workers, within the profile's memory budget.")
    command.add_argument("--resume", action="store_true", help="Reuse matching complete page checkpoints.")
    return command


def main(args):
    from .runner import run

    try:
        output = args.output.resolve()
        data = ROOT / "data"
        if data.is_symlink() or output == data or not output.is_relative_to(data):
            raise ValueError("--output must be under ignored data/")
        if args.selection_manifest:
            if args.pilot:
                raise ValueError("--pilot cannot be combined with --selection-manifest")
            root = args.root.resolve() if args.root else ROOT
            if not root.is_dir():
                raise ValueError("--root must be a readable directory")
            assets, exclusions, asset_roots = select_from_manifest(
                args.selection_manifest, args.document, args.pages, root=root,
            )
            roots_to_check = [*(asset_roots.values()), *([root] if args.root else [])]
        else:
            if not args.root:
                raise ValueError("--root is required with --inventory")
            root = args.root.resolve()
            if not root.is_dir():
                raise ValueError("--root must be a readable directory")
            assets, exclusions = select(args.inventory, args.pilot, args.document, args.pages)
            asset_roots = None
            roots_to_check = [root]
        if any(output.is_relative_to(item) for item in roots_to_check):
            raise ValueError("--output must be outside the collection root")
        input_paths = [args.inventory, args.selection_manifest, args.pilot, args.profile]
        if any(p and p.resolve().is_relative_to(output) for p in input_paths):
            raise ValueError("Run output overlaps an input file")
        profile = load_profile(args.profile, workers=args.workers)
        tools, versions = local_tools(profile)
        result = run(root, assets, exclusions, output, profile, tools, versions, resume=args.resume,
                     progress=lambda line: print(line, file=sys.stderr, flush=True), asset_roots=asset_roots)
    except (OSError, ValueError, TypeError, KeyError, IntakeError) as error:
        print(f"Extraction failed: {error}", file=sys.stderr)
        return 2
    printable = {key: result[key] for key in (
        "assets", "excluded_assets", "page_count", "page_statuses", "reused_pages",
        "review_items", "page_worker_failures", "elapsed_seconds", "peak_worker_rss_mib",
    )}
    print(json.dumps(printable, ensure_ascii=False, sort_keys=True))
    return 1 if result["page_statuses"].get("failed") or result["page_statuses"].get("unreadable") or not result["page_count"] else 0
