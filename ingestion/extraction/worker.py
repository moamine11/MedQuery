"""Isolated CPU page worker. The parent owns deadlines and checkpoint commits."""

from contextlib import redirect_stdout
import json
from pathlib import Path
import resource
import sys


def main():
    request = json.load(sys.stdin)
    limit = request["profile"]["worker_memory_mib"] * 1024**2
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    import pymupdf

    from .engines import PageEngines
    from .page import process_page
    path = Path(request["snapshot"])
    context = request["context"]
    with redirect_stdout(sys.stderr), pymupdf.open(path) as document:
        if document.is_encrypted or document.page_count != request["expected_pages"]:
            raise ValueError("PDF encryption or page tree does not match the validated inventory")
        page = document[context["physical_page"] - 1]
        record = process_page(PageEngines(path, context["physical_page"], page, request["profile"],
                                          request["tools"]), context)
    print(json.dumps(record, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
