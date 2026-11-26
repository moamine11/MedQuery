"""Confined discovery and PDF structure observation."""

import hashlib
import os
import re
import stat
import subprocess
from contextlib import contextmanager
from pathlib import PurePosixPath

EXCLUDED_DIRS = {
    "archive", "archives", "backup", "backups", "docs", "reports", "collection_reports",
    "metadata", "tmp", "temp", "cache", "processed",
}


class IntakeError(Exception):
    def __init__(self, code, message, observed=None):
        super().__init__(message)
        self.code = code
        self.observed = observed


def excluded(name):
    value = name.casefold()
    return (value.startswith((".", "~", "report_category_"))
            or value in EXCLUDED_DIRS
            or "collection_report" in value or "collectionreport" in value
            or value.endswith((".tmp.pdf", ".part.pdf", ".bak.pdf", ".~pdf")))


def asset_parts(value):
    if not isinstance(value, str):
        raise IntakeError("invalid_path", "Asset path must be a string.")
    parts = value.split("/")
    if (any(part in ("", ".", "..") for part in parts) or "\\" in value
            or any(ord(char) < 32 for char in value) or re.match(r"^[A-Za-z]:", value)):
        raise IntakeError("invalid_path", "Use a relative POSIX path contained by the collection root.")
    if any(excluded(part) for part in parts):
        raise IntakeError("excluded_path", "Asset is in an excluded directory or has an excluded name.")
    return parts


@contextmanager
def open_local(root, parts):
    """Open a regular file through directory descriptors, rejecting symlink components."""
    descriptors = []
    try:
        parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(parent)
        for component in parts[:-1]:
            parent = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            descriptors.append(parent)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        descriptors.append(fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise IntakeError("invalid_file", "Expected a regular file.")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            yield stream
    except FileNotFoundError as error:
        raise IntakeError("missing_file", "File does not exist under the collection root.") from error
    except OSError as error:
        raise IntakeError("unsafe_or_unreadable_path", f"Cannot open confined path (errno {error.errno}).") from error
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def discover(root):
    files = []
    for directory, folders, names in os.walk(root, followlinks=False):
        relative = PurePosixPath(os.path.relpath(directory, root))
        folders[:] = sorted(name for name in folders
                            if not excluded(name) and not root.joinpath(relative, name).is_symlink())
        files.extend((relative / name).as_posix() for name in names
                     if not excluded(name) and PurePosixPath(name).suffix.casefold() == ".pdf")
    return sorted(files)


def observe_pdf(root, path, pdfinfo):
    parts = asset_parts(path)
    if PurePosixPath(path).suffix.casefold() != ".pdf":
        raise IntakeError("unsupported_file", "Phase 2 accepts PDF assets.")
    with open_local(root, parts) as stream:
        before = os.fstat(stream.fileno())
        if not re.search(rb"%PDF-\d\.\d", stream.read(1024)):
            raise IntakeError("corrupt_pdf", "Missing PDF signature.")
        stream.seek(max(0, before.st_size - 65536))
        if not re.search(rb"(?:^|[\r\n])%%EOF[ \t]*(?:[\r\n]|$)", stream.read()):
            raise IntakeError("corrupt_pdf", "Missing PDF end marker.")
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        try:
            result = subprocess.run(
                [pdfinfo, f"/proc/self/fd/{stream.fileno()}"], pass_fds=(stream.fileno(),),
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=30, env={**os.environ, "LC_ALL": "C"},
            )
        except subprocess.TimeoutExpired as error:
            raise IntakeError("probe_timeout", "PDF structure probe timed out.") from error
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise IntakeError("file_changed", "File changed during inspection. Retry with a stable copy.")
    pages = re.search(r"(?m)^Pages:\s+(\d+)", result.stdout)
    observed = {"sha256": digest, "bytes": before.st_size, "media_type": "application/pdf"}
    if pages and int(pages[1]) > 0:
        observed["pages"] = int(pages[1])
    if "incorrect password" in result.stderr.lower() or re.search(r"(?m)^Encrypted:\s+yes\b", result.stdout):
        raise IntakeError("encrypted_pdf", "PDF is encrypted.", observed)
    if result.returncode or not pages or int(pages[1]) < 1:
        raise IntakeError("corrupt_pdf", "PDF structure or page tree could not be read.")
    warnings = ["pdf_structure_warning"] if result.stderr.strip() else []
    return observed, warnings
