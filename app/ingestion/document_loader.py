"""Turns files in the docs folder into Document objects.

File format: `key: value` header lines, then `---`, then the body.
One bad file is logged and skipped.
"""

import logging
from datetime import date
from pathlib import Path

from app.core.exceptions import DocumentLoadError, UnsupportedFileTypeError
from app.models import Document, DocumentCategory

log = logging.getLogger("linemate.ingestion")

REQUIRED_FIELDS = ("id", "title", "category", "owner_id", "last_reviewed_at")
SEPARATOR = "---"


def _parse_frontmatter(text: str, source: Path) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    try:
        split_at = next(i for i, line in enumerate(lines) if line.strip() == SEPARATOR)
    except StopIteration:
        raise DocumentLoadError(f"{source.name}: missing '---' separator line") from None

    fields: dict[str, str] = {}
    for line in lines[:split_at]:
        if not line.strip():
            continue
        if ":" not in line:
            raise DocumentLoadError(f"{source.name}: malformed header line {line!r}")
        key, _, value = line.partition(":")  # only the first colon splits, so titles may contain ':'
        fields[key.strip()] = value.strip()

    missing = [name for name in REQUIRED_FIELDS if name not in fields]
    if missing:
        raise DocumentLoadError(f"{source.name}: missing required field(s) {missing}")

    return fields, "\n".join(lines[split_at + 1:]).strip()


def load_one_document(path: Path) -> Document:
    if path.suffix != ".md":  # cheap check first, before touching the file
        raise UnsupportedFileTypeError(
            f"{path.name}: unsupported file type {path.suffix!r} (only .md is supported)"
        )

    fields, body = _parse_frontmatter(path.read_text(encoding="utf-8"), path)

    try:
        category = DocumentCategory(fields["category"])
    except ValueError as exc:
        raise DocumentLoadError(f"{path.name}: unknown category {fields['category']!r}") from exc

    try:
        last_reviewed_at = date.fromisoformat(fields["last_reviewed_at"])
    except ValueError as exc:
        raise DocumentLoadError(
            f"{path.name}: invalid last_reviewed_at {fields['last_reviewed_at']!r} (want YYYY-MM-DD)"
        ) from exc

    try:
        document_id = int(fields["id"])
        owner_id = int(fields["owner_id"])
    except ValueError as exc:
        raise DocumentLoadError(f"{path.name}: id and owner_id must be integers") from exc

    if not body:
        raise DocumentLoadError(f"{path.name}: document body is empty")

    return Document(document_id, fields["title"], category, body, owner_id, last_reviewed_at)


def load_documents_from_folder(folder: str | Path) -> list[Document]:
    documents: list[Document] = []
    seen_ids: set[int] = set()

    for path in sorted(Path(folder).iterdir()):
        if path.is_dir():
            continue
        try:
            document = load_one_document(path)
            if document.id in seen_ids:
                raise DocumentLoadError(f"{path.name}: duplicate document id {document.id}")
        except DocumentLoadError as exc:
            log.warning("SKIPPED %s", exc)
            continue
        seen_ids.add(document.id)
        documents.append(document)

    return documents