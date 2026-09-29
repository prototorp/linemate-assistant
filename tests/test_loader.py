"""Unit tests for the document loader"""

import logging

from app.ingestion.document_loader import load_documents_from_folder

VALID = (
    "id: 1\n"
    "title: Test Doc\n"
    "category: SOP\n"
    "owner_id: 100\n"
    "last_reviewed_at: 2026-01-01\n"
    "---\n"
    "Body text.\n"
)


def _write(folder, name, text):
    (folder / name).write_text(text, encoding="utf-8")


def test_loads_valid_document(tmp_path):
    _write(tmp_path, "good.md", VALID)
    documents = load_documents_from_folder(tmp_path)
    assert len(documents) == 1
    assert documents[0].title == "Test Doc"
    assert documents[0].owner_id == 100
    assert documents[0].body == "Body text."


def test_title_may_contain_a_colon(tmp_path):
    _write(tmp_path, "colon.md", VALID.replace("Test Doc", "Incident Report: Grease Fire"))
    assert load_documents_from_folder(tmp_path)[0].title == "Incident Report: Grease Fire"


def test_dashes_inside_the_body_are_preserved(tmp_path):
    _write(tmp_path, "dash.md", VALID + "\n---\nSecond section.\n")
    body = load_documents_from_folder(tmp_path)[0].body
    assert "---" in body and "Second section." in body


def test_skips_unsupported_file_type(tmp_path, caplog):
    _write(tmp_path, "notes.txt", "just text")
    with caplog.at_level(logging.WARNING):
        assert load_documents_from_folder(tmp_path) == []
    assert "unsupported file type" in caplog.text


def test_skips_document_missing_required_field(tmp_path, caplog):
    _write(tmp_path, "broken.md", VALID.replace("owner_id: 100\n", ""))
    with caplog.at_level(logging.WARNING):
        assert load_documents_from_folder(tmp_path) == []
    assert "owner_id" in caplog.text


def test_skips_invalid_category(tmp_path, caplog):
    _write(tmp_path, "bad.md", VALID.replace("SOP", "Poem"))
    with caplog.at_level(logging.WARNING):
        assert load_documents_from_folder(tmp_path) == []
    assert "unknown category" in caplog.text


def test_skips_invalid_date(tmp_path, caplog):
    _write(tmp_path, "bad.md", VALID.replace("2026-01-01", "01/01/2026"))
    with caplog.at_level(logging.WARNING):
        assert load_documents_from_folder(tmp_path) == []
    assert "last_reviewed_at" in caplog.text


def test_skips_missing_separator(tmp_path, caplog):
    _write(tmp_path, "nosep.md", "id: 1\ntitle: x\n")
    with caplog.at_level(logging.WARNING):
        assert load_documents_from_folder(tmp_path) == []
    assert "separator" in caplog.text


def test_skips_empty_body(tmp_path):
    _write(tmp_path, "empty.md", VALID.replace("Body text.\n", ""))
    assert load_documents_from_folder(tmp_path) == []


def test_skips_duplicate_ids(tmp_path):
    _write(tmp_path, "a.md", VALID)
    _write(tmp_path, "b.md", VALID.replace("Test Doc", "Copy"))
    documents = load_documents_from_folder(tmp_path)
    assert [d.title for d in documents] == ["Test Doc"]


def test_one_bad_file_does_not_block_the_rest(tmp_path):
    _write(tmp_path, "good.md", VALID)
    _write(tmp_path, "bad.txt", "nope")
    assert len(load_documents_from_folder(tmp_path)) == 1


def test_empty_folder_returns_empty_list(tmp_path):
    assert load_documents_from_folder(tmp_path) == []