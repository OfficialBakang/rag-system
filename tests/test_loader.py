import pytest

from core.loader import DocumentLoader


def test_text_metadata_and_stable_id(document_file):
    loader = DocumentLoader()
    doc = loader.load_file(document_file)
    assert doc.content == "Local passage"
    assert doc.doc_id == loader.load_file(document_file).doc_id
    assert doc.metadata | {"load_ms": 0} == {
        "filename": "report.txt",
        "source": str(document_file),
        "format": "txt",
        "char_count": 13,
        "title": "report",
        "load_ms": 0,
    }


@pytest.mark.parametrize(
    ("suffix", "raw", "content", "title"),
    [
        ("md", "---\nsecret: true\n---\n# Text", "# Text", "page"),
        (
            "html",
            "<title>A</title><style>x</style><script>y</script><p>B</p>",
            "A\nB",
            "A",
        ),
    ],
)
def test_extract_formats(tmp_path, suffix, raw, content, title):
    path = tmp_path / f"page.{suffix}"
    path.write_text(raw)
    doc = DocumentLoader().load_file(path)
    assert (doc.content, doc.metadata["title"]) == (content, title)


def test_pdf(tmp_path, mocker):
    path = tmp_path / "test.pdf"
    path.write_bytes(b"%PDF-1.7")
    reader = mocker.patch("pypdf.PdfReader").return_value
    reader.is_encrypted = False
    reader.pages = [mocker.Mock(), mocker.Mock()]
    for page, text in zip(reader.pages, ("First", "Second"), strict=True):
        page.extract_text.return_value = text
    reader.metadata = {"/Title": "Report"}
    doc = DocumentLoader().load_file(path)
    assert (doc.content, doc.metadata["page_count"], doc.metadata["title"]) == (
        "First\n\nSecond",
        2,
        "Report",
    )


def test_directory_sorting_filtering_and_errors(tmp_path, document_file):
    loader = DocumentLoader()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.md").write_text("B")
    other = tmp_path / "skip.exe"
    other.touch()
    assert [d.content for d in loader.load_directory(tmp_path)] == [
        "B",
        "Local passage",
    ]
    assert [d.source for d in loader.load_directory(tmp_path, False)] == [
        str(document_file)
    ]
    with pytest.raises(ValueError, match="Unsupported format"):
        loader.load_file(other)
    with pytest.raises(FileNotFoundError):
        loader.load_file(tmp_path / "missing.txt")
    with pytest.raises(NotADirectoryError):
        loader.load_directory(document_file)
