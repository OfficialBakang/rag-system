from dataclasses import replace

import pytest
import tiktoken

from core.chunker import Chunker


def test_short_empty_and_multiple_documents(document):
    chunker = Chunker()
    other = replace(document, doc_id="other")
    chunks = chunker.chunk_many([document, replace(document, content="  "), other])
    assert [c.doc_id for c in chunks] == ["doc", "other"]
    assert [c.text for c in chunks] == [document.content] * 2


def test_bounded_windows_overlap_and_metadata(document):
    doc = replace(document, content="one two three four five six seven eight nine")
    chunker = Chunker(4, 1)
    chunks = chunker.chunk(doc)
    assert [c.text for c in chunks] == [
        "one two three four",
        "four five six seven",
        "seven eight nine",
    ]
    assert chunks == chunker.chunk(doc)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for index, c in enumerate(chunks):
        assert c.index == c.metadata["chunk_index"] == index
        assert c.metadata == {
            **doc.metadata,
            "chunk_index": index,
            "chunk_total": len(chunks),
        }
        assert (
            c.token_count
            == len(tiktoken.get_encoding("cl100k_base").encode(c.text))
            <= 4
        )


def test_full_segment_after_overlap_makes_progress(document):
    chunks = Chunker(3, 2).chunk(
        replace(document, content="one two three\n\nfour five six")
    )
    assert [c.text for c in chunks] == ["one two three", "four five six"]


def test_unbroken_word_preserves_characters(document):
    word = "Ab9" * 25
    chunks = Chunker(4, 0).chunk(replace(document, content=word))
    assert "".join(c.text for c in chunks) == word
    assert all(c.token_count <= 4 for c in chunks)


@pytest.mark.parametrize("overlap", [-1, 4, 5])
def test_invalid_overlap(overlap):
    with pytest.raises(ValueError):
        Chunker(4, overlap)


@pytest.mark.parametrize(("char", "budget", "tokens"), [("🙂", 1, 2), ("🦊", 2, 3)])
def test_unsplittable_unicode_character_reports_required_budget(
    document, char, budget, tokens
):
    doc = replace(document, content="abc" + char)
    message = rf"U\+{ord(char):04X} requires {tokens} tokens.*chunk_size={budget}"
    with pytest.raises(ValueError, match=message):
        Chunker(budget, 0).chunk(doc)


@pytest.mark.parametrize(
    ("text", "budget"),
    [
        ("é界" * 32, 1),
        ("漢🙂" * 128, 2),
        ("🦊e\u0301" * 64, 3),
    ],
)
def test_unbroken_unicode_round_trips_at_zero_overlap(document, text, budget):
    chunks = Chunker(budget, 0).chunk(replace(document, content=text))
    assert len(chunks) > 1
    assert "".join(chunk.text for chunk in chunks) == text
    encoding = tiktoken.get_encoding("cl100k_base")
    for chunk in chunks:
        assert 0 < chunk.token_count == len(encoding.encode(chunk.text)) <= budget
