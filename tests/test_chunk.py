from geek_crawler_rag.chunk import chunk_text


def test_chunk_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_short_text_single_piece():
    text = "Hello world. " * 20
    chunks = chunk_text(text, size_tokens=200, overlap_tokens=20)
    assert len(chunks) >= 1
    assert all(c.strip() for c in chunks)


def test_chunk_overlap_covers_long_doc():
    # Long enough to force multiple windows.
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(text, size_tokens=100, overlap_tokens=20)
    assert len(chunks) > 3
    # Reassembled token coverage: last chunk should include late words.
    assert "word1999" in chunks[-1]
    assert "word0" in chunks[0]


def test_chunk_rejects_bad_overlap():
    try:
        chunk_text("abc", size_tokens=10, overlap_tokens=10)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
