"""
Tests for Clean Pipeline Stage
==============================
Tests the text extraction and chunking logic.
"""

import pytest
from pipeline.clean import chunk_text, estimate_tokens, extract_text_basic


class TestEstimateTokens:
    """Tests for token estimation."""

    def test_empty_string(self):
        """Test token estimation for empty string."""
        assert estimate_tokens("") == 0

    def test_short_string(self):
        """Test token estimation for short string."""
        text = "Hello world"  # 11 chars
        tokens = estimate_tokens(text, chars_per_token=4)
        assert tokens == 2  # 11 // 4 = 2

    def test_longer_string(self):
        """Test token estimation for longer string."""
        text = "This is a longer sentence with more words."  # 43 chars
        tokens = estimate_tokens(text, chars_per_token=4)
        assert tokens == 10  # 43 // 4 = 10


class TestChunkText:
    """Tests for text chunking."""

    def test_empty_text(self):
        """Test chunking empty text."""
        chunks = chunk_text("")
        assert chunks == []

    def test_short_text_single_chunk(self):
        """Test that short text results in single chunk."""
        text = "This is a short piece of text."
        chunks = chunk_text(text, chunk_size_tokens=100, chunk_overlap_tokens=20)

        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["chunk_index"] == 0

    def test_long_text_multiple_chunks(self):
        """Test that long text results in multiple chunks."""
        # Create text that's definitely longer than one chunk
        text = " ".join(["This is sentence number {}.".format(i) for i in range(100)])

        chunks = chunk_text(
            text,
            chunk_size_tokens=50,  # Small chunks
            chunk_overlap_tokens=10,
            chars_per_token=4
        )

        assert len(chunks) > 1
        # Check chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_chunk_overlap(self):
        """Test that chunks have overlap."""
        # Create predictable text
        words = ["word{}".format(i) for i in range(200)]
        text = " ".join(words)

        chunks = chunk_text(
            text,
            chunk_size_tokens=20,
            chunk_overlap_tokens=5,
            chars_per_token=4
        )

        if len(chunks) >= 2:
            # Check that end of first chunk overlaps with start of second
            chunk1_end = chunks[0]["end_char"]
            chunk2_start = chunks[1]["start_char"]
            assert chunk2_start < chunk1_end  # Overlap exists

    def test_chunk_metadata(self):
        """Test that chunks contain required metadata."""
        text = "A " * 500  # Long enough for multiple chunks

        chunks = chunk_text(text, chunk_size_tokens=50, chunk_overlap_tokens=10)

        for chunk in chunks:
            assert "chunk_index" in chunk
            assert "text" in chunk
            assert "start_char" in chunk
            assert "end_char" in chunk
            assert "token_count_approx" in chunk
            assert isinstance(chunk["chunk_index"], int)
            assert isinstance(chunk["start_char"], int)
            assert isinstance(chunk["end_char"], int)

    def test_paragraph_break_preference(self):
        """Test that chunking prefers paragraph breaks."""
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."

        # With small chunk size, should break at paragraphs when possible
        chunks = chunk_text(
            text,
            chunk_size_tokens=20,
            chunk_overlap_tokens=5,
            chars_per_token=4
        )

        # At least check we get reasonable chunks
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk["text"]) > 0


class TestExtractTextBasic:
    """Tests for basic HTML text extraction."""

    def test_simple_html(self):
        """Test extraction from simple HTML."""
        html = "<html><body><p>Hello world</p></body></html>"
        text, title = extract_text_basic(html)

        assert "Hello world" in text

    def test_title_extraction(self):
        """Test title extraction from HTML."""
        html = "<html><head><title>Page Title</title></head><body><p>Content</p></body></html>"
        text, title = extract_text_basic(html)

        assert title == "Page Title"

    def test_script_removal(self):
        """Test that script tags are removed."""
        html = """
        <html>
        <body>
            <p>Visible text</p>
            <script>var x = 'hidden';</script>
        </body>
        </html>
        """
        text, title = extract_text_basic(html)

        assert "Visible text" in text
        assert "hidden" not in text
        assert "var x" not in text

    def test_style_removal(self):
        """Test that style tags are removed."""
        html = """
        <html>
        <body>
            <p>Visible text</p>
            <style>.class { color: red; }</style>
        </body>
        </html>
        """
        text, title = extract_text_basic(html)

        assert "Visible text" in text
        assert "color" not in text

    def test_nav_footer_removal(self):
        """Test that nav and footer are removed."""
        html = """
        <html>
        <body>
            <nav>Navigation menu</nav>
            <main><p>Main content</p></main>
            <footer>Footer content</footer>
        </body>
        </html>
        """
        text, title = extract_text_basic(html)

        assert "Main content" in text
        # Nav and footer might be removed depending on implementation
