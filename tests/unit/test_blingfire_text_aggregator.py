# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for BlingfireTextAggregator."""

import pytest

from nvidia_pipecat.services.blingfire_text_aggregator import BlingfireTextAggregator


class TestBlingfireTextAggregator:
    """Test suite for BlingfireTextAggregator."""

    def test_initialization(self):
        """Test that the aggregator initializes with empty text buffer."""
        aggregator = BlingfireTextAggregator()
        assert aggregator.text == ""

    @pytest.mark.asyncio()
    async def test_single_word_no_sentence(self):
        """Test that single words without sentence endings don't return sentences."""
        aggregator = BlingfireTextAggregator()
        result = await aggregator.aggregate("hello")
        assert result is None
        assert aggregator.text == "hello"

    @pytest.mark.asyncio()
    async def test_incomplete_sentence(self):
        """Test that incomplete sentences are buffered but not returned."""
        aggregator = BlingfireTextAggregator()
        result = await aggregator.aggregate("Hello there")
        assert result is None
        assert aggregator.text == "Hello there"

    @pytest.mark.asyncio()
    async def test_single_complete_sentence(self):
        """Test that a single complete sentence is detected and returned."""
        aggregator = BlingfireTextAggregator()
        result = await aggregator.aggregate("Hello world.")
        assert result is None  # Single sentence won't trigger return
        assert aggregator.text == "Hello world."

    @pytest.mark.asyncio()
    async def test_multiple_sentences_detection(self):
        """Test that multiple sentences trigger return of the first complete sentence."""
        aggregator = BlingfireTextAggregator()
        result = await aggregator.aggregate("Hello world. How are you?")
        assert result == "Hello world."
        assert "How are you?" in aggregator.text
        assert "Hello world." not in aggregator.text

    @pytest.mark.asyncio()
    async def test_incremental_sentence_building(self):
        """Test building a sentence incrementally."""
        aggregator = BlingfireTextAggregator()

        # Add text piece by piece
        result = await aggregator.aggregate("Hello")
        assert result is None
        assert aggregator.text == "Hello"

        result = await aggregator.aggregate(" world")
        assert result is None
        assert aggregator.text == "Hello world"

        result = await aggregator.aggregate(".")
        assert result is None
        assert aggregator.text == "Hello world."

    @pytest.mark.asyncio()
    async def test_incremental_multiple_sentences(self):
        """Test building multiple sentences incrementally."""
        aggregator = BlingfireTextAggregator()

        # Build first sentence
        result = await aggregator.aggregate("Hello world.")
        assert result is None
        assert aggregator.text == "Hello world."

        # Add second sentence - this should trigger return
        result = await aggregator.aggregate(" How are you?")
        assert result == "Hello world."
        assert "How are you?" in aggregator.text
        assert "Hello world." not in aggregator.text

    @pytest.mark.asyncio()
    async def test_empty_string_input(self):
        """Test handling of empty string input."""
        aggregator = BlingfireTextAggregator()
        result = await aggregator.aggregate("")
        assert result is None
        assert aggregator.text == ""

    @pytest.mark.asyncio()
    async def test_whitespace_handling(self):
        """Test proper handling of whitespace in text."""
        aggregator = BlingfireTextAggregator()
        result = await aggregator.aggregate("  Hello world.  ")
        assert result is None
        assert aggregator.text == "  Hello world.  "

    @pytest.mark.asyncio()
    async def test_multiple_sentences_in_single_call(self):
        """Test processing multiple sentences passed in a single aggregate call."""
        aggregator = BlingfireTextAggregator()
        result = await aggregator.aggregate("First sentence. Second sentence. Third sentence.")
        assert result == "First sentence."
        # Remaining text should contain the other sentences
        remaining_text = aggregator.text
        assert "Second sentence. Third sentence." in remaining_text

    @pytest.mark.asyncio()
    async def test_sentence_with_special_punctuation(self):
        """Test sentences with different punctuation marks."""
        aggregator = BlingfireTextAggregator()

        # Test exclamation mark
        result = await aggregator.aggregate("Hello world! How are you?")
        assert result == "Hello world!"
        assert "How are you?" in aggregator.text

        await aggregator.reset()

        # Test question mark
        result = await aggregator.aggregate("How are you? I'm fine.")
        assert result == "How are you?"
        assert "I'm fine." in aggregator.text

    @pytest.mark.asyncio()
    async def test_handle_interruption(self):
        """Test that handle_interruption clears the text buffer."""
        aggregator = BlingfireTextAggregator()
        await aggregator.aggregate("Hello world")
        assert aggregator.text == "Hello world"

        await aggregator.handle_interruption()
        assert aggregator.text == ""

    @pytest.mark.asyncio()
    async def test_reset(self):
        """Test that reset clears the text buffer."""
        aggregator = BlingfireTextAggregator()
        await aggregator.aggregate("Hello world")
        assert aggregator.text == "Hello world"

        await aggregator.reset()
        assert aggregator.text == ""

    @pytest.mark.asyncio()
    async def test_reset_after_sentence_detection(self):
        """Test reset functionality after sentence detection."""
        aggregator = BlingfireTextAggregator()
        result = await aggregator.aggregate("First sentence. Second sentence.")
        assert result == "First sentence."
        assert "Second sentence." in aggregator.text

        await aggregator.reset()
        assert aggregator.text == ""

    @pytest.mark.asyncio()
    async def test_consecutive_sentence_processing(self):
        """Test processing consecutive sentences through multiple aggregate calls."""
        aggregator = BlingfireTextAggregator()

        # First pair of sentences
        result = await aggregator.aggregate("First sentence. Second sentence.")
        assert result == "First sentence."

        # Add third sentence - should trigger return of second
        result = await aggregator.aggregate(" Third sentence.")
        assert result == "Second sentence."
        assert "Third sentence." in aggregator.text

    @pytest.mark.asyncio()
    async def test_long_sentence_handling(self):
        """Test handling of longer sentences."""
        aggregator = BlingfireTextAggregator()
        long_sentence = (
            "This is a very long sentence with many words that goes on and on "
            "and should still be handled correctly by the aggregator."
        )
        result = await aggregator.aggregate(long_sentence)
        assert result is None
        assert aggregator.text == long_sentence

    @pytest.mark.asyncio()
    async def test_sentence_boundaries_with_abbreviations(self):
        """Test sentence detection with abbreviations that contain periods."""
        aggregator = BlingfireTextAggregator()
        result = await aggregator.aggregate("Dr. Smith went to the U.S.A. He had a great time.")
        assert result == "Dr. Smith went to the U.S.A."
        assert "He had a great time." in aggregator.text

    @pytest.mark.asyncio()
    async def test_newline_handling(self):
        """Test handling of text with newlines."""
        aggregator = BlingfireTextAggregator()
        result = await aggregator.aggregate("First line.\nSecond line.")
        assert result == "First line."
        assert "Second line." in aggregator.text

    @pytest.mark.asyncio()
    async def test_mixed_sentence_endings(self):
        """Test text with mixed sentence ending punctuation."""
        aggregator = BlingfireTextAggregator()
        result = await aggregator.aggregate("What time is it? It's 3 PM! That's great.")
        assert result == "What time is it?"
        remaining = aggregator.text
        assert "It's 3 PM!" in remaining
        assert "That's great." in remaining

    @pytest.mark.asyncio()
    async def test_text_property_consistency(self):
        """Test that the text property always returns the current buffer state."""
        aggregator = BlingfireTextAggregator()

        # Initially empty
        assert aggregator.text == ""

        # After adding incomplete sentence
        await aggregator.aggregate("Hello")
        assert aggregator.text == "Hello"

        # After adding more text
        await aggregator.aggregate(" world")
        assert aggregator.text == "Hello world"

        # After completing sentence (but no return yet)
        await aggregator.aggregate(".")
        assert aggregator.text == "Hello world."

        # After triggering sentence return
        await aggregator.aggregate(" Next sentence.")
        assert "Next sentence." in aggregator.text
        assert "Hello world." not in aggregator.text

    @pytest.mark.asyncio()
    async def test_empty_sentences_filtering(self):
        """Test that empty sentences are properly filtered out."""
        aggregator = BlingfireTextAggregator()
        # Text with multiple periods that might create empty sentences
        result = await aggregator.aggregate("Hello... world.")
        assert result is None
        assert aggregator.text == "Hello... world."
