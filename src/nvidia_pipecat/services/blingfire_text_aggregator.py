#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Blingfire-based text aggregator for sentence detection."""

import re

import blingfire as bf
from loguru import logger
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator


def normalize(s):
    """Normalize text by collapsing multiple whitespace/newlines into a single space.

    Args:
        s (str): The input string to normalize.

    Returns:
        str: The normalized string with consistent whitespace.
    """
    # Collapse multiple whitespace/newlines into a single space
    return re.sub(r"\s+", " ", s.strip())


class BlingfireTextAggregator(BaseTextAggregator):
    """This is a text aggregator that uses blingfire for sentence detection.

    It aggregates text until a complete sentence is detected using blingfire's
    sentence segmentation capabilities.
    """

    def __init__(self):
        """Initialize the BlingfireTextAggregator with an empty text buffer."""
        self._text = ""
        logger.trace("BlingfireTextAggregator: Initialized new instance")

    @property
    def text(self) -> str:
        """Return the currently aggregated text.

        Returns:
            str: The text currently being aggregated.
        """
        return self._text

    async def aggregate(self, text: str) -> str | None:
        """Aggregate text and return a complete sentence when detected.

        Args:
            text (str): The text to be aggregated.

        Returns:
            str | None: A complete sentence if one is detected, otherwise None.
        """
        result: str | None = None

        self._text += text
        logger.trace(f"BlingfireTextAggregator: Aggregating text: '{self._text}'")

        # Normalize the accumulated text for consistent processing
        normalized_text = normalize(self._text)

        # Use blingfire to split normalized text into sentences
        sentences_text = bf.text_to_sentences(normalized_text)
        sentences = [s.strip() for s in sentences_text.split("\n") if s.strip()]
        # If we have multiple sentences, return the first complete one
        if len(sentences) > 1:
            result = sentences[0]

            # Find the position of the first sentence in the normalized text
            first_sentence = sentences[0]
            sentence_pos = normalized_text.find(first_sentence)
            if sentence_pos != -1:
                # Calculate how much of the original text corresponds to the first sentence
                # by finding the position in the original text that matches the normalized position
                remaining_normalized = normalized_text[sentence_pos + len(first_sentence) :]
                # Find where the remaining normalized text starts in the original text
                # by looking for the first occurrence of the remaining normalized content
                if remaining_normalized:
                    self._text = self._text[len(first_sentence) :]
                else:
                    # No remaining text, clear the buffer
                    self._text = ""

        return result

    async def handle_interruption(self):
        """Handle interruption by clearing the aggregated text buffer."""
        logger.trace("BlingfireTextAggregator: Handling interruption")
        self._text = ""

    async def reset(self):
        """Reset the aggregated text buffer to empty."""
        logger.trace("BlingfireTextAggregator: Resetting text buffer")
        self._text = ""
