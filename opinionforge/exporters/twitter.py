"""Twitter/X thread exporter that splits content into numbered tweets."""

from __future__ import annotations

import re

from opinionforge.core.generator import MANDATORY_DISCLAIMER
from opinionforge.exporters.base import BaseExporter
from opinionforge.models.piece import GeneratedPiece

_MAX_TWEET_LEN = 280
_MIN_TWEETS = 5
_MAX_TWEETS = 15


class TwitterExporter(BaseExporter):
    """Export a GeneratedPiece as a numbered Twitter/X thread.

    The thread structure is:
    - Tweet 1: title + opening hook (first sentence of the body).
    - Tweets 2 … N-1: key argument paragraphs, one tweet (or split tweets)
      per paragraph.
    - Tweet N: disclaimer tweet (always the final tweet).

    No individual tweet (including its 'N/' prefix) exceeds 280 characters.
    Sentences are never cut mid-word; breaks happen at sentence or clause
    boundaries.
    """

    def export(self, piece: GeneratedPiece) -> str:
        """Export *piece* as a Twitter/X thread string.

        Args:
            piece: The fully generated opinion piece to export.

        Returns:
            A string where individual tweets are separated by blank lines,
            each prefixed with its thread number in 'N/' format.
        """
        body_paragraphs = _extract_paragraphs(piece.body)

        # Build candidate content tweets (without numbering)
        content_tweets: list[str] = []

        # Tweet 1: title + opening hook (joined with a single newline to avoid
        # splitting on blank lines when the thread is serialised later)
        first_sentence = _first_sentence(piece.body)
        opening = f"{piece.title} — {first_sentence}" if first_sentence else piece.title
        content_tweets.extend(_split_to_tweets(opening))

        # Middle tweets: argument paragraphs
        for para in body_paragraphs:
            if not para.strip():
                continue
            content_tweets.extend(_split_to_tweets(para))

        # Final tweet: disclaimer — always the last tweet in the thread
        disclaimer_tweet = _truncate_to_fit(MANDATORY_DISCLAIMER)
        content_tweets.append(disclaimer_tweet)

        # Enforce min/max bounds
        content_tweets = _enforce_tweet_limits(content_tweets)

        # Number and join
        numbered: list[str] = []
        for i, tweet_text in enumerate(content_tweets, start=1):
            prefix = f"{i}/"
            # Ensure prefix fits; it always will since prefix is very short
            numbered.append(f"{prefix} {tweet_text}")

        thread = "\n\n".join(numbered)

        # Optional image prompt appended as a note after the thread.
        # It is separated by a single newline so it does not become a spurious
        # numbered-tweet block when callers split on double newlines.
        if piece.image_prompt:
            thread += f"\n[Header image prompt: {piece.image_prompt}]"

        return thread


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def _extract_paragraphs(text: str) -> list[str]:
    """Split *text* into paragraphs separated by blank lines."""
    raw_paras = re.split(r"\n{2,}", text.strip())
    result: list[str] = []
    for para in raw_paras:
        cleaned = re.sub(r"^#+\s*", "", para.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"^>\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\n", " ", cleaned).strip()
        if cleaned:
            result.append(cleaned)
    return result


def _first_sentence(text: str) -> str:
    """Extract the first sentence from *text*."""
    plain = re.sub(r"^#+\s*", "", text.strip(), flags=re.MULTILINE)
    plain = re.sub(r"^>\s*", "", plain, flags=re.MULTILINE)
    plain = plain.replace("\n", " ").strip()
    match = re.search(r"[^.!?]+[.!?]", plain)
    if match:
        return match.group(0).strip()
    # Fallback: up to 200 chars
    return plain[:200].rstrip()


def _split_to_tweets(text: str) -> list[str]:
    """Split *text* into chunks that each fit within the tweet character budget.

    The budget for text content is ``_MAX_TWEET_LEN - 3`` to leave room for
    the number prefix (e.g. '10/ ').  Breaks are made at sentence boundaries
    first; if a single sentence exceeds the budget it is split at the last
    whitespace within the budget.

    Args:
        text: Plain text to split.

    Returns:
        A list of string chunks, each short enough to fit in a tweet after
        the 'N/ ' prefix is prepended.
    """
    # Reserve space for the widest possible prefix "15/ " = 4 chars
    budget = _MAX_TWEET_LEN - 4

    if len(text) <= budget:
        return [text]

    # Split on sentence boundaries
    sentences = _SENTENCE_END_RE.split(text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= budget:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If the sentence itself is too long, split at word boundary
            if len(sentence) > budget:
                chunks.extend(_hard_split(sentence, budget))
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks if chunks else [text[:budget]]


def _hard_split(text: str, budget: int) -> list[str]:
    """Split *text* at word boundaries to fit within *budget* characters.

    Args:
        text: Text that exceeds *budget*.
        budget: Maximum characters per chunk.

    Returns:
        A list of chunks each no longer than *budget* characters.
    """
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= budget:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks


def _truncate_to_fit(text: str, budget: int = _MAX_TWEET_LEN - 4) -> str:
    """Truncate *text* to *budget* characters, breaking on a word boundary.

    Args:
        text: Text to truncate.
        budget: Maximum character count.

    Returns:
        The text, truncated at a word boundary if necessary.
    """
    if len(text) <= budget:
        return text
    truncated = text[:budget]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip(".,;: ") + "…"


def _enforce_tweet_limits(tweets: list[str]) -> list[str]:
    """Enforce the 5-tweet minimum and 15-tweet maximum on *tweets*.

    If there are too many tweets, the middle content tweets (excluding the
    first and final disclaimer) are merged or summarised until the count
    fits.  If there are too few, the opening tweet is split across
    additional cards.

    Args:
        tweets: List of tweet text strings (unnumbered).

    Returns:
        A list with length between ``_MIN_TWEETS`` and ``_MAX_TWEETS``.
    """
    if _MIN_TWEETS <= len(tweets) <= _MAX_TWEETS:
        return tweets

    # Enforce max: collapse middle tweets by dropping them until we fit
    if len(tweets) > _MAX_TWEETS:
        first = tweets[0]
        disclaimer = tweets[-1]
        middle = list(tweets[1:-1])
        budget = _MAX_TWEET_LEN - 4
        # Try to merge adjacent pairs first; when merge exceeds budget, drop the item
        i = 0
        while len(middle) + 2 > _MAX_TWEETS and len(middle) > 1:
            if i + 1 < len(middle):
                merged = f"{middle[i]} {middle[i + 1]}"
                if len(merged) <= budget:
                    middle[i: i + 2] = [merged]
                else:
                    # Drop the shorter of the two
                    del middle[i + 1]
            else:
                i = 0  # restart from beginning
            if i >= len(middle) - 1:
                i = 0
            else:
                i += 1
        tweets = [first] + middle + [disclaimer]

    # Enforce min: pad with a filler if somehow under 5
    while len(tweets) < _MIN_TWEETS:
        tweets.insert(-1, "There's more to this story than the headlines suggest.")

    return tweets
