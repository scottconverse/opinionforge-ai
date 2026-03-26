"""Medium-ready markdown exporter with drop cap and pull quote support."""

from __future__ import annotations

import re

from opinionforge.exporters.base import BaseExporter
from opinionforge.models.piece import GeneratedPiece


class MediumExporter(BaseExporter):
    """Export a GeneratedPiece as Medium-ready markdown.

    Adds a DROP CAP marker before the opening paragraph, inserts pull quote
    markers for longer pieces, demotes heading levels to avoid conflicts with
    the piece title, and strips raw HTML.
    """

    _PULL_QUOTE_WORD_THRESHOLD = 300

    def export(self, piece: GeneratedPiece) -> str:
        """Export *piece* as Medium-ready markdown.

        Heading levels in the body are demoted by one level so that ``##``
        becomes ``###`` and ``#`` becomes ``##``, avoiding conflicts with the
        piece-title ``#`` heading.  A ``> DROP CAP`` marker is inserted before
        the first paragraph, and at least one blockquote pull quote is injected
        for pieces whose body exceeds the word threshold.

        Args:
            piece: The fully generated opinion piece to export.

        Returns:
            A markdown string suitable for pasting into Medium's editor,
            with no raw HTML tags.
        """
        sections: list[str] = []

        # Title
        sections.append(f"# {piece.title}")

        # Optional subtitle
        if piece.subtitle:
            sections.append(piece.subtitle)

        # Process body
        body = _strip_html(piece.body.strip())
        body = _demote_headings(body)
        body = _insert_drop_cap(body)

        word_count = len(piece.body.split())
        if word_count >= self._PULL_QUOTE_WORD_THRESHOLD:
            body = _insert_pull_quote(body)

        sections.append(body)

        # Sources appendix
        sources_text = self._format_sources(piece)
        if sources_text:
            sections.append(sources_text)

        # Disclaimer
        sections.append(self._format_disclaimer(piece))

        # Optional image prompt
        if piece.image_prompt:
            sections.append(f"**Header image prompt:** {piece.image_prompt}")

        return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_RE = re.compile(r"^(#{1,5})(\s)", re.MULTILINE)


def _strip_html(text: str) -> str:
    """Remove any raw HTML tags from *text*."""
    return _HTML_TAG_RE.sub("", text)


def _demote_headings(text: str) -> str:
    """Demote ATX headings in *text* by one level (e.g. ## -> ###).

    Headings at level 6 (``######``) are left unchanged to avoid exceeding
    the maximum heading depth.

    Args:
        text: Markdown text whose headings should be demoted.

    Returns:
        The text with each heading level incremented by one.
    """

    def _add_hash(m: re.Match[str]) -> str:
        hashes = m.group(1)
        space = m.group(2)
        if len(hashes) < 6:
            return f"#{hashes}{space}"
        return m.group(0)

    return _HEADING_RE.sub(_add_hash, text)


def _insert_drop_cap(text: str) -> str:
    """Insert a '> DROP CAP' marker before the first non-empty, non-heading paragraph.

    Args:
        text: The body markdown text.

    Returns:
        The text with the DROP CAP marker inserted before the first paragraph.
    """
    lines = text.splitlines()
    result: list[str] = []
    inserted = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if not inserted and line.strip() and not line.startswith("#") and not line.startswith(">"):
            result.append("> DROP CAP")
            inserted = True
        result.append(line)
        i += 1
    return "\n".join(result)


def _insert_pull_quote(text: str) -> str:
    """Insert a blockquote pull quote after the first full paragraph.

    Finds the first non-empty paragraph (a block of non-empty lines) and
    inserts a pull quote immediately after it.

    Args:
        text: The body markdown text (may already contain DROP CAP marker).

    Returns:
        The text with a pull quote blockquote inserted.
    """
    lines = text.splitlines()
    # Find a sentence to use as the pull quote: pick the first sentence from
    # a non-marker, non-heading line.
    pull_sentence = ""
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith(">"):
            # Grab the first sentence (up to first period/exclamation/question)
            match = re.search(r"[^.!?]+[.!?]", stripped)
            if match:
                pull_sentence = match.group(0).strip()
            else:
                pull_sentence = stripped[:120]
            break

    if not pull_sentence:
        return text

    # Locate end of first paragraph block (sequences of non-empty lines)
    result: list[str] = []
    in_first_para = False
    inserted = False
    i = 0
    while i < len(lines):
        line = lines[i]
        result.append(line)
        if not inserted:
            if line.strip():
                in_first_para = True
            elif in_first_para:
                # We just passed the first paragraph block
                result.append(f"> {pull_sentence}")
                result.append("")
                inserted = True
                in_first_para = False
        i += 1

    if not inserted:
        result.append("")
        result.append(f"> {pull_sentence}")

    return "\n".join(result)
