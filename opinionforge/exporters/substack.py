"""Substack-ready markdown exporter."""

from __future__ import annotations

import re

from opinionforge.exporters.base import BaseExporter
from opinionforge.models.piece import GeneratedPiece


class SubstackExporter(BaseExporter):
    """Export a GeneratedPiece as Substack-ready markdown.

    Produces clean ATX-style markdown with no raw HTML tags, properly
    separated paragraphs, and all required appendix sections.
    """

    def export(self, piece: GeneratedPiece) -> str:
        """Export *piece* as Substack-ready markdown.

        The output begins with a level-1 ATX heading containing the title,
        followed by an optional subtitle, the body, the sources appendix,
        the disclaimer, and (when present) the image prompt.

        Args:
            piece: The fully generated opinion piece to export.

        Returns:
            A markdown string with no raw HTML, no trailing whitespace per
            line, and all required sections present.
        """
        sections: list[str] = []

        # Title
        sections.append(f"# {piece.title}")

        # Optional subtitle
        if piece.subtitle:
            sections.append(piece.subtitle)

        # Body — strip any stray HTML that may have crept in
        body = _strip_html(piece.body.strip())
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

        raw = "\n\n".join(sections)
        return _clean_lines(raw)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Remove any raw HTML tags from *text*."""
    return _HTML_TAG_RE.sub("", text)


def _clean_lines(text: str) -> str:
    """Remove trailing whitespace from every line in *text*."""
    return "\n".join(line.rstrip() for line in text.splitlines())
