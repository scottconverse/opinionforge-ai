"""Abstract base class for all platform exporters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from opinionforge.core.generator import MANDATORY_DISCLAIMER
from opinionforge.models.piece import GeneratedPiece


class BaseExporter(ABC):
    """Abstract base class defining the interface all platform exporters must implement.

    Subclasses must implement the ``export`` method. Helper methods for
    formatting the sources appendix and the mandatory disclaimer are
    provided here so every exporter produces consistent output.
    """

    @abstractmethod
    def export(self, piece: GeneratedPiece) -> str:
        """Export *piece* to a platform-specific string.

        Args:
            piece: The fully generated opinion piece to export.

        Returns:
            A non-empty string containing the formatted piece ready for the
            target platform. The mandatory disclaimer is always included.
        """
        ...

    def _format_sources(self, piece: GeneratedPiece) -> str:
        """Return the '## Sources & Claims' appendix in PRD-specified format.

        Each citation is rendered as a numbered list item with the claim text
        in quotes, an em-dash, a markdown link to the source, and an
        'accessed YYYY-MM-DD' date.

        Args:
            piece: The generated piece whose ``sources`` list is used.

        Returns:
            A markdown string beginning with '## Sources & Claims\\n\\n'
            followed by numbered citations, or an empty string when the
            sources list is empty.
        """
        if not piece.sources:
            return ""

        lines: list[str] = ["## Sources & Claims", ""]
        for i, citation in enumerate(piece.sources, start=1):
            date_str = citation.accessed_at.strftime("%Y-%m-%d")
            lines.append(
                f'{i}. "{citation.claim}" \u2014 '
                f"[{citation.source_name}]({citation.source_url}), "
                f"accessed {date_str}"
            )
        return "\n".join(lines)

    def _format_disclaimer(self, piece: GeneratedPiece) -> str:  # noqa: ARG002
        """Return the fixed mandatory disclaimer string.

        The disclaimer is always the fixed constant defined in
        ``opinionforge.core.generator.MANDATORY_DISCLAIMER``. It is never
        constructed dynamically from piece data.

        Args:
            piece: Accepted for interface compatibility but not used. The
                disclaimer is fixed and not derived from piece content.

        Returns:
            The fixed mandatory disclaimer string:
            'This piece was generated with AI-assisted rhetorical controls.
            It is original content and is not written by, endorsed by, or
            affiliated with any real person.'
        """
        return MANDATORY_DISCLAIMER
