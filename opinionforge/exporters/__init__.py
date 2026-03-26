"""Export engine for OpinionForge.

Re-exports the four platform exporter classes and provides the ``export()``
dispatcher function for use by the CLI and other callers.
"""

from __future__ import annotations

from opinionforge.exporters.medium import MediumExporter
from opinionforge.exporters.substack import SubstackExporter
from opinionforge.exporters.twitter import TwitterExporter
from opinionforge.exporters.wordpress import WordPressExporter
from opinionforge.models.piece import GeneratedPiece

__all__ = [
    "SubstackExporter",
    "MediumExporter",
    "WordPressExporter",
    "TwitterExporter",
    "export",
]

_EXPORTERS: dict[str, type[SubstackExporter | MediumExporter | WordPressExporter | TwitterExporter]] = {
    "substack": SubstackExporter,
    "medium": MediumExporter,
    "wordpress": WordPressExporter,
    "twitter": TwitterExporter,
}


def export(piece: GeneratedPiece, format: str) -> str:  # noqa: A002
    """Dispatch *piece* to the appropriate exporter for *format*.

    Args:
        piece: The fully generated opinion piece to export.
        format: Lowercase platform name — one of ``'substack'``, ``'medium'``,
            ``'wordpress'``, or ``'twitter'``.

    Returns:
        A non-empty formatted string ready for the target platform.

    Raises:
        ValueError: If *format* is not one of the supported platform names.
    """
    exporter_class = _EXPORTERS.get(format.lower())
    if exporter_class is None:
        supported = ", ".join(sorted(_EXPORTERS.keys()))
        raise ValueError(
            f"Unknown export format '{format}'. Supported formats: {supported}."
        )
    return exporter_class().export(piece)
