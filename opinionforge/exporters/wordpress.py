"""WordPress Gutenberg block-editor HTML exporter."""

from __future__ import annotations

import html
import re

from opinionforge.exporters.base import BaseExporter
from opinionforge.models.piece import GeneratedPiece


class WordPressExporter(BaseExporter):
    """Export a GeneratedPiece as WordPress Gutenberg block-editor markup.

    Produces Gutenberg block comments (``<!-- wp:... -->`` / ``<!-- /wp:... -->``)
    wrapping semantic HTML elements.  Does not target the Classic Editor.
    """

    def export(self, piece: GeneratedPiece) -> str:
        """Export *piece* as WordPress Gutenberg block markup.

        The output includes:
        - A featured image placeholder comment.
        - An SEO meta description comment (first 160 chars of the body).
        - An optional image prompt comment.
        - The piece title in a ``wp:heading`` level-1 block.
        - An optional subtitle paragraph.
        - Body paragraphs each wrapped in ``wp:paragraph`` blocks.
        - Sources appendix in a ``wp:paragraph`` block.
        - Disclaimer in a ``<p class='opinionforge-disclaimer'>`` paragraph.

        Args:
            piece: The fully generated opinion piece to export.

        Returns:
            A string of Gutenberg block markup valid for WordPress 5.0+.
        """
        blocks: list[str] = []

        # Featured image placeholder
        blocks.append("<!-- FEATURED IMAGE PLACEHOLDER -->")

        # SEO meta description (first 160 chars of plain body text)
        plain_body = _strip_markdown(piece.body)
        seo_snippet = plain_body[:160].rstrip()
        # Remove any '-->' sequences that would break the HTML comment syntax
        seo_snippet = seo_snippet.replace("-->", "- ->").replace("--", "\u2014")
        blocks.append(f"<!-- SEO META: {seo_snippet} -->")

        # Optional image prompt comment
        if piece.image_prompt:
            blocks.append(f"<!-- IMAGE PROMPT: {piece.image_prompt} -->")

        # Title heading block
        title_escaped = html.escape(piece.title)
        blocks.append(
            "<!-- wp:heading {\"level\":1} -->\n"
            f"<h1>{title_escaped}</h1>\n"
            "<!-- /wp:heading -->"
        )

        # Optional subtitle
        if piece.subtitle:
            subtitle_escaped = html.escape(piece.subtitle)
            blocks.append(
                "<!-- wp:paragraph -->\n"
                f"<p>{subtitle_escaped}</p>\n"
                "<!-- /wp:paragraph -->"
            )

        # Body paragraphs
        paragraphs = _split_into_paragraphs(piece.body)
        for para in paragraphs:
            if not para.strip():
                continue
            para_html = _markdown_to_simple_html(para)
            blocks.append(
                "<!-- wp:paragraph -->\n"
                f"<p>{para_html}</p>\n"
                "<!-- /wp:paragraph -->"
            )

        # Sources appendix
        sources_md = self._format_sources(piece)
        if sources_md:
            sources_html = _markdown_to_simple_html(sources_md)
            blocks.append(
                "<!-- wp:paragraph -->\n"
                f"<p>{sources_html}</p>\n"
                "<!-- /wp:paragraph -->"
            )

        # Disclaimer
        disclaimer_escaped = html.escape(self._format_disclaimer(piece))
        blocks.append(
            "<!-- wp:paragraph -->\n"
            f"<p class='opinionforge-disclaimer'>{disclaimer_escaped}</p>\n"
            "<!-- /wp:paragraph -->"
        )

        return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MARKDOWN_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_MARKDOWN_ITALIC_RE = re.compile(r"\*([^*]+)\*")
_MARKDOWN_BLOCKQUOTE_RE = re.compile(r"^>\s*", re.MULTILINE)


def _strip_markdown(text: str) -> str:
    """Return a single-line plain-text version of *text* by removing markdown.

    Used for the SEO meta snippet; newlines are collapsed to spaces so the
    result fits on one HTML comment line.

    Args:
        text: Markdown-formatted text.

    Returns:
        A single-line plain text string with headings, links, and emphasis
        markers removed, suitable for embedding in an HTML comment.
    """
    text = _MARKDOWN_HEADING_RE.sub("", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _MARKDOWN_BOLD_RE.sub(r"\1", text)
    text = _MARKDOWN_ITALIC_RE.sub(r"\1", text)
    text = _MARKDOWN_BLOCKQUOTE_RE.sub("", text)
    # Collapse newlines and excess whitespace into single spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_into_paragraphs(text: str) -> list[str]:
    """Split *text* into paragraph blocks separated by blank lines.

    Args:
        text: Multi-paragraph markdown text.

    Returns:
        A list of paragraph strings (may include heading lines).
    """
    return [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]


def _markdown_to_simple_html(text: str) -> str:
    """Convert a subset of markdown inline markup to HTML within a single block.

    Handles: bold, italic, links, and heading lines (converted to ``<strong>``
    text since headings are rendered as separate wp:heading blocks in full
    documents, but for sources/single-paragraph contexts we keep it simple).

    Args:
        text: A markdown paragraph or block of text.

    Returns:
        HTML string with inline markup converted.
    """
    # Convert links first (before escaping)
    links: dict[str, tuple[str, str]] = {}
    placeholder_map: dict[str, str] = {}

    def _stash_link(m: re.Match[str]) -> str:
        key = f"\x00LINK{len(links)}\x00"
        links[key] = (m.group(1), m.group(2))
        return key

    text = _MARKDOWN_LINK_RE.sub(_stash_link, text)

    # Escape remaining HTML special chars
    text = html.escape(text)

    # Restore links as <a href="...">
    for key, (label, url) in links.items():
        label_escaped = html.escape(label)
        url_escaped = html.escape(url)
        text = text.replace(html.escape(key), f'<a href="{url_escaped}">{label_escaped}</a>')

    # Bold and italic (operate on escaped text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)

    # Strip leading markdown heading markers (already escaped, so just #s)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

    # Blockquote markers
    text = re.sub(r"^&gt;\s*", "", text, flags=re.MULTILINE)

    # Newlines within a paragraph become <br>
    text = text.replace("\n", "<br>")

    return text
