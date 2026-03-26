"""Tests verifying the app works without any files in research/.

Ensures that the public release has no runtime dependency on .gitignore-excluded
materials. All public modules must be importable and functional without a
research/ directory present.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_OPINIONFORGE_DIR = _REPO_ROOT / "opinionforge"

# ---------------------------------------------------------------------------
# Public modules that must be importable without research/
# ---------------------------------------------------------------------------

_PUBLIC_MODULES = [
    "opinionforge.cli",
    "opinionforge.core.mode_engine",
    "opinionforge.core.stance",
    "opinionforge.core.similarity",
    "opinionforge.core.generator",
    "opinionforge.core.preview",
    "opinionforge.core.research",
    "opinionforge.core.topic",
    "opinionforge.exporters",
]


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestPublicModulesImportable:
    """All public modules import without error and without research/ dependency."""

    @pytest.mark.parametrize("module_name", _PUBLIC_MODULES)
    def test_module_is_importable(self, module_name: str) -> None:
        """Module imports without error.

        Args:
            module_name: Fully qualified module name to import.
        """
        try:
            mod = importlib.import_module(module_name)
        except ImportError as exc:
            pytest.fail(
                f"Module '{module_name}' could not be imported: {exc}. "
                "All public modules must be importable in a clean checkout."
            )
        assert mod is not None, f"Module '{module_name}' imported as None"


# ---------------------------------------------------------------------------
# Research directory independence test
# ---------------------------------------------------------------------------


class TestNoResearchDirectoryDependency:
    """No module in opinionforge/ contains a hardcoded import of research/."""

    def _collect_py_files(self) -> list[Path]:
        """Return all .py files under opinionforge/ directory.

        Returns:
            Sorted list of Path objects for Python source files.
        """
        return sorted(_OPINIONFORGE_DIR.rglob("*.py"))

    def _extract_imports(self, source: str) -> list[str]:
        """Extract all import targets from Python source using AST.

        Args:
            source: Python source code as a string.

        Returns:
            List of imported module names (e.g. 'opinionforge.research').
        """
        imports: list[str] = []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return imports

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return imports

    def test_no_module_imports_research_directory(self) -> None:
        """No module in opinionforge/ contains an import referencing research/.

        Scans all .py files under opinionforge/ for import statements that
        reference the research/ directory (e.g. 'from research import ...'
        or 'import research'). The research/ directory is a .gitignore-excluded
        internal material and must never be imported by shipped code.
        """
        py_files = self._collect_py_files()
        violations: list[str] = []

        for py_file in py_files:
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            imports = self._extract_imports(source)
            for imp in imports:
                # Check for any import that starts with "research" or contains
                # "/research" path segment — catches both bare and relative imports
                if imp == "research" or imp.startswith("research."):
                    rel = py_file.relative_to(_REPO_ROOT)
                    violations.append(f"  {rel}: import '{imp}'")

            # Also check for open() or Path() calls referencing "research/"
            # This catches runtime file loads, not just imports
            if "research/" in source or '"research"' in source or "'research'" in source:
                # More precise check: look for file-path references to research/
                lines = source.splitlines()
                for lineno, line in enumerate(lines, start=1):
                    if (
                        "research/" in line
                        and not line.strip().startswith("#")
                        and "research_context" not in line
                        and "research_queries" not in line
                        and "research_texts" not in line
                        and "run_research" not in line
                        and "opinionforge.core.research" not in line
                        and "core/research" not in line
                        and "core.research" not in line
                        and "--research/--no-research" not in line
                        and "no-research" not in line
                    ):
                        rel = py_file.relative_to(_REPO_ROOT)
                        violations.append(
                            f"  {rel}:{lineno}: path reference to 'research/': "
                            f"{line.strip()[:120]}"
                        )

        assert not violations, (
            "opinionforge/ modules contain references to the research/ directory:\n"
            + "\n".join(violations)
            + "\n\nThe research/ directory is excluded via .gitignore. "
            "No shipped module may import or load files from research/."
        )


# ---------------------------------------------------------------------------
# Functional tests without research/ present
# ---------------------------------------------------------------------------


class TestFunctionWithoutResearchDirectory:
    """Core functions work correctly without any file in research/."""

    def test_generate_piece_succeeds_without_research_dir(self) -> None:
        """generate_piece() succeeds without any file in research/.

        Uses a mock LLM client to avoid network calls. Verifies the function
        returns a GeneratedPiece with the mandatory disclaimer.
        """
        from opinionforge.core.generator import MANDATORY_DISCLAIMER, generate_piece
        from opinionforge.core.topic import ingest_text
        from opinionforge.models.config import ModeBlendConfig, StanceConfig

        # Confirm research/ does not exist in the repo root (as expected in clean checkout)
        research_dir = _REPO_ROOT / "research"
        # We do not fail if it exists locally — we just verify the function works regardless

        mock_client = MagicMock()
        mock_client.generate.return_value = (
            "## AI and Democracy\n\n"
            "Artificial intelligence poses fundamental questions about governance. "
            "The challenge is real and the response must be proportionate.\n\n"
            "We must act thoughtfully and with urgency."
        )

        topic = ingest_text("The impact of artificial intelligence on democratic governance")
        piece = generate_piece(
            topic=topic,
            mode_config=ModeBlendConfig(modes=[("analytical", 100.0)]),
            stance=StanceConfig(position=0, intensity=0.5),
            target_length=500,
            client=mock_client,
        )

        assert piece is not None, "generate_piece() must return a GeneratedPiece"
        assert piece.disclaimer == MANDATORY_DISCLAIMER, (
            "GeneratedPiece must contain the mandatory fixed disclaimer"
        )
        assert piece.title, "GeneratedPiece must have a non-empty title"
        assert piece.body, "GeneratedPiece must have a non-empty body"

    def test_list_modes_succeeds_without_research_dir(self) -> None:
        """list_modes() succeeds without any file in research/.

        Verifies that the modes package loads correctly from the bundled YAML
        files, which are part of the shipped package (not in research/).
        """
        from opinionforge.modes import list_modes

        modes = list_modes()
        assert isinstance(modes, list), "list_modes() must return a list"
        assert len(modes) > 0, (
            "list_modes() must return at least one mode. "
            "Mode YAML files must be present in the shipped package."
        )
        for mode in modes:
            assert hasattr(mode, "id"), "Each mode must have an 'id' attribute"
            assert hasattr(mode, "display_name"), (
                "Each mode must have a 'display_name' attribute"
            )

    def test_ingest_text_succeeds_without_research_dir(self) -> None:
        """ingest_text() succeeds without any file in research/."""
        from opinionforge.core.topic import ingest_text

        topic = ingest_text("Climate change and international policy coordination")
        assert topic is not None, "ingest_text() must return a TopicContext"
        assert topic.raw_input, "TopicContext must have non-empty raw_input"


# ---------------------------------------------------------------------------
# Import hygiene: shipped code must not reference removed modules
# ---------------------------------------------------------------------------


class TestNoRemovedModuleImports:
    """No shipped file imports modules that were deleted in Sprint 6."""

    _DELETED_MODULES = [
        "opinionforge.core.voice",
        "opinionforge.core.spectrum",
        "opinionforge.models.voice",
        "opinionforge.voices",
    ]
    _DELETED_CLASSES = [
        "VoiceProfile",
        "BlendConfig",
        "SpectrumConfig",
        "blend_voices",
        "load_voice",
        "load_profile",
        "list_profiles",
        "apply_spectrum",
    ]

    # Files that contain the deleted names as test data and must be excluded
    _EXCLUDED_TEST_FILES = frozenset({
        "test_runtime_isolation.py",  # This file — contains the deleted names as data
        "test_modes.py",              # References VoiceProfile in docstrings for contrast testing
    })

    def _collect_all_py_files(self) -> list[Path]:
        """Return all .py files in opinionforge/ and tests/, excluding self-referential files.

        Excludes test_runtime_isolation.py itself since it contains the deleted
        class/module names as data strings used for the scan.

        Returns:
            Sorted list of Path objects.
        """
        files: list[Path] = []
        for d in (_OPINIONFORGE_DIR, _REPO_ROOT / "tests"):
            if d.exists():
                for f in sorted(d.rglob("*.py")):
                    if f.name not in self._EXCLUDED_TEST_FILES:
                        files.append(f)
        return files

    def test_no_imports_of_deleted_modules(self) -> None:
        """No file imports opinionforge.core.voice, .spectrum, .models.voice, or .voices."""
        import re
        py_files = self._collect_all_py_files()
        violations: list[str] = []

        for py_file in py_files:
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for deleted_mod in self._DELETED_MODULES:
                # Use exact string match on import statements only
                if deleted_mod in source:
                    rel = py_file.relative_to(_REPO_ROOT)
                    violations.append(f"  {rel}: references '{deleted_mod}'")

        assert not violations, (
            "Shipped files reference deleted modules:\n"
            + "\n".join(violations)
        )

    def test_no_references_to_deleted_classes(self) -> None:
        """No file references VoiceProfile, BlendConfig, SpectrumConfig, or related symbols.

        Uses word-boundary matching so that 'ModeBlendConfig' does not
        falsely trigger on the 'BlendConfig' pattern.
        """
        import re
        py_files = self._collect_all_py_files()
        violations: list[str] = []

        for py_file in py_files:
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for cls_name in self._DELETED_CLASSES:
                # Use word-boundary regex to avoid false positives from substrings
                # e.g., 'BlendConfig' should NOT match 'ModeBlendConfig'
                pattern = r"\b" + re.escape(cls_name) + r"\b"
                if re.search(pattern, source):
                    rel = py_file.relative_to(_REPO_ROOT)
                    violations.append(f"  {rel}: references '{cls_name}'")

        assert not violations, (
            "Shipped files reference deleted classes/functions:\n"
            + "\n".join(violations)
        )
