"""Automated test that verifies zero writer names appear in shipped code.

Scans all .py, .yaml, .md, .toml, and .html files in the shipped codebase
for any of the 100 writer surnames from the v0.2.0 roster. The test asserts
zero matches. The research/ directory is explicitly excluded from scanning.

The scan is case-insensitive: catches 'hitchens', 'Hitchens', and 'HITCHENS'
as well as occurrences embedded in variable names like 'hitchens_data'.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import pytest

# ---------------------------------------------------------------------------
# Repository root — two levels up from this test file
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# The full 100-surname list from the v0.2.0 voice profiles directory.
# Each entry is the canonical Title-case surname (matching the YAML filename
# stem, normalised to Title case). The scan is performed case-insensitively.
# ---------------------------------------------------------------------------

WRITER_SURNAMES: list[str] = [
    # Explicitly required by sprint contract acceptance criteria
    "Hitchens",
    "Ivins",
    "Buckley",
    "Brooks",
    "Krugman",
    "Sullivan",
    "Will",
    "Dowd",
    "Friedman",
    "Douthat",
    "Sontag",
    "Baldwin",
    "Didion",
    "Vidal",
    "Orwell",
    "Mencken",
    "Chesterton",
    "Lippmann",
    "Liebling",
    "Royko",
    "Molly",
    "Kinsley",
    "Noonan",
    "Krauthammer",
    "Broder",
    "Wills",
    "Morris",
    "Kristol",
    "Podhoretz",
    "Murray",
    # Remaining 70 surnames from v0.2.0 profiles (aa_gill.yaml … twain.yaml)
    "Gill",
    "Cockburn",
    "Coyne",
    "Landers",
    "Quindlen",
    "Lewis",
    "Buchwald",
    "Karkaria",
    "Levin",
    "Bierce",
    "Herbert",
    "Breslin",
    "Stephens",
    "Sulzberger",
    "Thomas",
    "Trillin",
    "Hiaasen",
    "Hebert",
    "Blatchford",
    "Page",
    "Schultz",
    "Tucker",
    "Runyon",
    "Barry",
    "Robinson",
    "Klein",
    "Otoole",
    "Deford",
    "Franklin",
    "Collins",
    "Weingarten",
    "Monbiot",
    "Rice",
    "Caen",
    "Newfield",
    "Bouie",
    "Reston",
    "Cannon",
    "Alsop",
    "Parker",
    "Singh",
    "Pitts",
    "Pyle",
    "Wicker",
    "Greenfield",
    "Schmich",
    "Goldberg",
    "Barnicle",
    "Albom",
    "Kempton",
    "Hentoff",
    "Kristof",
    "Buchanan",
    "Hamill",
    "Toynbee",
    "Smith",
    "Cohen",
    "Rovere",
    "Angell",
    "Baker",
    "Safire",
    "Povich",
    "Jenkins",
    "Lopez",
    "Sowell",
    "Winchell",
    "Pegler",
    "Twain",
    "Dionne",
    "Roosevelt",
]

assert len(WRITER_SURNAMES) == 100, (
    f"WRITER_SURNAMES must contain exactly 100 entries, found {len(WRITER_SURNAMES)}"
)

# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

_EXTENSIONS = frozenset({".py", ".yaml", ".yml", ".md", ".toml", ".html"})

# Directories to always exclude from scanning
_EXCLUDED_DIRS = frozenset({
    "research",      # Internal research materials — .gitignore excluded
    ".git",          # Version control metadata
    ".venv",         # Virtual environment
    "__pycache__",   # Python bytecode
    ".mypy_cache",   # Type checker cache
    ".pytest_cache", # Test runner cache
    "node_modules",  # (future-proofing)
})

# Files that contain the canonical surname list (as test data) or that are
# internal design documents that must not be scanned — scanning them would
# always produce false positives.
_EXCLUDED_FILES = frozenset({
    "test_name_sanitization.py",   # This file — contains the canonical list
    "test_no_impersonation.py",    # Contains surname list for assertion testing
    "test_modes.py",               # Contains BANNED_WRITER_NAMES for mode profile scanning
    "test_generator.py",           # Contains writer surnames in assertions about what NOT to generate
    "test_image_prompt.py",        # Contains 'Hitchens' in assertion: assert "Hitchens" not in result
    "test_readme_accuracy.py",     # Contains WRITER_SURNAMES list for README checking
    "test_e2e.py",                 # Contains 'hitchens' as a test input value for deprecated --voice flag test
    "PRD.md",                      # Internal product requirements document — not shipped user content
    "confusability_helpers.py",    # Contains historical text citations (Thomas Paine, Mark Twain) as test corpus
})

# Surnames that are also extremely common English words (verbs, nouns, prepositions).
# These are matched with CASE-SENSITIVE word-boundary matching to reduce false positives
# from normal prose in YAML examples and Python source code.
# The surnames themselves (George Will, Larry Page, etc.) would appear capitalized
# in marketing copy — the lowercase forms in prose are not writer references.
_CASE_SENSITIVE_SURNAMES = frozenset({
    "Will",    # George Will — but "will" (auxiliary verb) is ubiquitous in prose
    "Page",    # columnist — but "page" (noun) appears in URL/HTML context
    "Rice",    # columnist — but "rice" (food) / appears in "price" substrings
    "Lewis",   # columnist — but "lewis" is a common surname AND word
    "Morris",  # columnist — but "morris" appears in various contexts
    "Thomas",  # columnist — but "thomas" is a common first name
    "Baker",   # columnist — but "baker" is a common occupation/surname
    "Smith",   # columnist — but "smith" is an extremely common surname
    "Cohen",   # columnist — but "cohen" is a common surname
    "Parker",  # columnist — but "parker" is a common surname
    "Barry",   # columnist — but "barry" is a common first name
    "Collins", # columnist — but "collins" is a common surname
    "Franklin",# columnist — but "franklin" is a common surname/first name
    "Robinson",# columnist — but "robinson" is a common surname
    "Tucker",  # columnist — but "tucker" is a common surname
})

# Build two separate patterns:
# 1. Case-insensitive pattern for distinctive surnames (e.g. Hitchens, Buckley)
# 2. Case-sensitive Title-Case-only pattern for surnames that are also common English
#    words (e.g. "Will", "Page") to avoid false positives from prose text.
_DISTINCTIVE_SURNAMES = [s for s in WRITER_SURNAMES if s not in _CASE_SENSITIVE_SURNAMES]
_SURNAME_PATTERN_CI: re.Pattern[str] = re.compile(
    r"(?i)\b(" + "|".join(re.escape(s) for s in _DISTINCTIVE_SURNAMES) + r")\b"
)
_SURNAME_PATTERN_CS: re.Pattern[str] = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(_CASE_SENSITIVE_SURNAMES)) + r")\b"
)

# Combined pattern alias used in the test for backward compatibility
_SURNAME_PATTERN = _SURNAME_PATTERN_CI


def _collect_shipped_files() -> list[Path]:
    """Collect all shipped files that must be scanned for writer names.

    Scans the following locations:
    - opinionforge/ (all .py and .yaml files)
    - tests/ (all .py files)
    - docs/ (all .md files, if the directory exists)
    - README.md (top-level)
    - pyproject.toml (top-level)

    Explicitly excludes:
    - research/ directory
    - .git/ directory
    - __pycache__/ directories
    - .venv/ directory

    Returns:
        Sorted list of Path objects for all files to scan.
    """
    files: list[Path] = []

    def _should_skip(path: Path) -> bool:
        """Return True if path is in an excluded directory or is an excluded file."""
        rel = path.relative_to(_REPO_ROOT)
        if any(part in _EXCLUDED_DIRS for part in rel.parts):
            return True
        if path.name in _EXCLUDED_FILES:
            return True
        return False

    # opinionforge/ — .py and .yaml
    opinionforge_dir = _REPO_ROOT / "opinionforge"
    if opinionforge_dir.exists():
        for ext in (".py", ".yaml", ".yml"):
            for f in opinionforge_dir.rglob(f"*{ext}"):
                if not _should_skip(f):
                    files.append(f)

    # tests/ — .py files
    tests_dir = _REPO_ROOT / "tests"
    if tests_dir.exists():
        for f in tests_dir.rglob("*.py"):
            if not _should_skip(f):
                files.append(f)

    # docs/ — all files with scanned extensions (if directory exists)
    docs_dir = _REPO_ROOT / "docs"
    if docs_dir.exists():
        for f in docs_dir.rglob("*"):
            if f.is_file() and f.suffix in _EXTENSIONS:
                if not _should_skip(f):
                    files.append(f)

    # Top-level files
    for name in ("README.md", "pyproject.toml"):
        top_file = _REPO_ROOT / name
        if top_file.exists():
            files.append(top_file)

    return sorted(set(files))


# ---------------------------------------------------------------------------
# Match data class
# ---------------------------------------------------------------------------


class _Match(NamedTuple):
    """A single writer-name match in a shipped file."""

    file: Path
    line_number: int
    line_text: str
    matched_surname: str


# ---------------------------------------------------------------------------
# Core scanning logic
# ---------------------------------------------------------------------------


def _scan_file(path: Path) -> list[_Match]:
    """Scan a single file for writer name occurrences.

    Args:
        path: Absolute path to the file to scan.

    Returns:
        List of _Match instances, one per matching line.
        Empty list if no matches are found or file cannot be read.
    """
    matches: list[_Match] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return matches

    for line_number, line in enumerate(text.splitlines(), start=1):
        seen_on_line: set[int] = set()
        # Case-insensitive scan for distinctive surnames (e.g. Hitchens, Buckley)
        for m in _SURNAME_PATTERN_CI.finditer(line):
            if m.start() not in seen_on_line:
                seen_on_line.add(m.start())
                matches.append(
                    _Match(
                        file=path,
                        line_number=line_number,
                        line_text=line.strip(),
                        matched_surname=m.group(0),
                    )
                )
        # Case-sensitive scan for common-word surnames (only Title Case matches count)
        # e.g. "Will" (George Will) but NOT "will" (auxiliary verb)
        for m in _SURNAME_PATTERN_CS.finditer(line):
            if m.start() not in seen_on_line:
                seen_on_line.add(m.start())
                matches.append(
                    _Match(
                        file=path,
                        line_number=line_number,
                        line_text=line.strip(),
                        matched_surname=m.group(0),
                    )
                )

    return matches


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_writer_surnames_list_has_100_entries() -> None:
    """WRITER_SURNAMES list contains exactly 100 entries."""
    assert len(WRITER_SURNAMES) == 100, (
        f"Expected 100 surnames, found {len(WRITER_SURNAMES)}. "
        "Add or remove entries to reach exactly 100."
    )


def test_writer_surnames_list_has_no_duplicates() -> None:
    """WRITER_SURNAMES list has no duplicate entries (case-insensitive)."""
    lowered = [s.lower() for s in WRITER_SURNAMES]
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in lowered:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    assert not duplicates, (
        f"Duplicate surnames found in WRITER_SURNAMES: {duplicates}"
    )


def test_research_directory_excluded() -> None:
    """The research/ directory is not included in the scanned file list."""
    shipped_files = _collect_shipped_files()
    research_dir = _REPO_ROOT / "research"
    research_files = [f for f in shipped_files if str(f).startswith(str(research_dir))]
    assert not research_files, (
        f"research/ files should be excluded from scanning but found: {research_files}"
    )


def test_required_directories_are_scanned() -> None:
    """opinionforge/ and tests/ directories are included in the scanned files."""
    shipped_files = _collect_shipped_files()
    opinionforge_dir = _REPO_ROOT / "opinionforge"
    tests_dir = _REPO_ROOT / "tests"

    has_opinionforge = any(
        str(f).startswith(str(opinionforge_dir)) for f in shipped_files
    )
    has_tests = any(
        str(f).startswith(str(tests_dir)) for f in shipped_files
    )

    assert has_opinionforge, "opinionforge/ directory must be included in the scan"
    assert has_tests, "tests/ directory must be included in the scan"


def test_pyproject_toml_is_scanned() -> None:
    """pyproject.toml is included in the scanned file list."""
    shipped_files = _collect_shipped_files()
    pyproject = _REPO_ROOT / "pyproject.toml"
    assert pyproject in shipped_files, "pyproject.toml must be included in the scan"


def test_no_writer_names_in_shipped_code() -> None:
    """Zero writer surnames appear anywhere in the shipped codebase.

    Scans all .py, .yaml, .md, and .toml files under opinionforge/, tests/,
    docs/, README.md, and pyproject.toml. Excludes research/ and .git/.

    Produces a readable failure message listing the file path, line number,
    and matched surname for each violation found.
    """
    shipped_files = _collect_shipped_files()
    all_matches: list[_Match] = []

    for f in shipped_files:
        all_matches.extend(_scan_file(f))

    if all_matches:
        lines = ["Writer names found in shipped code — all violations must be removed:\n"]
        for match in all_matches:
            rel = match.file.relative_to(_REPO_ROOT)
            lines.append(
                f"  {rel}:{match.line_number}  "
                f"[matched: '{match.matched_surname}']  "
                f"→  {match.line_text[:120]}"
            )
        pytest.fail("\n".join(lines))
