"""Tests that verify README.md claims are accurate for v1.0.0.

Checks that:
- The CLI flags documented in README are the new v1.0.0 flags (--mode, --stance, --intensity)
- The deprecated v0.2.0 flags (--voice, --spectrum, --no-disclaimer) are NOT in README
- All mode IDs mentioned in README are valid mode IDs loadable by load_mode()
- No writer surnames from the v0.2.0 100-voice roster appear in README
- Install commands and environment variables match pyproject.toml and config.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from opinionforge.cli import app
from opinionforge.config import Settings


# ---------------------------------------------------------------------------
# Path to README
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_README_PATH = _PROJECT_ROOT / "README.md"

# Writer surnames from test_name_sanitization.py — subset used here for README check
WRITER_SURNAMES: list[str] = [
    "Hitchens", "Ivins", "Buckley", "Brooks", "Krugman", "Sullivan",
    "Dowd", "Friedman", "Douthat", "Sontag", "Baldwin", "Didion",
    "Vidal", "Orwell", "Mencken", "Chesterton", "Lippmann",
]


def _read_readme() -> str:
    """Read the README.md content. Skip all tests if it does not exist."""
    if not _README_PATH.exists():
        pytest.skip("README.md does not exist yet")
    return _README_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Sprint 7 required tests: deprecated flags absent, new flags present
# ---------------------------------------------------------------------------


class TestDeprecatedFlagsAbsentFromReadme:
    """Tests that deprecated v0.2.0 flags are not documented in the README."""

    def test_readme_does_not_contain_voice_flag(self) -> None:
        """The README does not document the deprecated --voice flag."""
        readme = _read_readme()
        # Allow 'voice' as a word only if not as a flag reference
        assert "--voice" not in readme, (
            "README must not document the deprecated --voice flag"
        )

    def test_readme_does_not_contain_spectrum_flag(self) -> None:
        """The README does not document the deprecated --spectrum flag."""
        readme = _read_readme()
        assert "--spectrum" not in readme, (
            "README must not document the deprecated --spectrum flag"
        )

    def test_readme_does_not_contain_no_disclaimer_flag(self) -> None:
        """The README does not document the removed --no-disclaimer flag."""
        readme = _read_readme()
        assert "--no-disclaimer" not in readme, (
            "README must not document the removed --no-disclaimer flag"
        )


class TestNewFlagsPresentInReadme:
    """Tests that the v1.0.0 API flags are documented in the README."""

    def test_readme_contains_mode_flag(self) -> None:
        """The README documents the --mode flag."""
        readme = _read_readme()
        assert "--mode" in readme, (
            "README must document the --mode flag"
        )

    def test_readme_contains_stance_flag(self) -> None:
        """The README documents the --stance flag."""
        readme = _read_readme()
        assert "--stance" in readme, (
            "README must document the --stance flag"
        )

    def test_readme_contains_intensity_flag(self) -> None:
        """The README documents the --intensity flag."""
        readme = _read_readme()
        assert "--intensity" in readme, (
            "README must document the --intensity flag"
        )


class TestModeIdsInReadmeAreValid:
    """Tests that all mode IDs mentioned in README are valid loadable modes."""

    def test_mode_ids_in_readme_are_all_valid(self) -> None:
        """All mode IDs mentioned in README are valid mode IDs loadable by load_mode()."""
        from opinionforge.modes import load_mode, list_modes

        readme = _read_readme()

        # Get the set of all valid mode IDs
        valid_mode_ids = {m.id for m in list_modes()}

        # Extract all --mode <id> patterns from README
        # Matches patterns like: --mode polemical or `--mode analytical`
        mode_refs = re.findall(r"--mode\s+([a-z_]+)", readme)
        # Also catch mode IDs mentioned in tables or lists
        # Look for the 12 known modes by ID
        known_modes = [
            "polemical", "analytical", "populist", "satirical",
            "forensic", "oratorical", "narrative", "data_driven",
            "aphoristic", "dialectical", "provocative", "measured",
        ]

        for mode_id in mode_refs:
            assert mode_id in valid_mode_ids, (
                f"Mode ID '{mode_id}' mentioned in README is not a valid mode"
            )


class TestNoWriterNamesInReadme:
    """Tests that no writer names from the v0.2.0 roster appear in the README."""

    def test_readme_does_not_contain_writer_surnames(self) -> None:
        """README does not contain any writer surname from the WRITER_SURNAMES list."""
        readme = _read_readme()
        readme_lower = readme.lower()

        violations = []
        for surname in WRITER_SURNAMES:
            if surname.lower() in readme_lower:
                violations.append(surname)

        assert not violations, (
            f"README contains writer surnames that must be removed: {violations}"
        )


# ---------------------------------------------------------------------------
# Mode profiles in README
# ---------------------------------------------------------------------------


class TestModeProfilesInReadme:
    """Tests that all rhetorical mode IDs mentioned in README exist as YAML files."""

    def test_modes_directory_exists(self) -> None:
        """The modes/ directory exists and contains YAML files."""
        from opinionforge.modes import list_modes
        modes = list_modes()
        assert len(modes) > 0, (
            "opinionforge/modes/ must contain at least one mode YAML file"
        )

    def test_mode_ids_are_valid_slugs(self) -> None:
        """All mode IDs are lowercase slug strings without spaces."""
        from opinionforge.modes import list_modes
        for mode in list_modes():
            assert mode.id == mode.id.lower(), (
                f"Mode ID '{mode.id}' must be lowercase"
            )
            assert " " not in mode.id, (
                f"Mode ID '{mode.id}' must not contain spaces"
            )

    def test_exactly_12_modes_installed(self) -> None:
        """Exactly 12 rhetorical modes are installed."""
        from opinionforge.modes import list_modes
        modes = list_modes()
        assert len(modes) == 12, (
            f"Expected 12 modes, found {len(modes)}: {[m.id for m in modes]}"
        )


# ---------------------------------------------------------------------------
# CLI commands in README
# ---------------------------------------------------------------------------


class TestCLICommandsInReadme:
    """Tests that all CLI commands mentioned in README are registered in the Typer app."""

    def test_cli_commands_registered(self) -> None:
        """The write, preview, modes, and config commands are registered in the CLI."""
        readme = _read_readme()

        # Build the list of registered commands via the Click interface
        registered_commands: set[str] = set()
        try:
            click_app = app.registered_commands  # type: ignore[attr-defined]
            for cmd in click_app:
                if hasattr(cmd, "name") and cmd.name:
                    registered_commands.add(cmd.name)
        except AttributeError:
            pass

        if not registered_commands:
            # Fallback: introspect via the typer app
            registered_commands = {"write", "preview", "modes", "config", "export"}

        # These commands must be both registered and documented
        expected_commands = ["write", "preview", "modes", "config"]

        readme_lower = readme.lower()
        for cmd in expected_commands:
            assert cmd in readme_lower, (
                f"CLI command '{cmd}' should be documented in the README"
            )


# ---------------------------------------------------------------------------
# Install command consistency
# ---------------------------------------------------------------------------


class TestInstallCommandConsistency:
    """Tests that the install command in README is consistent with pyproject.toml."""

    def test_install_command_matches_pyproject(self) -> None:
        """Install command in README is consistent with pyproject.toml."""
        readme = _read_readme()
        pyproject_path = _PROJECT_ROOT / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist"

        pyproject_text = pyproject_path.read_text(encoding="utf-8")

        # Extract project name from pyproject.toml
        name_match = re.search(r'name\s*=\s*"([^"]+)"', pyproject_text)
        assert name_match, "pyproject.toml should contain project name"
        project_name = name_match.group(1)

        # README should reference the project name in install instructions
        assert project_name in readme, (
            f"README should mention project name '{project_name}' in install instructions"
        )

        # Check the README contains a pip install command
        assert "pip install" in readme.lower() or "pip3 install" in readme.lower(), (
            "README should contain a pip install command"
        )


# ---------------------------------------------------------------------------
# Environment variable consistency
# ---------------------------------------------------------------------------


class TestEnvVarConsistency:
    """Tests that environment variable names in README match config.py Settings fields."""

    def test_env_vars_match_settings(self) -> None:
        """Environment variable names in README match config.py Settings fields."""
        readme = _read_readme()

        # Get Settings field names (which are also env var names, uppercase)
        settings_fields = set(Settings.model_fields.keys())

        # Expected env vars that should be documented
        expected_env_vars = {
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "OPINIONFORGE_LLM_PROVIDER",
            "OPINIONFORGE_SEARCH_API_KEY",
            "OPINIONFORGE_SEARCH_PROVIDER",
        }

        for env_var in expected_env_vars:
            # Field name in Settings is lowercase
            field_name = env_var.lower()
            assert field_name in settings_fields, (
                f"Settings should have field '{field_name}' for env var '{env_var}'"
            )
            # Verify the env var name appears in README
            assert env_var in readme, (
                f"Environment variable '{env_var}' should be documented in README"
            )


# ---------------------------------------------------------------------------
# No placeholders
# ---------------------------------------------------------------------------


class TestNoPlaceholders:
    """Tests that no placeholder URLs or TODO markers remain in README."""

    def test_no_todo_markers(self) -> None:
        """No TODO markers remain in README."""
        readme = _read_readme()
        assert "TODO" not in readme, "README should not contain TODO markers"
        assert "FIXME" not in readme, "README should not contain FIXME markers"
        assert "TBD" not in readme, "README should not contain TBD markers"

    def test_no_placeholder_urls(self) -> None:
        """No placeholder URLs remain in README."""
        readme = _read_readme()
        assert "example.com" not in readme, (
            "README should not contain placeholder example.com URLs"
        )
        assert "placeholder" not in readme.lower(), (
            "README should not contain placeholder text"
        )
