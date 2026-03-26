"""Modes package with profile loading utilities."""

from __future__ import annotations

from pathlib import Path

import yaml

from opinionforge.models.mode import ModeProfile

PROFILES_DIR = Path(__file__).parent / "profiles"


def load_mode(mode_id: str) -> ModeProfile:
    """Load a single mode profile from a YAML file.

    Args:
        mode_id: The slug identifier for the mode (e.g. 'polemical').

    Returns:
        A validated ModeProfile instance.

    Raises:
        FileNotFoundError: If no YAML file exists for the given mode_id.
    """
    profile_path = PROFILES_DIR / f"{mode_id}.yaml"
    if not profile_path.exists():
        available = [p.stem for p in PROFILES_DIR.glob("*.yaml")]
        msg = f"Mode profile '{mode_id}' not found at {profile_path}."
        if available:
            msg += f" Available modes: {', '.join(sorted(available))}"
        else:
            msg += " No mode profiles are currently installed."
        raise FileNotFoundError(msg)

    with open(profile_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ModeProfile.model_validate(data)


def list_modes() -> list[ModeProfile]:
    """Load all mode profiles from the profiles directory.

    Returns:
        A list of validated ModeProfile instances, sorted by id.
        Returns an empty list if no profiles are installed.
    """
    profiles: list[ModeProfile] = []
    for path in sorted(PROFILES_DIR.glob("*.yaml")):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        profiles.append(ModeProfile.model_validate(data))
    return profiles
