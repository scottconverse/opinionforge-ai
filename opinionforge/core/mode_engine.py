"""Mode engine: loads mode profiles, applies blending, and provides fuzzy-match suggestions."""

from __future__ import annotations

from difflib import get_close_matches

from opinionforge.models.config import ModeBlendConfig
from opinionforge.models.mode import ModeProfile
from opinionforge.modes import list_modes
from opinionforge.modes import load_mode as _load_mode_from_package


def _available_mode_ids() -> list[str]:
    """Return all available mode IDs from the profiles directory.

    Returns:
        Sorted list of mode ID strings.
    """
    return [m.id for m in list_modes()]


def _suggest_matches(mode_id: str) -> list[str]:
    """Return fuzzy-match suggestions for a mistyped mode_id.

    Args:
        mode_id: The mode_id that was not found.

    Returns:
        A list of suggested mode IDs sorted by relevance.
    """
    ids = _available_mode_ids()
    id_matches = get_close_matches(
        mode_id.lower(), [i.lower() for i in ids], n=3, cutoff=0.5
    )
    id_map = {i.lower(): i for i in ids}
    return [id_map[m] for m in id_matches]


def load_mode(mode_id: str) -> ModeProfile:
    """Load and validate a single mode profile by ID.

    Args:
        mode_id: The slug identifier for the mode (e.g. 'polemical').

    Returns:
        A validated ModeProfile instance.

    Raises:
        ValueError: When the mode_id is not found, with a message containing
            'not found' and, if any close matches exist, one or more suggestions.
    """
    try:
        return _load_mode_from_package(mode_id)
    except FileNotFoundError:
        suggestions = _suggest_matches(mode_id)
        msg = f"Mode '{mode_id}' not found."
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}?"
        raise ValueError(msg) from None


def blend_modes(blend_config: ModeBlendConfig) -> str:
    """Produce a merged system_prompt_fragment from weighted mode profiles.

    For a single mode at 100%, returns that mode's system_prompt_fragment
    unmodified. For multiple modes, produces a blended prompt fragment that
    references each mode's characteristics with proportional emphasis.

    The blending is deterministic: same inputs always produce the same output.

    Args:
        blend_config: A ModeBlendConfig with (mode_id, weight) pairs summing to 100.

    Returns:
        A composed system prompt fragment string.

    Raises:
        ValueError: If any mode_id in the blend is not found.
    """
    profiles: list[tuple[ModeProfile, float]] = []
    for mode_id, weight in blend_config.modes:
        profile = load_mode(mode_id)
        profiles.append((profile, weight))

    # Single mode at 100% — return unmodified
    if len(profiles) == 1:
        return profiles[0][0].system_prompt_fragment

    return _compose_blended_fragment(profiles)


def _compose_blended_fragment(profiles: list[tuple[ModeProfile, float]]) -> str:
    """Compose a blended system prompt fragment from multiple weighted mode profiles.

    Creates a naturally-reading set of style instructions that proportionally
    represents each mode's characteristics based on their weight.

    Args:
        profiles: List of (ModeProfile, weight) tuples.

    Returns:
        A blended system prompt fragment string.
    """
    parts: list[str] = []
    parts.append(
        "You are writing in a blended rhetorical mode that combines the following styles:\n"
    )

    for profile, weight in sorted(profiles, key=lambda x: x[1], reverse=True):
        pct = int(weight)
        parts.append(
            f"--- {profile.display_name} ({pct}% influence) ---\n"
            f"{profile.system_prompt_fragment}\n"
        )

    mode_descriptions = []
    for profile, weight in sorted(profiles, key=lambda x: x[1], reverse=True):
        pct = int(weight)
        mode_descriptions.append(f"{profile.display_name}'s approach ({pct}%)")

    parts.append(
        "\n--- Blending Instructions ---\n"
        "Combine these modes into a single coherent voice. "
        "The dominant mode should drive the overall structure and argument approach, "
        "while secondary modes contribute their distinctive rhetorical qualities "
        "and tonal characteristics in proportion to their weight.\n\n"
        f"Primary blend: {'; '.join(mode_descriptions)}.\n"
        "The result should read as one unified voice, not as alternating styles."
    )

    return "\n".join(parts)
