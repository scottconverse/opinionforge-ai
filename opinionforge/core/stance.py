"""Stance modifier that adjusts mode prompts based on position and intensity.

Replaces spectrum.py for the v1.0.0 architecture. Decouples direction
(position) from rhetorical heat (intensity), using non-political vocabulary
throughout. No named publications or political alignment labels appear.

The modifier affects five dimensions:
1. Argument selection
2. Framing
3. Source preference
4. Rhetorical intensity
5. Counterargument handling
"""

from __future__ import annotations

from opinionforge.models.config import StanceConfig


def _direction_label(position: int) -> str:
    """Return a non-political direction label for the stance position.

    Uses equity/liberty vocabulary rather than left/right political labels.

    Args:
        position: Stance position from -100 to +100.

    Returns:
        A descriptive label string using non-partisan vocabulary.
    """
    if position <= -50:
        return "strongly equity-focused"
    if position < -25:
        return "equity-leaning"
    if position <= 25:
        return "balanced"
    if position < 50:
        return "liberty-leaning"
    return "strongly liberty-focused"


def _argument_selection_instruction(position: int, intensity: float) -> str:
    """Generate argument selection instructions based on stance position and intensity.

    Args:
        position: Stance position from -100 to +100.
        intensity: Rhetorical heat from 0.0 to 1.0.

    Returns:
        Instruction string for argument selection.
    """
    if intensity < 0.2:
        return (
            "Argument Selection: Present arguments from multiple perspectives with equal weight. "
            "Give fair representation to both equity-focused and liberty-focused reasoning. "
            "Prioritize the strongest arguments regardless of their emphasis."
        )

    if position < 0:
        # Equity-focused
        if intensity > 0.8:
            return (
                "Argument Selection: Foreground equity, systemic analysis, and collective solutions. "
                "Center arguments around structural dynamics, institutional reform, and shared outcomes. "
                "Treat market-based or purely individualist arguments as insufficient by default."
            )
        return (
            "Argument Selection: Emphasize equity-based and systemic arguments while acknowledging "
            "other perspectives. Favor collective and institutional solutions. "
            "Present equity-focused reasoning as the primary framework."
        )
    elif position > 0:
        # Liberty-focused
        if intensity > 0.8:
            return (
                "Argument Selection: Foreground individual agency, market mechanisms, and institutional continuity. "
                "Center arguments around personal responsibility, voluntary exchange, and decentralized decision-making. "
                "Treat collectivist or centralized solutions as inherently problematic."
            )
        return (
            "Argument Selection: Emphasize individual agency, market-based solutions, and institutional "
            "stability while acknowledging other perspectives. Favor decentralized approaches. "
            "Present liberty-focused reasoning as the primary framework."
        )
    else:
        # Balanced (position == 0)
        return (
            "Argument Selection: Present a balanced synthesis of equity-focused and liberty-focused "
            "arguments. Acknowledge the strongest case for each emphasis. "
            "Prioritize arguments with broad empirical support."
        )


def _framing_instruction(position: int, intensity: float) -> str:
    """Generate framing instructions based on stance position and intensity.

    Args:
        position: Stance position from -100 to +100.
        intensity: Rhetorical heat from 0.0 to 1.0.

    Returns:
        Instruction string for fact framing.
    """
    if intensity < 0.2:
        return (
            "Framing: Present facts and statistics with neutral contextualization. "
            "Avoid loading language in either direction. Let data speak for itself "
            "with multiple interpretive lenses offered."
        )

    if position < 0:
        if intensity > 0.8:
            return (
                "Framing: Contextualize all facts through an equity-focused lens. "
                "Highlight disparities, systemic patterns, and collective impact. "
                "Frame statistics to emphasize the need for structural change."
            )
        return (
            "Framing: Lean toward equity-focused contextualization of facts. "
            "When presenting statistics, foreground social impact and systemic dimensions. "
            "Acknowledge alternative framings but center the equity-focused interpretation."
        )
    elif position > 0:
        if intensity > 0.8:
            return (
                "Framing: Contextualize all facts through a liberty-focused lens. "
                "Highlight individual agency, voluntary coordination, and market efficiency. "
                "Frame statistics to emphasize the costs of centralized intervention."
            )
        return (
            "Framing: Lean toward liberty-focused contextualization of facts. "
            "When presenting statistics, foreground economic impact and individual outcomes. "
            "Acknowledge alternative framings but center the liberty-focused interpretation."
        )
    else:
        return (
            "Framing: Present a balanced framing that acknowledges both systemic and individual "
            "dimensions of the evidence. Offer multiple interpretations without privileging either."
        )


def _source_preference_instruction(position: int, intensity: float) -> str:
    """Generate source preference instructions based on stance position and intensity.

    Uses evidence-category vocabulary only. No named publications.

    Args:
        position: Stance position from -100 to +100.
        intensity: Rhetorical heat from 0.0 to 1.0.

    Returns:
        Instruction string for source preferences.
    """
    if intensity < 0.2:
        return (
            "Source Preference: Draw from a balanced mix of evidence types. "
            "Prioritize peer-reviewed research, government data, and primary sources. "
            "Include empirical case studies from multiple contexts."
        )

    if position < 0:
        if intensity > 0.8:
            return (
                "Source Preference: Prioritize peer-reviewed research on systemic outcomes, "
                "government data on equity indicators, and investigative journalism documenting "
                "institutional patterns. Favor empirical case studies demonstrating collective impact."
            )
        return (
            "Source Preference: Favor peer-reviewed research and government data "
            "while including empirical case studies for credibility. "
            "Prefer investigative journalism and policy-oriented research."
        )
    elif position > 0:
        if intensity > 0.8:
            return (
                "Source Preference: Prioritize market analysis, institutional research, and "
                "empirical case studies on voluntary exchange and economic outcomes. "
                "Favor quantitative research demonstrating individual and market-level results."
            )
        return (
            "Source Preference: Favor market analysis and institutional research "
            "while including peer-reviewed empirical studies for credibility. "
            "Prefer research emphasizing measurable economic and individual outcomes."
        )
    else:
        return (
            "Source Preference: Draw from a balanced mix including peer-reviewed research, "
            "government data, market analysis, and investigative journalism. "
            "Prioritize sources with strong empirical methodology regardless of emphasis."
        )


def _rhetorical_intensity_instruction(position: int, intensity: float) -> str:
    """Generate rhetorical intensity instructions based on stance position and intensity.

    Args:
        position: Stance position from -100 to +100.
        intensity: Rhetorical heat from 0.0 to 1.0.

    Returns:
        Instruction string for rhetorical intensity.
    """
    if intensity < 0.2:
        return (
            "Rhetorical Intensity: Maintain a measured, deliberative tone. "
            "Write with intellectual precision rather than emotional force. "
            "Acknowledge complexity and avoid sweeping declarations."
        )

    if intensity < 0.5:
        return (
            "Rhetorical Intensity: Write with moderate conviction. "
            "Take a clear position but express it with analytical confidence "
            "rather than polemical force. Use persuasion over provocation."
        )

    if intensity < 0.8:
        return (
            "Rhetorical Intensity: Write with strong conviction and argumentative energy. "
            "Take firm positions and argue them forcefully. Use vivid language "
            "and pointed rhetoric to drive home key arguments."
        )

    return (
        "Rhetorical Intensity: Write with maximum argumentative conviction. "
        "Deploy aggressive rhetoric, pointed moral judgments, and uncompromising language. "
        "Treat the opposing position as not merely wrong but consequentially so."
    )


def _counterargument_instruction(position: int, intensity: float) -> str:
    """Generate counterargument handling instructions based on stance position and intensity.

    Args:
        position: Stance position from -100 to +100.
        intensity: Rhetorical heat from 0.0 to 1.0.

    Returns:
        Instruction string for counterargument handling.
    """
    if intensity < 0.2:
        return (
            "Counterargument Handling: Present counterarguments fairly and in their strongest form. "
            "Engage with the best version of opposing views before offering rebuttals. "
            "Acknowledge genuine points of uncertainty or legitimate disagreement."
        )

    if intensity < 0.5:
        return (
            "Counterargument Handling: Acknowledge major counterarguments and address them "
            "substantively, but give your position more argumentative weight. "
            "Rebut opposing views with evidence rather than dismissal."
        )

    if intensity < 0.8:
        return (
            "Counterargument Handling: Reference counterarguments briefly before dismissing them "
            "with pointed critique. Spend more time building your case than engaging opponents. "
            "Treat opposing views as well-meaning but ultimately misguided."
        )

    return (
        "Counterargument Handling: Dismiss counterarguments with rhetorical force. "
        "Minimize opposing views where stylistically appropriate. "
        "Treat the opposing position as self-evidently flawed or consequentially wrong."
    )


def apply_stance(
    mode_prompt: str,
    stance_config: StanceConfig,
) -> str:
    """Modify a mode prompt with stance-aware instructions.

    Applies argumentative emphasis (position) and rhetorical heat (intensity)
    independently, producing stance-modified instructions appended to the base
    mode prompt. There is no ideological_baseline parameter — mode profiles
    have no ideological default.

    The output uses non-political vocabulary throughout: no 'progressive',
    'conservative', 'left', 'right', 'liberal', 'Democrat', 'Republican', or
    named publication references appear in the modifier section.

    Args:
        mode_prompt: The base mode prompt fragment (from blending or single mode).
        stance_config: The requested stance configuration with position and intensity.

    Returns:
        The mode prompt with stance modifier instructions appended.
    """
    position = stance_config.position
    intensity = stance_config.intensity
    direction_label = _direction_label(position)

    modifier_parts = [
        f"\n\n--- Stance Modifier (Position: {position:+d}, {direction_label}, "
        f"Intensity: {intensity:.2f}) ---",
        "",
        _argument_selection_instruction(position, intensity),
        _framing_instruction(position, intensity),
        _source_preference_instruction(position, intensity),
        _rhetorical_intensity_instruction(position, intensity),
        _counterargument_instruction(position, intensity),
    ]

    return mode_prompt + "\n".join(modifier_parts)
