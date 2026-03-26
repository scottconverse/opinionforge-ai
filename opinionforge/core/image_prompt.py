"""Image prompt generation engine for opinion piece header images.

Produces DALL-E / Midjourney / Stable Diffusion prompts with platform-specific
aspect ratios, pixel dimensions, and style directives derived from piece content.
"""

from __future__ import annotations

from opinionforge.config import Settings
from opinionforge.core.preview import LLMClient, create_llm_client
from opinionforge.models.config import ImagePromptConfig
from opinionforge.models.piece import GeneratedPiece

# System prompt for image prompt generation
_IMAGE_PROMPT_SYSTEM = (
    "You are an expert visual prompt engineer specializing in header images for "
    "long-form opinion and journalism content. Given a topic description, key entities, "
    "and domain, you produce a vivid, specific subject description for an image generation "
    "model. Focus on: concrete visual metaphors, symbolic imagery, and scene composition. "
    "Do NOT reference any specific living person's name or likeness. "
    "Do NOT include text overlays or title text in the image description. "
    "Respond with ONLY the subject description — one to two sentences, no commentary."
)

# Style directive fragments for each style value
_STYLE_DIRECTIVES: dict[str, str] = {
    "photorealistic": (
        "photorealistic photography, shot on a high-end camera, "
        "natural lighting, documentary photograph style"
    ),
    "editorial": (
        "editorial illustration, professional editorial art, "
        "bold composition, magazine cover quality"
    ),
    "cartoon": (
        "cartoon illustration, illustrated style, bold outlines, "
        "vibrant colors, stylized and illustrated"
    ),
    "minimalist": (
        "minimalist design, clean composition, minimal elements, "
        "geometric simplicity, negative space, flat design"
    ),
    "vintage": (
        "vintage style, retro aesthetic, aged texture, "
        "nostalgic color palette, classic design sensibility"
    ),
    "abstract": (
        "abstract art, abstract composition, non-representational, "
        "conceptual imagery, artistic abstraction"
    ),
}

# Human-readable style label for the Style: line
_STYLE_LABELS: dict[str, str] = {
    "photorealistic": "photorealistic photography",
    "editorial": "editorial illustration",
    "cartoon": "cartoon illustration",
    "minimalist": "minimalist design",
    "vintage": "vintage illustration",
    "abstract": "abstract art",
}


def generate_image_prompt(
    piece: GeneratedPiece,
    config: ImagePromptConfig,
    *,
    client: LLMClient | None = None,
    settings: Settings | None = None,
) -> str:
    """Generate a DALL-E / Midjourney / Stable Diffusion prompt for a piece header image.

    Uses the LLM to derive a vivid, topic-specific subject description from the
    piece's title, topic summary, subject domain, and key entities. Style directives,
    aspect ratio, and pixel dimensions are appended deterministically from the config.

    The prompt intentionally omits any writer name to avoid creating visual style
    associations with real people.

    Args:
        piece: The generated opinion piece providing topic context.
        config: Image prompt configuration specifying style, platform, and custom keywords.
        client: Optional LLM client for dependency injection (used in tests).
        settings: Optional settings override for LLM client creation.

    Returns:
        A non-empty prompt string ready for use with DALL-E 3, Midjourney v6, or
        Stable Diffusion XL. Includes aspect ratio and pixel dimensions as suffix.

    Raises:
        RuntimeError: If the LLM API call fails.
    """
    if client is None:
        client = create_llm_client(settings)

    topic = piece.topic
    width, height = config.dimensions
    aspect_ratio = config.aspect_ratio
    style_directive = _STYLE_DIRECTIVES[config.style]
    style_label = _STYLE_LABELS[config.style]

    # Build user prompt to elicit a subject description from the LLM
    user_prompt_parts = [
        f"Generate a vivid subject description for a header image for an opinion piece.\n\n",
        f"Topic title: {topic.title}\n",
        f"Topic summary: {topic.summary}\n",
        f"Subject domain: {topic.subject_domain}\n",
    ]
    if topic.key_entities:
        user_prompt_parts.append(
            f"Key entities (for context only, do not name people): "
            f"{', '.join(topic.key_entities)}\n"
        )
    user_prompt_parts.append(
        f"\nProduce a one-to-two sentence subject description for a {style_label} "
        f"header image. Do not include any person's name. Do not include text overlays."
    )
    user_prompt = "".join(user_prompt_parts)

    try:
        subject_description = client.generate(
            system_prompt=_IMAGE_PROMPT_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=200,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate image prompt: {exc}. "
            "Check your API key configuration and network connection."
        ) from exc

    subject_description = subject_description.strip()

    # Assemble final prompt deterministically
    parts: list[str] = [subject_description]

    # Style directives
    parts.append(f"{style_directive}.")

    # Custom keywords (if any)
    if config.custom_keywords:
        parts.append(", ".join(config.custom_keywords) + ".")

    # Composition guidance
    parts.append("Clean composition, no text overlays.")

    # Style label line
    parts.append(f"Style: {style_label}.")

    # Aspect ratio and dimensions
    parts.append(f"Aspect ratio: {aspect_ratio}.")
    parts.append(f"Size: {width}x{height} px.")

    return " ".join(parts)
