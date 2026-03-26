========================================================================
    OPINIONFORGE -- EXTENDED DOCUMENTATION
========================================================================

    Version 1.0.0  |  Python 3.11+  |  MIT License

    An editorial craft engine for generating publication-ready
    opinion pieces with precise rhetorical control.


========================================================================
    TABLE OF CONTENTS
========================================================================

    1.  Overview
    2.  Installation
    3.  Quick Start Tutorial
    4.  CLI Reference
    5.  The 12 Rhetorical Modes
    6.  Mode Blending
    7.  Stance and Intensity
    8.  Length Presets
    9.  Topic Ingestion
    10. Source Research Pipeline
    11. Similarity Screening
    12. Export Formats
    13. Image Prompt Generation
    14. Configuration Reference
    15. Architecture Overview
    16. API Reference
    17. Exit Codes
    18. Troubleshooting
    19. Running Tests
    20. Mandatory Disclaimer


========================================================================
    1. OVERVIEW
========================================================================

OpinionForge is an AI-powered editorial craft engine. It takes a topic
(provided as text, URL, or file) and generates an opinion piece using
a selected rhetorical mode. You control three independent dimensions:

    - Mode -- the rhetorical approach (12 built-in modes across four
      categories)
    - Stance -- argumentative emphasis direction from -100
      (equity-focused) to +100 (liberty-focused)
    - Intensity -- rhetorical heat from 0.0 (measured, deliberative)
      to 1.0 (maximum conviction)

OpinionForge is not a voice-cloning or writer-imitation tool. It uses
abstract rhetorical modes that define argumentative structure, prose
style, and rhetorical techniques without replicating the style or
identity of any real individual.

Every generated piece includes a mandatory, non-removable disclaimer
identifying it as AI-assisted content.


========================================================================
    2. INSTALLATION
========================================================================

    pip install opinionforge

OpinionForge requires Python 3.11 or later.

DEPENDENCIES
------------------------------------------------------------------------

OpinionForge installs the following dependencies automatically:

    - typer>=0.9.0 -- CLI framework
    - rich>=13.0 -- Terminal formatting
    - pydantic>=2.0 -- Data validation
    - pydantic-settings>=2.0 -- Environment-based configuration
    - httpx>=0.24 -- HTTP client
    - trafilatura>=1.6 -- Article content extraction
    - anthropic>=0.30 -- Anthropic API client
    - openai>=1.0 -- OpenAI API client
    - tavily-python>=0.3 -- Tavily search API client
    - pyyaml>=6.0 -- YAML profile parsing
    - python-dotenv>=1.0 -- .env file loading

DEVELOPMENT DEPENDENCIES
------------------------------------------------------------------------

    pip install -e ".[dev]"

Development extras include pytest, pytest-asyncio, pytest-mock, ruff,
mypy, and coverage.

API KEY SETUP
------------------------------------------------------------------------

Before generating content, configure at least one LLM API key:

    # For Anthropic (default provider):
    export ANTHROPIC_API_KEY=your-key-here

    # Or for OpenAI:
    export OPENAI_API_KEY=your-key-here
    export OPINIONFORGE_LLM_PROVIDER=openai

For source research, also configure a search API key:

    export OPINIONFORGE_SEARCH_API_KEY=your-tavily-key-here

You can also place these in a .env file in your working directory.


========================================================================
    3. QUICK START TUTORIAL
========================================================================

1. Generate your first piece:

    opinionforge write "The rise of algorithmic governance" \
        --mode polemical --no-preview

2. Preview the tone before generating:

    opinionforge write "Climate policy tradeoffs" \
        --mode analytical --stance -20 --intensity 0.4

   Without --no-preview, OpinionForge shows a 2-3 sentence tone
   preview and asks for confirmation before generating the full piece.

3. Use a standalone tone preview:

    opinionforge preview "The attention economy" \
        --mode satirical --intensity 0.9

4. Blend two modes:

    opinionforge write "Immigration policy" \
        --mode polemical:60,analytical:40 --no-preview

   Up to three modes can be blended; weights must sum to 100.

5. Ingest from a URL:

    opinionforge write --url "https://example.com/article" \
        --mode forensic --no-preview

6. Ingest from a file:

    opinionforge write --file ./my-article.txt \
        --mode narrative --no-preview

7. Export to a platform format:

    opinionforge write "The state of journalism" \
        --mode oratorical --export substack --no-preview

8. Generate with a header image prompt:

    opinionforge write "Tech regulation" \
        --mode analytical --image-prompt \
        --image-style photorealistic \
        --image-platform medium --no-preview

9. Save output to a file:

    opinionforge write "Housing policy" \
        --mode data_driven --output piece.md --no-preview

10. View all available modes:

    opinionforge modes
    opinionforge modes --category literary
    opinionforge modes --detail forensic


========================================================================
    4. CLI REFERENCE
========================================================================

WRITE -- Generate an opinion piece
------------------------------------------------------------------------

    opinionforge write [TOPIC] [OPTIONS]

Runs the full pipeline: topic ingestion, mode loading, stance
adjustment, tone preview, source research, and generation. Optionally
exports to a platform format and generates an image prompt.

Arguments:
    TOPIC               Topic text (optional if --url or --file given)

Options:
    --mode, -m          Rhetorical mode or blend
                        Default: analytical
    --stance, -s        Direction: -100 to +100
                        Default: 0
    --intensity, -i     Rhetorical heat: 0.0 to 1.0
                        Default: 0.5
    --length, -l        Preset or word count (200-8000)
                        Default: standard
    --url               Ingest topic from a URL
    --file, -f          Ingest topic from a local file
    --no-preview        Skip tone preview
    --research /
      --no-research     Enable/disable source research
                        Default: --research
    --output, -o        Write output to a file
    --verbose           Show progress details
    --export            Format: substack, medium, wordpress, twitter
    --image-prompt      Generate header image prompt
    --image-platform    Platform for image dimensions
                        Default: substack
                        Options: substack, medium, wordpress,
                                 facebook, twitter, instagram
    --image-style       Visual style for image prompt
                        Default: editorial
                        Options: photorealistic, editorial, cartoon,
                                 minimalist, vintage, abstract

Deprecated flags (produce error messages):
    --voice             Removed; use --mode instead
    --spectrum          Removed; use --stance instead
    --no-disclaimer     Removed; disclaimer is mandatory

PREVIEW -- Generate a tone preview
------------------------------------------------------------------------

    opinionforge preview [TOPIC] [OPTIONS]

Produces a 2-3 sentence preview in the selected rhetorical mode
without performing full research or generation.

Options:
    --mode, -m          Rhetorical mode or blend (default: analytical)
    --stance, -s        Direction: -100 to +100 (default: 0)
    --intensity, -i     Rhetorical heat: 0.0 to 1.0 (default: 0.5)
    --url               Ingest topic from a URL
    --file, -f          Ingest topic from a local file

MODES -- List rhetorical modes
------------------------------------------------------------------------

    opinionforge modes [OPTIONS]

Options:
    --search, -s        Filter modes by search query
    --category, -c      Filter modes by category
    --detail, -d        Show full detail for a mode

EXPORT -- Export a previously generated piece
------------------------------------------------------------------------

    opinionforge export PIECE_ID --format FORMAT

Generation history storage is a Phase 3 feature not yet implemented.
Use the --export flag on the write command instead.

Options:
    --format, -f        Required. substack, medium, wordpress, twitter

CONFIG -- Show or modify configuration
------------------------------------------------------------------------

    opinionforge config [OPTIONS]

Without options, displays all current settings with API keys masked.

Options:
    --set KEY           Set a configuration value (session-only).
                        Settable keys: opinionforge_llm_provider,
                        opinionforge_search_provider

API keys cannot be set via CLI for security reasons.


========================================================================
    5. THE 12 RHETORICAL MODES
========================================================================

CONFRONTATIONAL
------------------------------------------------------------------------

    polemical       Polemical
                    Combative moral urgency. Names targets, indicts
                    positions, demands the reader choose sides.

    populist        Populist
                    Plain-spoken, ground-level clarity grounded in
                    concrete human detail against abstract power.

    provocative     Provocative
                    Deliberately unsettling claims designed to
                    challenge conventional thinking.

INVESTIGATIVE
------------------------------------------------------------------------

    forensic        Forensic
                    Evidence-first argumentation. Treats opinion as
                    a prosecutorial brief from primary sources.

    data_driven     Data-Driven
                    Quantitative framing where statistics anchor
                    every claim.

DELIBERATIVE
------------------------------------------------------------------------

    analytical      Analytical
                    Rigorous evidence-first reasoning with measured,
                    authoritative tone.

    dialectical     Dialectical
                    Thesis-antithesis-synthesis reasoning that
                    acknowledges genuine complexity.

    measured        Measured
                    Calm, earned authority through restraint and
                    precision. Claims only what evidence warrants.

LITERARY
------------------------------------------------------------------------

    satirical       Satirical
                    Irony, wit, and exaggeration to expose absurdity.

    oratorical      Oratorical
                    Grand, elevated classical rhetoric designed to
                    be spoken aloud.

    narrative       Narrative
                    Story-led opinion through scene and detail.
                    Argument emerges from the story.

    aphoristic      Aphoristic
                    Compressed, epigram-driven writing. Every
                    sentence carries maximum weight.

Each mode profile defines prose patterns, rhetorical devices,
vocabulary register, argument structure, signature patterns,
suppressed phrases, a system prompt fragment, and few-shot examples.

Use "opinionforge modes --detail <mode-id>" for full profile details.


========================================================================
    6. MODE BLENDING
========================================================================

Blend up to three modes with colon-separated weights summing to 100:

    # 60% polemical, 40% analytical
    opinionforge write "Topic" \
        --mode polemical:60,analytical:40

    # Three-way blend
    opinionforge write "Topic" \
        --mode forensic:50,narrative:30,measured:20

A single mode without weights is treated as 100%. In blended mode,
the dominant mode drives overall structure while secondary modes
contribute their rhetorical qualities proportionally.


========================================================================
    7. STANCE AND INTENSITY
========================================================================

STANCE (--stance, -s)
------------------------------------------------------------------------

Argumentative emphasis direction on a scale from -100 to +100:

    -100 to -50     Strongly equity-focused
    -49 to -25      Equity-leaning
    -24 to +25      Balanced
    +26 to +49      Liberty-leaning
    +50 to +100     Strongly liberty-focused

Stance modifies five dimensions:

    1. Argument selection -- which arguments get primary emphasis
    2. Framing -- how facts and statistics are contextualized
    3. Source preference -- which evidence types are prioritized
    4. Rhetorical intensity -- how forcefully arguments are made
    5. Counterargument handling -- how opposing views are treated

INTENSITY (--intensity, -i)
------------------------------------------------------------------------

Rhetorical heat, independent of stance direction:

    0.0 - 0.19      Measured, deliberative; acknowledges complexity
    0.2 - 0.49      Moderate conviction; analytical confidence
    0.5 - 0.79      Strong conviction; vivid and pointed rhetoric
    0.8 - 1.0       Maximum conviction; uncompromising language


========================================================================
    8. LENGTH PRESETS
========================================================================

Use --length with a preset name or a custom word count (200-8000):

    Preset        Word Count
    ----------    ----------
    short         500
    standard      750
    long          1200
    essay         2500
    feature       5000

The generator targets the specified count within a 10% tolerance.

Custom word counts are accepted as integers:

    opinionforge write "Topic" --length 1500 --no-preview


========================================================================
    9. TOPIC INGESTION
========================================================================

OpinionForge accepts topics through three input methods:

    - Plain text:   opinionforge write "Your topic text"
    - URL:          opinionforge write --url "https://example.com/article"
    - File:         opinionforge write --file ./my-topic.txt

For all methods, OpinionForge produces a normalized TopicContext with:

    - Title (from first line or sentence)
    - Summary (first 2-3 sentences)
    - Key claims (sentences with statistical/factual patterns)
    - Key entities (capitalized multi-word sequences)
    - Subject domain (keyword-based classification)

URL content is fetched with httpx and extracted with trafilatura.
Source text is truncated to 10,000 words if it exceeds that limit.


========================================================================
    10. SOURCE RESEARCH PIPELINE
========================================================================

When --research is enabled (the default), OpinionForge conducts
automated web research before generating:

    1. Query generation -- 5-8 search queries covering factual
       background, developments, expert opinions, statistics, and
       counterarguments
    2. Web search -- using the configured provider (Tavily, Brave,
       or SerpAPI)
    3. Source scoring:
       - Relevance (60% weight) -- keyword overlap with the topic
       - Credibility (40% weight) -- domain reputation from a
         built-in map of 30+ known publications
       - Spectrum weight -- slight adjustment based on source
         political lean vs. requested stance
    4. Content fetching -- top results fetched with trafilatura
    5. Claim extraction -- sentences with statistics, studies,
       data, and expert references
    6. Citation formatting -- claim text, source name, URL,
       access date

Minimum sources scale with target length: 3 for short pieces
(800 words or fewer), 8 for features (2500+ words).


========================================================================
    11. SIMILARITY SCREENING
========================================================================

Every generated piece is screened before delivery. Four checks:

    1. Verbatim matching -- 5-word n-gram matching against research
       source texts (case-sensitive)
    2. Near-verbatim matching -- 6-word n-gram matching after
       normalization (lowercase, punctuation stripped)
    3. Suppressed phrase matching -- case-insensitive check against
       global and per-mode suppressed phrase lists
    4. Structural fingerprint scoring -- cosine similarity of
       sentence-level syllable-count sequences against known
       patterns (blocking threshold: 0.85)

If violations are found, the screening module performs up to 2
automatic rewrite passes. If the piece still fails, generation
is blocked (exit code 8).


========================================================================
    12. EXPORT FORMATS
========================================================================

Use --export with the write command for platform-optimized output.

SUBSTACK (--export substack)
    Clean ATX-style markdown. No raw HTML tags. Includes title,
    body, sources appendix, disclaimer, and optional image prompt.

MEDIUM (--export medium)
    Medium-compatible markdown with DROP CAP marker, demoted
    heading levels, and pull quote blockquotes for longer pieces.

WORDPRESS (--export wordpress)
    Gutenberg block-editor markup with wp:paragraph and wp:heading
    blocks, featured image placeholder, SEO meta description
    comment, and disclaimer with opinionforge-disclaimer CSS class.

TWITTER/X (--export twitter)
    Numbered tweet thread. Tweet 1: title + opening hook.
    Middle tweets: key argument paragraphs. Final tweet: disclaimer.
    No tweet exceeds 280 characters. Thread length: 5 to 15 tweets.


========================================================================
    13. IMAGE PROMPT GENERATION
========================================================================

Use --image-prompt with the write command to generate a header
image prompt suitable for DALL-E, Midjourney, or Stable Diffusion.

IMAGE STYLES
------------------------------------------------------------------------

    photorealistic   Documentary photograph with natural lighting
    editorial        Professional editorial art, bold composition
    cartoon          Illustrated style, bold outlines, vibrant colors
    minimalist       Clean composition, geometric simplicity
    vintage          Retro aesthetic, aged texture
    abstract         Non-representational conceptual imagery

IMAGE PLATFORM DIMENSIONS
------------------------------------------------------------------------

    Platform      Aspect Ratio     Dimensions (px)
    ----------    ------------     ---------------
    substack      16:9             1456 x 819
    medium        16:9             1400 x 788
    wordpress     16:9             1200 x 675
    facebook      1.91:1           1200 x 628
    twitter       16:9             1600 x 900
    instagram     1:1              1080 x 1080


========================================================================
    14. CONFIGURATION REFERENCE
========================================================================

All configuration is via environment variables or a .env file.

    Variable                      Default       Description
    ----------------------------  -----------   ---------------------
    ANTHROPIC_API_KEY             (none)        Anthropic API key
    OPENAI_API_KEY                (none)        OpenAI API key
    OPINIONFORGE_LLM_PROVIDER    anthropic     anthropic or openai
    OPINIONFORGE_SEARCH_API_KEY  (none)        Search provider key
    OPINIONFORGE_SEARCH_PROVIDER tavily        tavily, brave, serpapi

LLM Models:
    - Anthropic: claude-sonnet-4-20250514
    - OpenAI: gpt-4o

View current config:

    opinionforge config

Set a value (session-only):

    opinionforge config --set opinionforge_llm_provider openai


========================================================================
    15. ARCHITECTURE OVERVIEW
========================================================================

    opinionforge/
        __init__.py             Package init, version = "1.0.0"
        __main__.py             Entry point: python -m opinionforge
        cli.py                  Typer CLI application (5 commands)
        config.py               Settings via pydantic-settings
        core/
            generator.py        Main LLM generation engine
            mode_engine.py      Mode loading, blending, fuzzy-match
            stance.py           Stance modifier (5 dimensions)
            preview.py          Tone preview + LLM clients
            research.py         Source research pipeline
            similarity.py       Similarity screening (4 checks)
            length.py           Length presets and validation
            topic.py            Topic ingestion (text/URL/file)
            image_prompt.py     Image prompt generation
        exporters/
            base.py             Abstract base exporter
            substack.py         Substack markdown exporter
            medium.py           Medium markdown exporter
            wordpress.py        WordPress Gutenberg block exporter
            twitter.py          Twitter/X thread exporter
        modes/
            __init__.py         load_mode() and list_modes()
            categories.yaml     Category-to-mode mapping
            profiles/           12 YAML mode profiles
        models/
            config.py           ModeBlendConfig, StanceConfig,
                                ImagePromptConfig
            mode.py             ModeProfile and sub-models
            piece.py            GeneratedPiece, ScreeningResult,
                                SourceCitation
            topic.py            TopicContext
        utils/
            fetcher.py          URL fetching + trafilatura extraction
            search.py           Search clients (Tavily, Brave, SerpAPI)
            text.py             word_count, truncate, citations
        data/
            suppressed_phrases.yaml
            structural_fingerprints.yaml

GENERATION PIPELINE
------------------------------------------------------------------------

The "write" command executes these steps in order:

    1.  Input validation (mode, stance, intensity, length, export,
        image config)
    2.  Topic ingestion (text/URL/file to TopicContext)
    3.  Mode loading (YAML profiles, blend if multiple)
    4.  Stance modification (append stance instructions to prompt)
    5.  Tone preview (unless --no-preview; interactive confirmation)
    6.  Source research (unless --no-research; multi-query search)
    7.  Generation (compose system prompt, call LLM, parse output)
    8.  Similarity screening (4 checks + up to 2 rewrite passes)
    9.  Image prompt generation (if --image-prompt)
    10. Output (format with disclaimer, export, write to file/stdout)


========================================================================
    16. API REFERENCE
========================================================================

MODELS (from opinionforge.models)
------------------------------------------------------------------------

    TopicContext        Normalized topic: title, summary, key_claims,
                        key_entities, subject_domain, source_url,
                        source_text, fetched_at

    ModeBlendConfig     List of (mode_id, weight) tuples.
                        1-3 modes, weights sum to 100.

    StanceConfig        position (-100 to +100), intensity (0.0 to 1.0).
                        Property: label (non-political descriptor)

    ImagePromptConfig   style, platform, custom_keywords.
                        Properties: aspect_ratio, dimensions

    ModeProfile         Full profile: id, display_name, description,
                        category, prose_patterns, rhetorical_devices,
                        vocabulary_register, argument_structure,
                        signature_patterns, suppressed_phrases,
                        system_prompt_fragment, few_shot_examples

    GeneratedPiece      Complete output: id, created_at, topic,
                        mode_config, stance, target/actual length,
                        title, body, preview_text, sources,
                        research_queries, image_prompt,
                        image_platform, disclaimer, screening_result

    ScreeningResult     passed, verbatim_matches,
                        near_verbatim_matches,
                        suppressed_phrase_matches,
                        structural_fingerprint_score,
                        rewrite_iterations, warning

    SourceCitation      claim, source_name, source_url, accessed_at,
                        political_lean, credibility_score,
                        recency_score

LLM CLIENT PROTOCOL (opinionforge.core.preview)
------------------------------------------------------------------------

    LLMClient.generate(system_prompt, user_prompt, max_tokens) -> str

Implementations:
    - AnthropicLLMClient (claude-sonnet-4-20250514)
    - OpenAILLMClient (gpt-4o)

KEY FUNCTIONS
------------------------------------------------------------------------

    opinionforge.core.generator.generate_piece()
        Orchestrates full generation

    opinionforge.core.mode_engine.blend_modes()
        Produces blended prompt from ModeBlendConfig

    opinionforge.core.stance.apply_stance()
        Modifies mode prompt with stance instructions

    opinionforge.core.research.research_topic()
        Conducts multi-query research

    opinionforge.core.similarity.screen_output()
        Screens output for originality

    opinionforge.core.topic.ingest_text()
    opinionforge.core.topic.ingest_url()
    opinionforge.core.topic.ingest_file()
        Topic normalization

    opinionforge.core.image_prompt.generate_image_prompt()
        Generates image prompts

    opinionforge.exporters.export()
        Dispatches to platform exporters


========================================================================
    17. EXIT CODES
========================================================================

    Code    Meaning
    ----    ------------------------------------------
    0       Success
    1       General / generation error
    2       Invalid arguments
    3       Network error
    4       Mode not found
    5       API key not configured
    6       Rate limit exceeded
    7       Content policy violation
    8       Similarity screening failure


========================================================================
    18. TROUBLESHOOTING
========================================================================

"ANTHROPIC_API_KEY is not set"
    Set your LLM API key as an environment variable or in a .env file:
        export ANTHROPIC_API_KEY=your-key-here

"OPINIONFORGE_SEARCH_API_KEY is not set"
    Set a search API key or disable research:
        export OPINIONFORGE_SEARCH_API_KEY=your-tavily-key-here
    Or:
        opinionforge write "Topic" --no-research --no-preview

"Mode 'X' not found"
    The mode ID does not match any installed profile. OpinionForge
    provides fuzzy-match suggestions. Check available modes:
        opinionforge modes

"Blend weights must sum to 100"
    When blending modes, weights must total exactly 100.

"Word count X is outside the allowed range"
    Custom word counts must be between 200 and 8000.

"--voice is not a valid option"
    Removed in v1.0.0. Use --mode instead.

"--spectrum is not a valid option"
    Removed in v1.0.0. Use --stance instead.

"--no-disclaimer is not a valid option"
    The disclaimer is mandatory in v1.0.0.

"Similarity screening failed -- output blocked"
    The piece failed originality screening after rewrite attempts.
    Try regenerating with a different mode, stance, or topic.

Generation is slow
    The full pipeline requires multiple API calls. To speed up:
        - Use --no-research to skip web research
        - Use --no-preview to skip the tone preview step


========================================================================
    19. RUNNING TESTS
========================================================================

    # Install dev dependencies
    pip install -e ".[dev]"

    # Run the full test suite (excluding slow LLM tests)
    pytest tests/ -m "not slow"

The test suite contains 1040 tests covering unit tests, integration
tests, and CLI integration tests. All tests use mocked LLM and
search clients -- no real API calls.

Test markers:
    - slow: tests that make real API calls (excluded by default)
    - confusability: confusability regression suite


========================================================================
    20. MANDATORY DISCLAIMER
========================================================================

Every piece generated by OpinionForge includes this mandatory
disclaimer:

    "This piece was generated with AI-assisted rhetorical controls.
    It is original content and is not written by, endorsed by, or
    affiliated with any real person."

The disclaimer cannot be suppressed. The --no-disclaimer flag was
removed in v1.0.0 and produces an error if used. This policy exists
because OpinionForge generates original AI content with defined
rhetorical character, and readers deserve to know what they are
reading.


========================================================================
    END OF DOCUMENT
========================================================================
