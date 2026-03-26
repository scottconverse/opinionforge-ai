"""Entry point for running OpinionForge as a module: python -m opinionforge."""

from opinionforge.cli import app


def main() -> None:
    """Invoke the Typer CLI application."""
    app()


if __name__ == "__main__":
    main()
