"""Minimal entrypoint placeholder for the PRD-aligned pipeline."""

from __future__ import annotations


def main() -> None:
    print("Pipeline modules are ready.")
    print("Use:")
    print("  python -m processing.parse_pdfs")
    print("  python -m processing.build_chunks")
    print("  python -m embedding.embed_chunks")


if __name__ == "__main__":
    main()
