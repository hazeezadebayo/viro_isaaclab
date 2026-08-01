#!/usr/bin/env python3
"""Lightweight launcher for playing a saved checkpoint."""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(f"Play entrypoint for {root.name}")
    print("Replace this stub with your IsaacLab play command.")


if __name__ == "__main__":
    main()
