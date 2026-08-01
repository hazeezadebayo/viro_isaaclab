#!/usr/bin/env python3
"""Lightweight launcher for starting RL training."""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(f"Training entrypoint for {root.name}")
    print("Replace this stub with your IsaacLab training command.")


if __name__ == "__main__":
    main()
