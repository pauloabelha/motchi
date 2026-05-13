"""Small console logging helpers for Motchi scripts."""

from __future__ import annotations


def info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def error(message: str) -> None:
    print(f"[ERROR] {message}", flush=True)

