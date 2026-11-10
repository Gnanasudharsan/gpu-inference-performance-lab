"""Triton-free tuning helpers (importable and testable without Triton or a GPU)."""

from __future__ import annotations


def autotune_m_key(m: int) -> int:
    """Autotune cache key for the row count: the next power of two.

    Keying on the exact M re-runs the autotuner (every config, benchmarked) for each new row
    count. Under dynamic batching that happened mid-serving on a T4: row counts like 3, 5, 6
    each triggered seconds of tuning (dynamic-policy p99 of 12.7 s). Power-of-two buckets
    match the serving calibration buckets, so warm-up covers every key.
    """
    return 1 if m <= 1 else 1 << (m - 1).bit_length()
