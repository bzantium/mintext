"""Parse human-readable file size strings to bytes."""

from __future__ import annotations

import re


_MULTIPLIERS = {
    "K": 1024,
    "KB": 1024,
    "M": 1024**2,
    "MB": 1024**2,
    "G": 1024**3,
    "GB": 1024**3,
    "T": 1024**4,
    "TB": 1024**4,
}

_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*(K|KB|M|MB|G|GB|T|TB)$", re.IGNORECASE)


def parse_file_size(size_str: str) -> int:
    """Parse human-readable file size to bytes. Supports K/M/G/T suffixes.

    Examples:
        parse_file_size("5G")   -> 5368709120
        parse_file_size("100M") -> 104857600
        parse_file_size("1024") -> 1024
    """
    s = size_str.strip()

    # Try raw integer first
    try:
        return int(s)
    except ValueError:
        pass

    m = _PATTERN.match(s)
    if not m:
        raise ValueError(
            f"Invalid file size: {size_str!r}. "
            f"Expected a number with optional suffix (K, M, G, T), e.g. '5G', '100M', '1024'."
        )

    value = float(m.group(1))
    suffix = m.group(2).upper()
    return int(value * _MULTIPLIERS[suffix])
