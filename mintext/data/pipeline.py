"""Grain pipeline construction for MinText training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import grain.python as grain
import jax
import numpy as np

from mintext.config import MinTextConfig
from mintext.data.dataset import BlendedDataSource, DocumentDataSource

logger = logging.getLogger(__name__)


def create_train_iterator(config: MinTextConfig, mesh: jax.sharding.Mesh) -> Iterator:
    """Build the full Grain training data pipeline.

    1. Parse data_path into (weight, path) pairs
    2. Create DocumentDataSource per path (auto-detect mmap vs arecord)
    3. If multiple: wrap in BlendedDataSource
    4. Grain DataLoader: shuffle, shard across hosts, batch, shift tokens
    5. Return iterator yielding {input_tokens: [B, S], target_tokens: [B, S]}
    """
    return _create_iterator(config, mesh, split_index=0, shuffle=True)


def create_eval_iterator(config: MinTextConfig, mesh: jax.sharding.Mesh) -> Iterator:
    """Same as train but no shuffling, uses eval split."""
    return _create_iterator(config, mesh, split_index=1, shuffle=False)


def _create_iterator(
    config: MinTextConfig,
    mesh: jax.sharding.Mesh,
    split_index: int,
    shuffle: bool,
) -> Iterator:
    """Build a Grain DataLoader iterator."""
    split = _parse_split(config.data_split)
    entries = _parse_data_path(config.data_path)

    # Create sources
    sources = []
    weights = []
    for weight, path in entries:
        data_type = _detect_data_type(path) if config.dataset_type == "auto" else config.dataset_type
        src = DocumentDataSource(
            data_path=path,
            data_type=data_type,
            seq_len=config.seq_length,
            seed=config.seed,
            num_epochs=config.num_data_epochs,
            split=split,
            split_index=split_index,
            cache_dir=config.data_cache_dir or None,
            add_extra_token=config.add_extra_token,
        )
        sources.append(src)
        weights.append(weight)

    # Blend or use single source
    if len(sources) == 1:
        data_source = sources[0]
    else:
        total_samples = sum(len(s) for s in sources)
        data_source = BlendedDataSource(sources, weights, size=total_samples)

    logger.info(
        "Data source: %d samples, %d source(s), split_index=%d",
        len(data_source), len(sources), split_index,
    )

    # Global batch size
    batch_size = config.per_device_batch_size * jax.device_count()

    # Grain sampler
    shard_options = grain.ShardByJaxProcess()
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        shard_options=shard_options,
        shuffle=shuffle,
        num_epochs=None,  # infinite
        seed=config.seed,
    )

    # Operations
    operations = [
        ShiftTokens(add_extra_token=config.add_extra_token),
        grain.Batch(batch_size=batch_size, drop_remainder=True),
    ]

    loader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=operations,
        worker_count=config.grain_worker_count,
        worker_buffer_size=config.grain_prefetch_buffer_size,
    )

    return iter(loader)


def _parse_data_path(data_path: str) -> list[tuple[float, str]]:
    """Parse 'weight1 path1 weight2 path2 ...' or plain 'path' format.

    Returns list of (weight, path) tuples.
    """
    parts = data_path.strip().split()

    if len(parts) == 1:
        return [(1.0, parts[0])]

    # Try to parse as alternating weight/path pairs
    entries = []
    i = 0
    while i < len(parts):
        try:
            weight = float(parts[i])
            if i + 1 >= len(parts):
                raise ValueError(f"Weight {weight} at position {i} has no path")
            path = parts[i + 1]
            entries.append((weight, path))
            i += 2
        except ValueError:
            if not entries:
                # First element isn't a float — treat entire string as a single path
                return [(1.0, data_path.strip())]
            raise

    return entries


def _detect_data_type(path: str) -> str:
    """Auto-detect 'mmap' (has .bin+.idx) or 'arecord' (has .arecord files)."""
    p = Path(path)

    # Check for mmap: path_prefix.bin (or multi-part .bin.00000) and path_prefix.idx
    has_bin = Path(f"{path}.bin").exists() or Path(f"{path}.bin.00000").exists()
    if Path(f"{path}.idx").exists() and has_bin:
        return "mmap"

    # Check for arecord: directory with .arecord files, or single .arecord file
    if p.is_dir():
        if list(p.glob("*.arecord")):
            return "arecord"
    elif p.suffix == ".arecord":
        return "arecord"

    raise ValueError(
        f"Cannot detect data type for '{path}'. "
        f"Expected .bin+.idx files or .arecord files/directory."
    )


def _parse_split(split_str: str) -> tuple[float, float, float]:
    """Parse 'train,val,test' proportions string."""
    parts = [float(x) for x in split_str.split(",")]
    if len(parts) != 3:
        raise ValueError(f"data_split must have 3 values, got: {split_str}")
    return (parts[0], parts[1], parts[2])


class ShiftTokens(grain.MapTransform):
    """Split tokens into input_tokens and target_tokens.

    When add_extra_token=True (default), tokens has length seq_len+1:
        input = tokens[:-1], target = tokens[1:]
    When add_extra_token=False, tokens has length seq_len:
        input = tokens, target = roll(tokens, -1)
    """

    def __init__(self, add_extra_token: bool = True):
        self._add_extra_token = add_extra_token

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        tokens = element["tokens"]
        if self._add_extra_token:
            return {
                "input_tokens": tokens[:-1],
                "target_tokens": tokens[1:],
            }
        else:
            return {
                "input_tokens": tokens,
                "target_tokens": np.roll(tokens, -1),
            }
