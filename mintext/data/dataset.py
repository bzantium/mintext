"""Grain-compatible data sources with cached sample indexing.

Works with either MMapIndexedDataset or ArrayRecordDocDataset backend.
Implements document/sample indexing with cached sample boundaries in pure NumPy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import grain.python as grain
import numpy as np

from mintext.data.indexed_dataset import ArrayRecordDocDataset, MMapIndexedDataset

logger = logging.getLogger(__name__)


class DocumentDataSource(grain.RandomAccessDataSource):
    """Grain data source backed by document-level data (mmap or arecord).

    Builds document_index and sample_index per sequence length,
    cached as .npy files for instant reload on subsequent runs.
    """

    def __init__(
        self,
        data_path: str,
        data_type: str,
        seq_len: int,
        seed: int,
        num_epochs: int,
        split: tuple[float, float, float],
        split_index: int,
        cache_dir: str | None = None,
        add_extra_token: bool = True,
    ) -> None:
        super().__init__()
        self._seq_len = seq_len
        self._seed = seed
        self._add_extra_token = add_extra_token

        # Open backend
        if data_type == "mmap":
            self._backend = MMapIndexedDataset(data_path)
        elif data_type == "arecord":
            self._backend = ArrayRecordDocDataset(data_path)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        all_doc_lengths = self._backend.doc_lengths
        total_docs = len(all_doc_lengths)

        # Compute split boundaries
        split_sum = sum(split)
        fracs = [s / split_sum for s in split]
        boundaries = [0]
        for frac in fracs:
            boundaries.append(boundaries[-1] + int(round(frac * total_docs)))
        boundaries[-1] = total_docs  # ensure last split covers remainder

        doc_start = boundaries[split_index]
        doc_end = boundaries[split_index + 1]
        self._doc_ids = np.arange(doc_start, doc_end, dtype=np.int32)
        self._split_doc_lengths = all_doc_lengths[doc_start:doc_end]

        if len(self._doc_ids) == 0:
            self._sample_index = np.zeros((1, 2), dtype=np.int64)
            return

        # Build or load cached indices
        if cache_dir:
            cache_path = Path(cache_dir)
        else:
            cache_path = Path(data_path if Path(data_path).is_dir() else Path(data_path).parent) / "cache"

        self._document_index, self._sample_index = self._load_or_build_indices(
            data_path=data_path,
            seq_len=seq_len,
            seed=seed,
            num_epochs=num_epochs,
            split=(doc_start, doc_end),
            cache_dir=cache_path,
            add_extra_token=add_extra_token,
        )

    def __len__(self) -> int:
        return max(0, len(self._sample_index) - 1)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Return {"tokens": np.ndarray} of length seq_len + add_extra_token."""
        target_len = self._seq_len + self._add_extra_token

        doc_pos_start, offset_start = self._sample_index[idx]
        doc_pos_end, offset_end = self._sample_index[idx + 1]

        tokens_parts = []
        remaining = target_len

        pos = int(doc_pos_start)
        offset = int(offset_start)

        while remaining > 0 and pos < len(self._document_index):
            doc_id = int(self._document_index[pos])
            real_doc_id = int(self._doc_ids[doc_id]) if doc_id < len(self._doc_ids) else doc_id
            doc_len = int(self._split_doc_lengths[doc_id]) if doc_id < len(self._split_doc_lengths) else 0
            available = doc_len - offset

            if available <= 0:
                pos += 1
                offset = 0
                continue

            take = min(available, remaining)
            chunk = self._backend.get(real_doc_id, offset=offset, length=take)
            tokens_parts.append(chunk)
            remaining -= take

            if take >= available:
                pos += 1
                offset = 0
            else:
                offset += take

        if tokens_parts:
            tokens = np.concatenate(tokens_parts)
        else:
            tokens = np.zeros(0, dtype=np.int32)

        # Pad if needed (shouldn't happen if index is built correctly)
        if len(tokens) < target_len:
            tokens = np.pad(tokens, (0, target_len - len(tokens)), constant_values=0)

        return {"tokens": tokens[:target_len]}

    def _load_or_build_indices(
        self,
        data_path: str,
        seq_len: int,
        seed: int,
        num_epochs: int,
        split: tuple[int, int],
        cache_dir: Path,
        add_extra_token: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load cached indices or build and cache them."""
        config_str = json.dumps(
            {
                "add_extra_token": add_extra_token,
                "dataset_path": data_path,
                "sequence_length": seq_len,
                "random_seed": seed,
                "split": f"{split[0]}:{split[1]}",
                "num_epochs": num_epochs,
            },
            indent=4,
            sort_keys=True,
        )
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        cache_dir.mkdir(parents=True, exist_ok=True)
        doc_idx_path = cache_dir / f"{config_hash}-document_index.npy"
        sample_idx_path = cache_dir / f"{config_hash}-sample_index.npy"

        if doc_idx_path.exists() and sample_idx_path.exists():
            logger.info("Loading cached indices from %s", cache_dir)
            document_index = np.load(doc_idx_path)
            sample_index = np.load(sample_idx_path)
            return document_index, sample_index

        logger.info("Building indices (seq_len=%d, epochs=%d, docs=%d)...",
                     seq_len, num_epochs, len(self._doc_ids))

        rng = np.random.RandomState(seed)
        num_split_docs = len(self._doc_ids)

        document_index = _build_document_index(num_split_docs, num_epochs, rng)
        sample_index = _build_sample_index(
            document_index, self._split_doc_lengths, seq_len,
            add_extra_token=add_extra_token,
        )

        np.save(doc_idx_path, document_index)
        np.save(sample_idx_path, sample_index)
        logger.info(
            "Built %d samples, cached to %s", len(sample_index) - 1, cache_dir
        )

        return document_index, sample_index


def _build_document_index(
    num_documents: int, num_epochs: int, rng: np.random.RandomState
) -> np.ndarray:
    """Shuffled document ordering for N epochs. Returns 1-D int32 array."""
    epochs = []
    for _ in range(num_epochs):
        order = np.arange(num_documents, dtype=np.int32)
        rng.shuffle(order)
        epochs.append(order)
    return np.concatenate(epochs)


def _build_sample_index(
    document_index: np.ndarray,
    doc_lengths: np.ndarray,
    seq_len: int,
    add_extra_token: bool = True,
) -> np.ndarray:
    """Map samples to document spans.

    Returns 2-D int64 array of shape (N+1, 2).
    Each row = (doc_index_position, offset_within_doc).
    Pure NumPy implementation of sample index building.
    """
    target = seq_len + add_extra_token
    samples = [(0, 0)]
    doc_pos = 0
    offset = 0
    remaining = target

    while doc_pos < len(document_index):
        doc_id = int(document_index[doc_pos])
        doc_len = int(doc_lengths[doc_id])
        available = doc_len - offset

        if available <= 0:
            doc_pos += 1
            offset = 0
            continue

        if available >= remaining:
            offset += remaining
            samples.append((doc_pos, offset))
            remaining = target
            if offset >= doc_len:
                doc_pos += 1
                offset = 0
        else:
            remaining -= available
            doc_pos += 1
            offset = 0

    return np.array(samples, dtype=np.int64)


class BlendedDataSource(grain.RandomAccessDataSource):
    """Blends multiple DocumentDataSources with configurable weights.

    Uses greedy-by-error algorithm for proportional sampling.
    """

    def __init__(
        self,
        sources: list[DocumentDataSource],
        weights: list[float],
        size: int,
    ) -> None:
        super().__init__()
        if len(sources) != len(weights):
            raise ValueError("sources and weights must have same length")
        if not sources:
            raise ValueError("Must provide at least one source")

        self._sources = sources
        self._size = size

        # Normalize weights
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]

        # Build blending indices using greedy-by-error
        self._dataset_index = np.zeros(size, dtype=np.int32)
        self._dataset_sample_index = np.zeros(size, dtype=np.int64)

        # Track per-source counters
        num_sources = len(sources)
        counters = [0] * num_sources

        for i in range(size):
            # Pick the source whose actual proportion is furthest below target
            best_ds = 0
            best_error = -float("inf")
            for ds in range(num_sources):
                target_count = norm_weights[ds] * (i + 1)
                error = target_count - counters[ds]
                if error > best_error:
                    best_error = error
                    best_ds = ds

            self._dataset_index[i] = best_ds
            self._dataset_sample_index[i] = counters[best_ds] % len(sources[best_ds])
            counters[best_ds] += 1

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        ds_id = int(self._dataset_index[idx])
        sample_id = int(self._dataset_sample_index[idx])
        return self._sources[ds_id][sample_id]
