"""Low-level readers for .bin+.idx and document-level .arecord formats.

Both expose the same document-level interface so upstream indexing code is
backend-agnostic: __len__, __getitem__, get(idx, offset, length), doc_lengths.
"""

from __future__ import annotations

import itertools
import struct
from pathlib import Path

import numpy as np


# --- .idx format constants ---

_IDX_MAGIC = b"MMIDIDX\x00\x00"
_IDX_HEADER_SIZE = 9 + 8 + 1 + 8 + 8  # magic(9) + version(8) + dtype(1) + seq_count(8) + doc_count(8) = 34

_DTYPE_CODE_TO_NUMPY = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}

_NUMPY_TO_DTYPE_CODE = {v: k for k, v in _DTYPE_CODE_TO_NUMPY.items()}


class MMapIndexedDataset:
    """Read-only access to .bin+.idx mmap files.

    Documents are contiguous runs of sequences delimited by document_indices.
    This class presents a document-level view: __getitem__(i) returns all
    tokens in document i as a 1-D numpy array.
    """

    def __init__(self, path_prefix: str) -> None:
        """Open {path_prefix}.idx and {path_prefix}.bin."""
        idx_path = Path(f"{path_prefix}.idx")
        bin_path = Path(f"{path_prefix}.bin")

        if not idx_path.exists():
            raise FileNotFoundError(f"Index file not found: {idx_path}")

        # Parse .idx header
        with open(idx_path, "rb") as f:
            magic = f.read(9)
            if magic != _IDX_MAGIC:
                raise ValueError(f"Invalid .idx magic: {magic!r}")

            version = struct.unpack("<Q", f.read(8))[0]
            if version != 1:
                raise ValueError(f"Unsupported .idx version: {version}")

            dtype_code = struct.unpack("<B", f.read(1))[0]
            if dtype_code not in _DTYPE_CODE_TO_NUMPY:
                raise ValueError(f"Unknown dtype code: {dtype_code}")
            self._dtype = _DTYPE_CODE_TO_NUMPY[dtype_code]
            self._dtype_size = np.dtype(self._dtype).itemsize

            seq_count = struct.unpack("<Q", f.read(8))[0]
            doc_count = struct.unpack("<Q", f.read(8))[0]

        # Memory-map the index arrays after the 34-byte header
        idx_mmap = np.memmap(idx_path, mode="r", order="C")
        offset = _IDX_HEADER_SIZE

        # sequence_lengths: int32[seq_count]
        sl_bytes = seq_count * 4
        self._sequence_lengths = np.frombuffer(
            idx_mmap, dtype=np.int32, count=seq_count, offset=offset
        )
        offset += sl_bytes

        # sequence_pointers: int64[seq_count]
        sp_bytes = seq_count * 8
        self._sequence_pointers = np.frombuffer(
            idx_mmap, dtype=np.int64, count=seq_count, offset=offset
        )
        offset += sp_bytes

        # document_indices: int64[doc_count]
        self._document_indices = np.frombuffer(
            idx_mmap, dtype=np.int64, count=doc_count, offset=offset
        )

        # Keep reference to prevent GC
        self._idx_mmap = idx_mmap

        # Memory-map the .bin file(s)
        if bin_path.exists():
            # Single file (backward compatible)
            self._bin_parts = [np.memmap(bin_path, mode="r", dtype=np.uint8)]
            self._part_cumulative = np.array([0, len(self._bin_parts[0])], dtype=np.int64)
        elif Path(f"{path_prefix}.bin.00000").exists():
            # Multi-part: discover prefix.bin.00000, prefix.bin.00001, ...
            self._bin_parts = []
            cumulative = [0]
            for part_idx in itertools.count():
                p = Path(f"{path_prefix}.bin.{part_idx:05d}")
                if not p.exists():
                    break
                mm = np.memmap(p, mode="r", dtype=np.uint8)
                self._bin_parts.append(mm)
                cumulative.append(cumulative[-1] + len(mm))
            self._part_cumulative = np.array(cumulative, dtype=np.int64)
        else:
            raise FileNotFoundError(
                f"Data file not found: {bin_path} (also tried {path_prefix}.bin.00000)"
            )

        # Number of documents: document_indices is [0, end_doc0, end_doc1, ...]
        # so num_docs = len(document_indices) - 1
        self._num_docs = len(self._document_indices) - 1

    def __len__(self) -> int:
        return self._num_docs

    def __getitem__(self, idx: int) -> np.ndarray:
        """Return all tokens in document idx as a 1-D array."""
        if idx < 0:
            idx += self._num_docs
        if idx < 0 or idx >= self._num_docs:
            raise IndexError(f"Document index {idx} out of range [0, {self._num_docs})")

        seq_start = int(self._document_indices[idx])
        seq_end = int(self._document_indices[idx + 1])

        # Concatenate all sequences in this document
        if seq_end - seq_start == 1:
            return self._read_sequence(seq_start)

        parts = []
        for seq_idx in range(seq_start, seq_end):
            parts.append(self._read_sequence(seq_idx))
        return np.concatenate(parts)

    def get(self, idx: int, offset: int = 0, length: int | None = None) -> np.ndarray:
        """Get a slice of tokens from document idx."""
        tokens = self[idx]
        if length is None:
            return tokens[offset:]
        return tokens[offset : offset + length]

    @property
    def doc_lengths(self) -> np.ndarray:
        """Return int32 array of per-document token counts."""
        cumsum = np.zeros(len(self._sequence_lengths) + 1, dtype=np.int64)
        np.cumsum(self._sequence_lengths, out=cumsum[1:])
        starts = self._document_indices[:-1].astype(np.intp)
        ends = self._document_indices[1:].astype(np.intp)
        return (cumsum[ends] - cumsum[starts]).astype(np.int32)

    def _read_sequence(self, seq_idx: int) -> np.ndarray:
        """Read a single sequence from the .bin file(s)."""
        pointer = int(self._sequence_pointers[seq_idx])
        length = int(self._sequence_lengths[seq_idx])

        if len(self._bin_parts) == 1:
            # Fast path: single file (no regression for common case)
            return np.frombuffer(
                self._bin_parts[0],
                dtype=self._dtype,
                count=length,
                offset=pointer,
            ).copy()

        # Multi-part: binary search for the correct part
        part_idx = int(np.searchsorted(self._part_cumulative, pointer, side="right")) - 1
        local_offset = pointer - int(self._part_cumulative[part_idx])
        return np.frombuffer(
            self._bin_parts[part_idx],
            dtype=self._dtype,
            count=length,
            offset=local_offset,
        ).copy()


class ArrayRecordDocDataset:
    """Read-only access to document-level .arecord files.

    Each record = one document's token IDs (variable length, serialized np.array).
    Companion .doc_lengths.npy provides instant doc length lookup.
    """

    def __init__(self, path: str) -> None:
        """Open .arecord file(s) + load .doc_lengths.npy."""
        from array_record.python.array_record_module import ArrayRecordReader

        p = Path(path)

        if p.is_dir():
            # Directory: glob all .arecord files, sorted
            arecord_files = sorted(p.glob("*.arecord"))
            if not arecord_files:
                raise FileNotFoundError(f"No .arecord files found in {p}")
            doc_lengths_path = p / "doc_lengths.npy"
        else:
            # Single file
            arecord_files = [p]
            doc_lengths_path = p.parent / "doc_lengths.npy"

        if not doc_lengths_path.exists():
            raise FileNotFoundError(
                f"Companion doc_lengths.npy not found: {doc_lengths_path}"
            )

        self._doc_lengths_arr = np.load(doc_lengths_path).astype(np.int32)

        # Open readers and build shard mapping
        self._readers: list[ArrayRecordReader] = []
        self._shard_offsets: list[int] = []  # cumulative doc count per shard
        total = 0
        for f in arecord_files:
            reader = ArrayRecordReader(str(f))
            self._readers.append(reader)
            self._shard_offsets.append(total)
            total += reader.num_records()

        self._total_docs = total

        if len(self._doc_lengths_arr) != self._total_docs:
            raise ValueError(
                f"doc_lengths.npy has {len(self._doc_lengths_arr)} entries "
                f"but arecord files have {self._total_docs} records"
            )

        # Determine token dtype from first record
        first_record = self._readers[0].read([0])[0]
        # Try uint16 first (common for vocab < 65536), fall back to int32
        n_bytes = len(first_record)
        if n_bytes == self._doc_lengths_arr[0] * 2:
            self._token_dtype = np.uint16
        elif n_bytes == self._doc_lengths_arr[0] * 4:
            self._token_dtype = np.int32
        else:
            # Fallback: try int32
            self._token_dtype = np.int32

    def __len__(self) -> int:
        return self._total_docs

    def __getitem__(self, idx: int) -> np.ndarray:
        """Return all tokens in document idx as a 1-D array."""
        if idx < 0:
            idx += self._total_docs
        if idx < 0 or idx >= self._total_docs:
            raise IndexError(f"Document index {idx} out of range [0, {self._total_docs})")

        # Find which shard
        shard_id = self._find_shard(idx)
        local_idx = idx - self._shard_offsets[shard_id]

        record = self._readers[shard_id].read([local_idx])[0]
        return np.frombuffer(record, dtype=self._token_dtype).copy()

    def get(self, idx: int, offset: int = 0, length: int | None = None) -> np.ndarray:
        """Get a slice of tokens from document idx."""
        tokens = self[idx]
        if length is None:
            return tokens[offset:]
        return tokens[offset : offset + length]

    @property
    def doc_lengths(self) -> np.ndarray:
        return self._doc_lengths_arr

    def _find_shard(self, idx: int) -> int:
        """Binary search for the shard containing global idx."""
        lo, hi = 0, len(self._shard_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._shard_offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        return lo
