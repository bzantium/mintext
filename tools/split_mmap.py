"""Split a large .bin file into multiple parts at sequence boundaries.

The .idx file is left unchanged. The reader (MMapIndexedDataset) auto-discovers
multi-part .bin files via the naming convention: prefix.bin.00000, prefix.bin.00001, ...

Since splits happen at sequence boundaries, no sequence ever spans two parts.

Usage:
    python tools/split_mmap.py \
        --input /path/to/data_text_document \
        --max-file-size 5G

    # Custom output prefix (default: same as input)
    python tools/split_mmap.py \
        --input /path/to/data_text_document \
        --output /path/to/output_prefix \
        --max-file-size 5G
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mintext.data.indexed_dataset import MMapIndexedDataset
from mintext.utils.filesize import parse_file_size

_COPY_CHUNK = 64 * 1024 * 1024  # 64 MB


def main() -> None:
    parser = argparse.ArgumentParser(description="Split large .bin files at sequence boundaries")
    parser.add_argument("--input", required=True, help="Path prefix for .bin/.idx files (without extension)")
    parser.add_argument("--output", default=None, help="Output prefix (default: same as input)")
    parser.add_argument("--max-file-size", default="5G", help="Maximum size per part (e.g. 5G, 500M)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    max_bytes = parse_file_size(args.max_file_size)
    output_prefix = args.output or args.input

    bin_path = Path(f"{args.input}.bin")
    if not bin_path.exists():
        logging.error("Data file not found: %s", bin_path)
        sys.exit(1)

    bin_size = bin_path.stat().st_size
    if bin_size <= max_bytes:
        logging.info("File %s is %d bytes (<= %d), no split needed.", bin_path, bin_size, max_bytes)
        return

    # Open dataset to get sequence metadata
    dataset = MMapIndexedDataset(args.input)
    seq_count = len(dataset._sequence_pointers)
    dtype_size = dataset._dtype_size

    # Compute byte-end of each sequence
    seq_ends = np.empty(seq_count, dtype=np.int64)
    for i in range(seq_count):
        seq_ends[i] = int(dataset._sequence_pointers[i]) + int(dataset._sequence_lengths[i]) * dtype_size

    # Greedily assign sequences to parts
    parts: list[tuple[int, int]] = []  # (byte_start, byte_end) for each part
    part_start_byte = 0
    for i in range(seq_count):
        end_byte = int(seq_ends[i])
        part_size = end_byte - part_start_byte

        if part_size > max_bytes and part_start_byte < int(dataset._sequence_pointers[i]):
            # Current part already has content and adding this seq exceeds limit
            # Close current part at the start of this sequence
            parts.append((part_start_byte, int(dataset._sequence_pointers[i])))
            part_start_byte = int(dataset._sequence_pointers[i])

    # Final part
    parts.append((part_start_byte, bin_size))

    logging.info("Splitting %s (%d bytes) into %d parts", bin_path, bin_size, len(parts))

    # Write parts by copying byte ranges
    with open(bin_path, "rb") as src:
        for part_idx, (start, end) in enumerate(parts):
            out_path = Path(f"{output_prefix}.bin.{part_idx:05d}")
            src.seek(start)
            remaining = end - start
            with open(out_path, "wb") as dst:
                while remaining > 0:
                    chunk = src.read(min(_COPY_CHUNK, remaining))
                    dst.write(chunk)
                    remaining -= len(chunk)
            logging.info("  Part %d: %s (%d bytes)", part_idx, out_path, end - start)

    # Remove original .bin (parts replace it)
    bin_path.unlink()
    logging.info("Removed original %s. Split complete.", bin_path)


if __name__ == "__main__":
    main()
