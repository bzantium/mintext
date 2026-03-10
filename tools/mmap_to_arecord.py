"""Convert .bin+.idx files to document-level .arecord format.

Each ArrayRecord record = one document's tokens (variable length).
Also writes doc_lengths.npy companion file for fast index building.

Usage:
    python tools/mmap_to_arecord.py \
        --input /path/to/data_text_document \
        --output /path/to/output_dir \
        --max-file-size 5G
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add parent dir so we can import mintext
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mintext.data.indexed_dataset import MMapIndexedDataset
from mintext.utils.filesize import parse_file_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .bin+.idx to document-level .arecord")
    parser.add_argument("--input", required=True, help="Path prefix for .bin/.idx files (without extension)")
    parser.add_argument("--output", required=True, help="Output directory for .arecord files")
    parser.add_argument("--max-file-size", default="5G", help="Maximum size per output shard (e.g. 5G, 500M)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from array_record.python.array_record_module import ArrayRecordWriter

    max_bytes = parse_file_size(args.max_file_size)

    # Open mmap dataset
    dataset = MMapIndexedDataset(args.input)
    num_docs = len(dataset)
    logging.info("Opened %s: %d documents", args.input, num_docs)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine token dtype for serialization
    doc_lengths = dataset.doc_lengths
    max_token_id = 0
    # Sample a few docs to detect max token ID
    for i in range(min(100, num_docs)):
        tokens = dataset[i]
        max_token_id = max(max_token_id, int(tokens.max()))

    if max_token_id < 65536:
        out_dtype = np.uint16
        logging.info("Using uint16 for token storage (max_token_id=%d)", max_token_id)
    else:
        out_dtype = np.int32
        logging.info("Using int32 for token storage (max_token_id=%d)", max_token_id)

    # Streaming shard creation based on max file size
    shard_idx = 0
    current_bytes = 0
    tmp_paths: list[Path] = []

    def _open_writer(idx: int) -> ArrayRecordWriter:
        path = output_dir / f"shard-{idx:05d}.arecord.tmp"
        tmp_paths.append(path)
        return ArrayRecordWriter(str(path), "group_size:1")

    writer = _open_writer(shard_idx)

    for doc_idx in range(num_docs):
        tokens = dataset[doc_idx]
        record = tokens.astype(out_dtype).tobytes()
        record_size = len(record)

        # Check if adding this record exceeds the limit (but allow first record in a shard)
        if current_bytes > 0 and current_bytes + record_size > max_bytes:
            writer.close()
            shard_idx += 1
            writer = _open_writer(shard_idx)
            current_bytes = 0

        writer.write(record)
        current_bytes += record_size

        if (doc_idx + 1) % 10000 == 0:
            logging.info("Processed %d / %d documents (shard %d)", doc_idx + 1, num_docs, shard_idx)

    writer.close()
    total_shards = shard_idx + 1

    # Rename .tmp files to final names with total shard count
    for i, tmp_path in enumerate(tmp_paths):
        final_path = output_dir / f"shard-{i:05d}-of-{total_shards:05d}.arecord"
        tmp_path.rename(final_path)

    # Write doc_lengths.npy
    np.save(output_dir / "doc_lengths.npy", doc_lengths)
    logging.info("Done. Wrote %d documents to %d shards in %s", num_docs, total_shards, output_dir)


if __name__ == "__main__":
    main()
