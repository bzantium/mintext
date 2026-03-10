"""Convert raw text (jsonl/parquet/gz/zstd) to document-level .arecord format.

Tokenizes text, writes each document as a separate ArrayRecord record.
Also writes doc_lengths.npy companion file.

Usage:
    python tools/text_to_arecord.py \
        --input /path/to/data/*.jsonl \
        --output /path/to/output_dir \
        --tokenizer-path /path/to/tokenizer \
        --tokenizer-type huggingface \
        --max-file-size 5G \
        --workers 16 \
        --append-eos
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import multiprocessing
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mintext.data.tokenizer import Tokenizer
from mintext.utils.filesize import parse_file_size


def _read_documents(file_paths: list[str], text_key: str = "text"):
    """Yield text documents from files (jsonl/parquet/gz/zstd)."""
    for fpath in file_paths:
        p = Path(fpath)
        suffix = "".join(p.suffixes).lower()

        if suffix.endswith(".parquet"):
            try:
                from pyarrow import parquet as pq
            except ImportError:
                raise ImportError("pyarrow required for parquet files: pip install pyarrow")
            table = pq.read_table(fpath, columns=[text_key])
            for text in table[text_key].to_pylist():
                if text:
                    yield text

        elif suffix.endswith(".gz"):
            with gzip.open(fpath, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        text = data.get(text_key, "")
                        if text:
                            yield text

        elif suffix.endswith(".zstd") or suffix.endswith(".zst"):
            import zstandard as zstd
            with open(fpath, "rb") as raw:
                dctx = zstd.ZstdDecompressor()
                reader = io.BufferedReader(dctx.stream_reader(raw))
                for line in reader:
                    line = line.decode("utf-8").strip()
                    if line:
                        data = json.loads(line)
                        text = data.get(text_key, "")
                        if text:
                            yield text

        else:
            # Plain text / jsonl
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            text = data.get(text_key, "")
                        except json.JSONDecodeError:
                            text = line
                        if text:
                            yield text


# Global tokenizer for multiprocessing workers
_global_tokenizer: Tokenizer | None = None
_global_append_eos: bool = False


def _init_worker(tokenizer_path: str, tokenizer_type: str, append_eos: bool):
    global _global_tokenizer, _global_append_eos
    _global_tokenizer = Tokenizer(tokenizer_path, type=tokenizer_type)
    _global_append_eos = append_eos


def _tokenize_doc(text: str) -> list[int]:
    assert _global_tokenizer is not None
    ids = _global_tokenizer.encode(text)
    if _global_append_eos:
        ids.append(_global_tokenizer.eos_id)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw text to document-level .arecord")
    parser.add_argument("--input", nargs="+", required=True, help="Input file paths (jsonl/parquet/gz/zstd)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer")
    parser.add_argument("--tokenizer-type", default="huggingface", choices=["huggingface", "sentencepiece"])
    parser.add_argument("--max-file-size", default="5G", help="Maximum size per output shard (e.g. 5G, 500M)")
    parser.add_argument("--workers", type=int, default=1, help="Number of tokenization workers")
    parser.add_argument("--text-key", default="text", help="JSON key for text field")
    parser.add_argument("--append-eos", action="store_true", help="Append EOS token to each document")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from array_record.python.array_record_module import ArrayRecordWriter

    max_bytes = parse_file_size(args.max_file_size)

    # Expand globs
    input_files = []
    for pattern in args.input:
        p = Path(pattern)
        if "*" in str(p) or "?" in str(p):
            input_files.extend(str(f) for f in sorted(p.parent.glob(p.name)))
        else:
            input_files.append(str(p))

    if not input_files:
        logging.error("No input files found")
        sys.exit(1)

    logging.info("Processing %d input files", len(input_files))

    # Determine vocab size to pick dtype
    tok = Tokenizer(args.tokenizer_path, type=args.tokenizer_type)
    out_dtype = np.uint16 if tok.vocab_size < 65536 else np.int32
    logging.info("Vocab size: %d, token dtype: %s", tok.vocab_size, out_dtype)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Streaming shard creation based on max file size
    shard_idx = 0
    current_bytes = 0
    tmp_paths: list[Path] = []

    def _open_writer(idx: int) -> ArrayRecordWriter:
        path = output_dir / f"shard-{idx:05d}.arecord.tmp"
        tmp_paths.append(path)
        return ArrayRecordWriter(str(path), "group_size:1")

    writer = _open_writer(shard_idx)

    # Tokenize and write
    doc_lengths = []
    doc_count = 0

    documents = _read_documents(input_files, text_key=args.text_key)

    if args.workers > 1:
        pool = multiprocessing.Pool(
            args.workers,
            initializer=_init_worker,
            initargs=(args.tokenizer_path, args.tokenizer_type, args.append_eos),
        )
        token_iter = pool.imap(_tokenize_doc, documents, chunksize=64)
    else:
        _init_worker(args.tokenizer_path, args.tokenizer_type, args.append_eos)
        token_iter = (_tokenize_doc(doc) for doc in documents)

    for token_ids in token_iter:
        if not token_ids:
            continue

        arr = np.array(token_ids, dtype=out_dtype)
        record = arr.tobytes()
        record_size = len(record)

        # Check if adding this record exceeds the limit (but allow first record in a shard)
        if current_bytes > 0 and current_bytes + record_size > max_bytes:
            writer.close()
            shard_idx += 1
            writer = _open_writer(shard_idx)
            current_bytes = 0

        writer.write(record)
        current_bytes += record_size
        doc_lengths.append(len(token_ids))
        doc_count += 1

        if doc_count % 10000 == 0:
            logging.info("Processed %d documents (shard %d)", doc_count, shard_idx)

    writer.close()
    total_shards = shard_idx + 1

    if args.workers > 1:
        pool.close()
        pool.join()

    # Rename .tmp files to final names with total shard count
    for i, tmp_path in enumerate(tmp_paths):
        final_path = output_dir / f"shard-{i:05d}-of-{total_shards:05d}.arecord"
        tmp_path.rename(final_path)

    # Write doc_lengths.npy
    np.save(output_dir / "doc_lengths.npy", np.array(doc_lengths, dtype=np.int32))
    logging.info("Done. Wrote %d documents to %d shards in %s", doc_count, total_shards, output_dir)


if __name__ == "__main__":
    main()
