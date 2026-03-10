"""Tests for the MinText data pipeline.

Tests cover:
- MMapIndexedDataset: reading .bin+.idx fixtures
- ArrayRecordDocDataset: reading .arecord fixtures
- DocumentDataSource: sample index building + caching
- BlendedDataSource: weighted sampling
- Grain pipeline: end-to-end with correct shapes
- ShiftTokens transform
- _parse_data_path parsing
- _detect_data_type auto-detection
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from mintext.data.indexed_dataset import MMapIndexedDataset, ArrayRecordDocDataset
from mintext.data.dataset import (
    DocumentDataSource,
    BlendedDataSource,
    _build_document_index,
    _build_sample_index,
)
from mintext.data.pipeline import (
    ShiftTokens,
    _parse_data_path,
    _detect_data_type,
    _parse_split,
)
from mintext.utils.filesize import parse_file_size


# ============================================================
# Fixtures: create small .bin+.idx and .arecord test data
# ============================================================


def _write_mmap_fixture(tmp_path: Path, documents: list[np.ndarray], dtype=np.int32):
    """Create a minimal .bin+.idx file pair from document arrays."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    prefix = str(tmp_path / "test_data")
    idx_path = prefix + ".idx"
    bin_path = prefix + ".bin"

    dtype_codes = {
        np.dtype(np.uint8): 1,
        np.dtype(np.int8): 2,
        np.dtype(np.int16): 3,
        np.dtype(np.int32): 4,
        np.dtype(np.int64): 5,
        np.dtype(np.float32): 7,
        np.dtype(np.uint16): 8,
    }
    np_dtype = np.dtype(dtype)
    dtype_code = dtype_codes[np_dtype]
    dtype_size = np_dtype.itemsize

    # Each document is one sequence for simplicity
    seq_count = len(documents)
    doc_count = len(documents) + 1  # format: [0, end_doc0, end_doc1, ...]

    sequence_lengths = np.array([len(d) for d in documents], dtype=np.int32)

    # Compute sequence pointers (byte offsets in .bin)
    sequence_pointers = np.zeros(seq_count, dtype=np.int64)
    offset = 0
    for i, doc in enumerate(documents):
        sequence_pointers[i] = offset
        offset += len(doc) * dtype_size

    # Document indices: [0, 1, 2, ..., N]
    document_indices = np.arange(doc_count, dtype=np.int64)

    # Write .idx
    with open(idx_path, "wb") as f:
        f.write(b"MMIDIDX\x00\x00")
        f.write(struct.pack("<Q", 1))  # version
        f.write(struct.pack("<B", dtype_code))
        f.write(struct.pack("<Q", seq_count))
        f.write(struct.pack("<Q", doc_count))
        f.write(sequence_lengths.tobytes())
        f.write(sequence_pointers.tobytes())
        f.write(document_indices.tobytes())

    # Write .bin
    with open(bin_path, "wb") as f:
        for doc in documents:
            f.write(doc.astype(dtype).tobytes())

    return prefix


def _write_arecord_fixture(tmp_path: Path, documents: list[np.ndarray], dtype=np.int32):
    """Create a small .arecord directory with companion doc_lengths.npy."""
    from array_record.python.array_record_module import ArrayRecordWriter

    out_dir = tmp_path / "arecord_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = ArrayRecordWriter(str(out_dir / "shard-00000-of-00001.arecord"), "group_size:1")
    doc_lengths = []
    for doc in documents:
        arr = doc.astype(dtype)
        writer.write(arr.tobytes())
        doc_lengths.append(len(arr))
    writer.close()

    np.save(out_dir / "doc_lengths.npy", np.array(doc_lengths, dtype=np.int32))
    return str(out_dir)


@pytest.fixture
def sample_documents():
    """Create sample documents with known content."""
    return [
        np.array([10, 20, 30, 40, 50], dtype=np.int32),
        np.array([100, 200, 300], dtype=np.int32),
        np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32),
        np.array([42, 43], dtype=np.int32),
    ]


@pytest.fixture
def mmap_fixture(tmp_path, sample_documents):
    """Create a .bin+.idx fixture."""
    return _write_mmap_fixture(tmp_path, sample_documents)


@pytest.fixture
def arecord_fixture(tmp_path, sample_documents):
    """Create an .arecord fixture."""
    return _write_arecord_fixture(tmp_path, sample_documents)


# ============================================================
# MMapIndexedDataset tests
# ============================================================


class TestMMapIndexedDataset:
    def test_len(self, mmap_fixture, sample_documents):
        ds = MMapIndexedDataset(mmap_fixture)
        assert len(ds) == len(sample_documents)

    def test_getitem(self, mmap_fixture, sample_documents):
        ds = MMapIndexedDataset(mmap_fixture)
        for i, expected in enumerate(sample_documents):
            np.testing.assert_array_equal(ds[i], expected)

    def test_getitem_negative_index(self, mmap_fixture, sample_documents):
        ds = MMapIndexedDataset(mmap_fixture)
        np.testing.assert_array_equal(ds[-1], sample_documents[-1])

    def test_getitem_out_of_range(self, mmap_fixture):
        ds = MMapIndexedDataset(mmap_fixture)
        with pytest.raises(IndexError):
            ds[100]

    def test_get_slice(self, mmap_fixture, sample_documents):
        ds = MMapIndexedDataset(mmap_fixture)
        result = ds.get(0, offset=1, length=3)
        np.testing.assert_array_equal(result, sample_documents[0][1:4])

    def test_get_no_length(self, mmap_fixture, sample_documents):
        ds = MMapIndexedDataset(mmap_fixture)
        result = ds.get(0, offset=2)
        np.testing.assert_array_equal(result, sample_documents[0][2:])

    def test_doc_lengths(self, mmap_fixture, sample_documents):
        ds = MMapIndexedDataset(mmap_fixture)
        expected = np.array([len(d) for d in sample_documents], dtype=np.int32)
        np.testing.assert_array_equal(ds.doc_lengths, expected)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MMapIndexedDataset(str(tmp_path / "nonexistent"))

    def test_invalid_magic(self, tmp_path):
        prefix = str(tmp_path / "bad")
        # Write bad .idx
        with open(prefix + ".idx", "wb") as f:
            f.write(b"BADMAGIC!")
        with open(prefix + ".bin", "wb") as f:
            f.write(b"")
        with pytest.raises(ValueError, match="Invalid .idx magic"):
            MMapIndexedDataset(prefix)


# ============================================================
# ArrayRecordDocDataset tests
# ============================================================


class TestArrayRecordDocDataset:
    def test_len(self, arecord_fixture, sample_documents):
        ds = ArrayRecordDocDataset(arecord_fixture)
        assert len(ds) == len(sample_documents)

    def test_getitem(self, arecord_fixture, sample_documents):
        ds = ArrayRecordDocDataset(arecord_fixture)
        for i, expected in enumerate(sample_documents):
            np.testing.assert_array_equal(ds[i], expected)

    def test_getitem_negative_index(self, arecord_fixture, sample_documents):
        ds = ArrayRecordDocDataset(arecord_fixture)
        np.testing.assert_array_equal(ds[-1], sample_documents[-1])

    def test_get_slice(self, arecord_fixture, sample_documents):
        ds = ArrayRecordDocDataset(arecord_fixture)
        result = ds.get(0, offset=1, length=3)
        np.testing.assert_array_equal(result, sample_documents[0][1:4])

    def test_doc_lengths(self, arecord_fixture, sample_documents):
        ds = ArrayRecordDocDataset(arecord_fixture)
        expected = np.array([len(d) for d in sample_documents], dtype=np.int32)
        np.testing.assert_array_equal(ds.doc_lengths, expected)

    def test_missing_doc_lengths(self, tmp_path):
        from array_record.python.array_record_module import ArrayRecordWriter
        out_dir = tmp_path / "no_lengths"
        out_dir.mkdir()
        w = ArrayRecordWriter(str(out_dir / "data.arecord"), "group_size:1")
        w.write(np.array([1, 2, 3], dtype=np.int32).tobytes())
        w.close()
        with pytest.raises(FileNotFoundError, match="doc_lengths.npy"):
            ArrayRecordDocDataset(str(out_dir))


# ============================================================
# Index building tests
# ============================================================


class TestIndexBuilding:
    def test_build_document_index_single_epoch(self):
        rng = np.random.RandomState(42)
        idx = _build_document_index(5, 1, rng)
        assert len(idx) == 5
        assert set(idx.tolist()) == {0, 1, 2, 3, 4}

    def test_build_document_index_multi_epoch(self):
        rng = np.random.RandomState(42)
        idx = _build_document_index(3, 4, rng)
        assert len(idx) == 12
        # Each epoch should contain all docs
        for epoch in range(4):
            chunk = set(idx[epoch * 3 : (epoch + 1) * 3].tolist())
            assert chunk == {0, 1, 2}

    def test_build_sample_index_simple(self):
        # 3 docs: [5, 5, 5] tokens, seq_len=4 (target=5)
        doc_index = np.array([0, 1, 2], dtype=np.int32)
        doc_lengths = np.array([5, 5, 5], dtype=np.int32)
        sample_idx = _build_sample_index(doc_index, doc_lengths, seq_len=4)

        # Total tokens = 15, target = 5, so 3 samples
        assert len(sample_idx) - 1 == 3

    def test_build_sample_index_cross_document(self):
        # 2 docs: [3, 3] tokens, seq_len=4 (target=5)
        # Should create 1 sample spanning both docs
        doc_index = np.array([0, 1], dtype=np.int32)
        doc_lengths = np.array([3, 3], dtype=np.int32)
        sample_idx = _build_sample_index(doc_index, doc_lengths, seq_len=4)

        # 6 tokens total, target=5, so 1 sample
        assert len(sample_idx) - 1 == 1

    def test_build_sample_index_exact_fit(self):
        # 1 doc of exactly seq_len+1 tokens
        doc_index = np.array([0], dtype=np.int32)
        doc_lengths = np.array([10], dtype=np.int32)
        sample_idx = _build_sample_index(doc_index, doc_lengths, seq_len=9)

        assert len(sample_idx) - 1 == 1

    def test_build_sample_index_no_extra_token(self):
        # With add_extra_token=False, target = seq_len (not seq_len+1)
        # 3 docs of 5 tokens each = 15 tokens, target=4 -> 3 samples
        # vs add_extra_token=True: target=5 -> 3 samples (same here)
        # Use 3 docs of 10 tokens, seq_len=4:
        #   True:  target=5, 10+10+10=30 -> 6 samples
        #   False: target=4, 30 -> 7 samples
        doc_index = np.array([0, 1, 2], dtype=np.int32)
        doc_lengths = np.array([10, 10, 10], dtype=np.int32)

        idx_extra = _build_sample_index(doc_index, doc_lengths, seq_len=4, add_extra_token=True)
        idx_no_extra = _build_sample_index(doc_index, doc_lengths, seq_len=4, add_extra_token=False)

        # With smaller target, more samples fit
        assert len(idx_no_extra) - 1 > len(idx_extra) - 1


# ============================================================
# DocumentDataSource tests
# ============================================================


class TestDocumentDataSource:
    def test_mmap_source(self, tmp_path):
        docs = [np.arange(20, dtype=np.int32), np.arange(20, 40, dtype=np.int32)]
        prefix = _write_mmap_fixture(tmp_path, docs)

        src = DocumentDataSource(
            data_path=prefix,
            data_type="mmap",
            seq_len=4,
            seed=42,
            num_epochs=1,
            split=(1.0, 0.0, 0.0),
            split_index=0,
            cache_dir=str(tmp_path / "cache"),
        )

        assert len(src) > 0
        item = src[0]
        assert "tokens" in item
        assert len(item["tokens"]) == 5  # seq_len + 1

    def test_mmap_source_no_extra_token(self, tmp_path):
        docs = [np.arange(20, dtype=np.int32), np.arange(20, 40, dtype=np.int32)]
        prefix = _write_mmap_fixture(tmp_path, docs)

        src = DocumentDataSource(
            data_path=prefix,
            data_type="mmap",
            seq_len=4,
            seed=42,
            num_epochs=1,
            split=(1.0, 0.0, 0.0),
            split_index=0,
            cache_dir=str(tmp_path / "cache_no_extra"),
            add_extra_token=False,
        )

        assert len(src) > 0
        item = src[0]
        assert "tokens" in item
        assert len(item["tokens"]) == 4  # seq_len (no +1)

    def test_arecord_source(self, tmp_path):
        docs = [np.arange(20, dtype=np.int32), np.arange(20, 40, dtype=np.int32)]
        ar_dir = _write_arecord_fixture(tmp_path, docs)

        src = DocumentDataSource(
            data_path=ar_dir,
            data_type="arecord",
            seq_len=4,
            seed=42,
            num_epochs=1,
            split=(1.0, 0.0, 0.0),
            split_index=0,
            cache_dir=str(tmp_path / "cache"),
        )

        assert len(src) > 0
        item = src[0]
        assert "tokens" in item
        assert len(item["tokens"]) == 5

    def test_cache_reuse(self, tmp_path):
        docs = [np.arange(50, dtype=np.int32)]
        prefix = _write_mmap_fixture(tmp_path, docs)
        cache = str(tmp_path / "cache")

        # First build
        src1 = DocumentDataSource(
            data_path=prefix, data_type="mmap", seq_len=4,
            seed=42, num_epochs=1, split=(1.0, 0.0, 0.0),
            split_index=0, cache_dir=cache,
        )

        # Check cache files exist
        cache_files = list(Path(cache).glob("*.npy"))
        assert len(cache_files) == 2  # document_index + sample_index

        # Second build should load from cache (no error)
        src2 = DocumentDataSource(
            data_path=prefix, data_type="mmap", seq_len=4,
            seed=42, num_epochs=1, split=(1.0, 0.0, 0.0),
            split_index=0, cache_dir=cache,
        )

        assert len(src1) == len(src2)

    def test_split(self, tmp_path):
        # 10 documents, split 80/20/0
        docs = [np.arange(i * 10, (i + 1) * 10, dtype=np.int32) for i in range(10)]
        prefix = _write_mmap_fixture(tmp_path, docs)

        train_src = DocumentDataSource(
            data_path=prefix, data_type="mmap", seq_len=4,
            seed=42, num_epochs=1, split=(80.0, 20.0, 0.0),
            split_index=0, cache_dir=str(tmp_path / "cache_train"),
        )

        val_src = DocumentDataSource(
            data_path=prefix, data_type="mmap", seq_len=4,
            seed=42, num_epochs=1, split=(80.0, 20.0, 0.0),
            split_index=1, cache_dir=str(tmp_path / "cache_val"),
        )

        # Train should have more samples than val
        assert len(train_src) > len(val_src)


# ============================================================
# BlendedDataSource tests
# ============================================================


class TestBlendedDataSource:
    def test_basic_blending(self, tmp_path):
        docs_a = [np.arange(20, dtype=np.int32)]
        docs_b = [np.arange(100, 120, dtype=np.int32)]
        prefix_a = _write_mmap_fixture(tmp_path / "a", docs_a)
        prefix_b = _write_mmap_fixture(tmp_path / "b", docs_b)

        src_a = DocumentDataSource(
            data_path=prefix_a, data_type="mmap", seq_len=4,
            seed=42, num_epochs=1, split=(1.0, 0.0, 0.0),
            split_index=0, cache_dir=str(tmp_path / "cache_a"),
        )
        src_b = DocumentDataSource(
            data_path=prefix_b, data_type="mmap", seq_len=4,
            seed=42, num_epochs=1, split=(1.0, 0.0, 0.0),
            split_index=0, cache_dir=str(tmp_path / "cache_b"),
        )

        total = len(src_a) + len(src_b)
        blended = BlendedDataSource([src_a, src_b], [0.7, 0.3], size=total)
        assert len(blended) == total

        # Should be able to access all items
        for i in range(len(blended)):
            item = blended[i]
            assert "tokens" in item

    def test_proportional_sampling(self, tmp_path):
        docs = [np.arange(100, dtype=np.int32)]
        prefix_a = _write_mmap_fixture(tmp_path / "a", docs)
        prefix_b = _write_mmap_fixture(tmp_path / "b", docs)

        src_a = DocumentDataSource(
            data_path=prefix_a, data_type="mmap", seq_len=4,
            seed=42, num_epochs=1, split=(1.0, 0.0, 0.0),
            split_index=0, cache_dir=str(tmp_path / "cache_a"),
        )
        src_b = DocumentDataSource(
            data_path=prefix_b, data_type="mmap", seq_len=4,
            seed=42, num_epochs=1, split=(1.0, 0.0, 0.0),
            split_index=0, cache_dir=str(tmp_path / "cache_b"),
        )

        size = 100
        blended = BlendedDataSource([src_a, src_b], [0.7, 0.3], size=size)

        # Count assignments
        count_a = np.sum(blended._dataset_index == 0)
        count_b = np.sum(blended._dataset_index == 1)

        # Should be roughly 70/30
        assert abs(count_a / size - 0.7) < 0.05
        assert abs(count_b / size - 0.3) < 0.05


# ============================================================
# ShiftTokens tests
# ============================================================


class TestShiftTokens:
    def test_shift(self):
        transform = ShiftTokens()
        tokens = np.array([1, 2, 3, 4, 5])
        result = transform.map({"tokens": tokens})

        np.testing.assert_array_equal(result["input_tokens"], [1, 2, 3, 4])
        np.testing.assert_array_equal(result["target_tokens"], [2, 3, 4, 5])

    def test_shift_no_extra_token(self):
        transform = ShiftTokens(add_extra_token=False)
        tokens = np.array([1, 2, 3, 4])
        result = transform.map({"tokens": tokens})

        np.testing.assert_array_equal(result["input_tokens"], [1, 2, 3, 4])
        np.testing.assert_array_equal(result["target_tokens"], [2, 3, 4, 1])  # roll(-1)


# ============================================================
# Utility function tests
# ============================================================


class TestParseDataPath:
    def test_single_path(self):
        result = _parse_data_path("/data/corpus/tokenized")
        assert result == [(1.0, "/data/corpus/tokenized")]

    def test_weighted_paths(self):
        result = _parse_data_path("0.7 /data/en 0.3 /data/code")
        assert len(result) == 2
        assert result[0] == (0.7, "/data/en")
        assert result[1] == (0.3, "/data/code")

    def test_three_weighted_paths(self):
        result = _parse_data_path("0.7 /data/en 0.2 /data/code 0.1 /data/math")
        assert len(result) == 3
        assert result[2] == (0.1, "/data/math")

    def test_path_with_spaces_not_weights(self):
        # If first token isn't a float, treat whole thing as one path
        result = _parse_data_path("/data/my corpus")
        assert result == [(1.0, "/data/my corpus")]


class TestDetectDataType:
    def test_detect_mmap(self, tmp_path):
        prefix = str(tmp_path / "data")
        Path(prefix + ".bin").touch()
        Path(prefix + ".idx").touch()
        assert _detect_data_type(prefix) == "mmap"

    def test_detect_arecord_dir(self, tmp_path):
        ar_dir = tmp_path / "ardata"
        ar_dir.mkdir()
        (ar_dir / "shard.arecord").touch()
        assert _detect_data_type(str(ar_dir)) == "arecord"

    def test_detect_arecord_file(self, tmp_path):
        ar_file = tmp_path / "data.arecord"
        ar_file.touch()
        assert _detect_data_type(str(ar_file)) == "arecord"

    def test_detect_unknown(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot detect data type"):
            _detect_data_type(str(tmp_path / "nonexistent"))


class TestParseSplit:
    def test_valid(self):
        assert _parse_split("99,1,0") == (99.0, 1.0, 0.0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            _parse_split("50,50")


# ============================================================
# File size parsing tests
# ============================================================


class TestParseFileSize:
    def test_gigabytes(self):
        assert parse_file_size("5G") == 5 * 1024**3

    def test_gigabytes_suffix_gb(self):
        assert parse_file_size("5GB") == 5 * 1024**3

    def test_megabytes(self):
        assert parse_file_size("100M") == 100 * 1024**2

    def test_megabytes_suffix_mb(self):
        assert parse_file_size("100MB") == 100 * 1024**2

    def test_terabytes(self):
        assert parse_file_size("2T") == 2 * 1024**4

    def test_kilobytes(self):
        assert parse_file_size("512K") == 512 * 1024

    def test_raw_bytes(self):
        assert parse_file_size("1024") == 1024

    def test_case_insensitive(self):
        assert parse_file_size("5g") == 5 * 1024**3
        assert parse_file_size("100m") == 100 * 1024**2

    def test_whitespace(self):
        assert parse_file_size("  5G  ") == 5 * 1024**3

    def test_fractional(self):
        assert parse_file_size("1.5G") == int(1.5 * 1024**3)

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            parse_file_size("abc")

    def test_invalid_suffix(self):
        with pytest.raises(ValueError):
            parse_file_size("5X")


# ============================================================
# Multi-part .bin tests
# ============================================================


def _split_bin_for_test(prefix: str, split_after_seq: int):
    """Split a .bin file into two parts at a sequence boundary.

    Reads sequence pointers/lengths from MMapIndexedDataset to find the split point,
    then writes prefix.bin.00000 and prefix.bin.00001, removing the original.
    """
    ds = MMapIndexedDataset(prefix)
    # Compute byte offset where split_after_seq ends
    split_byte = int(ds._sequence_pointers[split_after_seq]) + \
        int(ds._sequence_lengths[split_after_seq]) * ds._dtype_size

    bin_path = Path(f"{prefix}.bin")
    data = bin_path.read_bytes()

    # Write two parts
    Path(f"{prefix}.bin.00000").write_bytes(data[:split_byte])
    Path(f"{prefix}.bin.00001").write_bytes(data[split_byte:])

    # Remove original
    bin_path.unlink()


class TestMMapMultiPart:
    def test_multipart_reads_match_single(self, tmp_path, sample_documents):
        """Multi-part .bin reads should match single .bin reads exactly."""
        prefix = _write_mmap_fixture(tmp_path, sample_documents)

        # Read all docs from single-file dataset
        ds_single = MMapIndexedDataset(prefix)
        single_results = [ds_single[i].copy() for i in range(len(ds_single))]

        # Split after sequence 1 (so part 0 has docs 0-1, part 1 has docs 2-3)
        _split_bin_for_test(prefix, split_after_seq=1)

        # Read all docs from multi-part dataset
        ds_multi = MMapIndexedDataset(prefix)
        for i, expected in enumerate(single_results):
            np.testing.assert_array_equal(ds_multi[i], expected)

    def test_multipart_doc_lengths(self, tmp_path, sample_documents):
        """doc_lengths should match between single and multi-part."""
        prefix = _write_mmap_fixture(tmp_path, sample_documents)

        ds_single = MMapIndexedDataset(prefix)
        expected_lengths = ds_single.doc_lengths.copy()

        _split_bin_for_test(prefix, split_after_seq=1)

        ds_multi = MMapIndexedDataset(prefix)
        np.testing.assert_array_equal(ds_multi.doc_lengths, expected_lengths)

    def test_single_part_backward_compatible(self, mmap_fixture, sample_documents):
        """Existing single .bin files should continue to work."""
        ds = MMapIndexedDataset(mmap_fixture)
        assert len(ds) == len(sample_documents)
        for i, expected in enumerate(sample_documents):
            np.testing.assert_array_equal(ds[i], expected)

    def test_multipart_get_slice(self, tmp_path, sample_documents):
        """get() with offset/length should work on multi-part data."""
        prefix = _write_mmap_fixture(tmp_path, sample_documents)
        _split_bin_for_test(prefix, split_after_seq=1)

        ds = MMapIndexedDataset(prefix)
        # doc 2 is in part 1, get a slice
        result = ds.get(2, offset=1, length=3)
        np.testing.assert_array_equal(result, sample_documents[2][1:4])

    def test_no_bin_files_raises(self, tmp_path, sample_documents):
        """Missing both .bin and .bin.00000 should raise FileNotFoundError."""
        prefix = _write_mmap_fixture(tmp_path, sample_documents)
        Path(f"{prefix}.bin").unlink()
        with pytest.raises(FileNotFoundError):
            MMapIndexedDataset(prefix)


class TestDetectDataTypeMultiPart:
    def test_detect_multipart_mmap(self, tmp_path):
        """Multi-part .bin.00000 + .idx should be detected as mmap."""
        prefix = str(tmp_path / "data")
        Path(prefix + ".idx").touch()
        Path(prefix + ".bin.00000").touch()
        assert _detect_data_type(prefix) == "mmap"

    def test_detect_single_mmap_still_works(self, tmp_path):
        """Single .bin + .idx should still be detected as mmap."""
        prefix = str(tmp_path / "data")
        Path(prefix + ".bin").touch()
        Path(prefix + ".idx").touch()
        assert _detect_data_type(prefix) == "mmap"
