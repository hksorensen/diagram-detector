"""
Test cache integrity and corruption prevention.

Critical tests:
- cache.set() MUST reject empty lists
- cache.get() MUST validate entries
- No 22-byte blobs should ever be created
"""

import pytest
import tempfile
import gzip
import json
import hashlib
from pathlib import Path
from diagram_detector import DetectionCache


class TestCacheIntegrity:
    """Test that cache prevents corruption at the source."""

    def test_cache_rejects_empty_list(self, cache, sample_pdf_path, cache_params):
        """CRITICAL: cache.set() must reject empty result lists."""
        # This should raise ValueError
        with pytest.raises(ValueError, match="empty results"):
            cache.set(
                sample_pdf_path,
                results=[],  # Empty list should be rejected!
                **cache_params
            )

    def test_cache_rejects_non_list(self, cache, sample_pdf_path, cache_params):
        """cache.set() must reject non-list results."""
        # This should raise TypeError
        with pytest.raises(TypeError):
            cache.set(
                sample_pdf_path,
                results=None,  # Not a list
                **cache_params
            )

    def test_cache_accepts_valid_results(self, cache, sample_pdf_path, cache_params, valid_detection_results):
        """cache.set() must accept non-empty result lists."""
        # This should succeed
        cache.set(
            sample_pdf_path,
            results=valid_detection_results,
            **cache_params
        )

        # Verify it was cached
        cached = cache.get(
            sample_pdf_path,
            **cache_params
        )

        assert cached is not None
        assert len(cached) == 1
        assert cached[0]["page_number"] == 0

    def test_cache_accepts_empty_detection_but_not_empty_list(self, cache, sample_pdf_path, cache_params, empty_detection_results):
        """cache.set() should accept results with no detections (but not empty list)."""
        # This should succeed - it's a valid result with no detections found
        cache.set(
            sample_pdf_path,
            results=empty_detection_results,
            **cache_params
        )

        # Verify it was cached
        cached = cache.get(
            sample_pdf_path,
            **cache_params
        )

        assert cached is not None
        assert len(cached) == 1
        assert cached[0]["boxes"] == []

    def test_cache_get_validates_empty_arrays(self, temp_cache_dir, sample_pdf_path, cache_params):
        """cache.get() must invalidate 22-byte empty array entries."""
        cache = DetectionCache(cache_dir=temp_cache_dir)

        # Manually insert a corrupted 22-byte entry (bypass validation)
        cache_key = cache._compute_cache_key(
            sample_pdf_path,
            **cache_params
        )

        # Create 22-byte compressed empty array
        empty_array = json.dumps([])
        compressed = gzip.compress(empty_array.encode('utf-8'))

        assert len(compressed) == 22  # Verify it's 22 bytes

        # Manually insert into DB (bypass cache.set() validation)
        from datetime import datetime
        cache.conn.execute(
            """INSERT INTO detection_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (cache_key, sample_pdf_path.name, 1000, 0.0, cache_params["model"],
             cache_params["confidence"], cache_params["iou"], cache_params["dpi"],
             cache_params["imgsz"], 0, 10, compressed, 22,
             datetime.now().isoformat(), datetime.now().isoformat(), 1)
        )

        # Now try to get it - should return None (invalidated)
        result = cache.get(
            sample_pdf_path,
            **cache_params
        )

        assert result is None  # Empty arrays should be invalidated!

    def test_no_22_byte_blobs_created(self, cache, sample_pdf_path, cache_params,
                                     valid_detection_results, multi_page_results,
                                     empty_detection_results):
        """Verify that normal operation never creates 22-byte blobs."""
        # Cache various result sizes
        test_cases = [
            (Path("valid.pdf"), valid_detection_results),
            (Path("multi.pdf"), multi_page_results),
            (Path("empty_det.pdf"), empty_detection_results),
        ]

        for pdf_name, results in test_cases:
            # Create fake PDF files
            pdf_path = sample_pdf_path.parent / pdf_name
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")

            cache.set(
                pdf_path,
                results=results,
                **cache_params
            )

        # Check database - no entries should be 22 bytes
        rows = cache.conn.query_all(
            "SELECT compressed_size FROM detection_cache WHERE compressed_size = 22"
        )

        assert len(rows) == 0  # No 22-byte entries!

    def test_cache_stats_after_operations(self, cache, sample_pdf_path, cache_params,
                                         valid_detection_results):
        """Test that cache stats work correctly."""
        # Add a valid entry
        cache.set(
            sample_pdf_path,
            results=valid_detection_results,
            **cache_params
        )

        # Check stats
        stats = cache.stats()
        assert stats['num_entries'] == 1

        # Verify get updates access count
        cache.get(sample_pdf_path, **cache_params)
        cache.get(sample_pdf_path, **cache_params)

        # Access count should be updated
        row = cache.conn.query_one(
            "SELECT access_count FROM detection_cache WHERE pdf_name = ?",
            (sample_pdf_path.name,)
        )
        assert row[0] == 2  # Accessed twice


class TestCacheValidation:
    """Test cache validation and health checking."""

    def test_detect_corruption_in_existing_cache(self, temp_cache_dir, sample_pdf_path, cache_params):
        """Test that we can detect corruption in existing cache."""
        cache = DetectionCache(cache_dir=temp_cache_dir)

        # Add a valid entry
        valid_results = [{"page_number": 0, "boxes": [[1, 2, 3, 4]], "scores": [0.9], "class_ids": [0]}]
        cache.set(sample_pdf_path, results=valid_results, **cache_params)

        # Manually insert a corrupted entry
        corrupted_path = sample_pdf_path.parent / "corrupted.pdf"
        corrupted_path.write_bytes(b"%PDF-1.4\n%%EOF")

        cache_key = cache._compute_cache_key(corrupted_path, **cache_params)
        empty_array = json.dumps([])
        compressed = gzip.compress(empty_array.encode('utf-8'))

        from datetime import datetime
        cache.conn.execute(
            """INSERT INTO detection_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (cache_key, corrupted_path.name, 1000, 0.0, cache_params["model"],
             cache_params["confidence"], cache_params["iou"], cache_params["dpi"],
             cache_params["imgsz"], 0, 10, compressed, 22,
             datetime.now().isoformat(), datetime.now().isoformat(), 1)
        )

        # Check corruption count
        corruption_count = cache.conn.query_one(
            "SELECT COUNT(*) FROM detection_cache WHERE compressed_size = 22"
        )[0]

        assert corruption_count == 1  # One corrupted entry

        total_count = cache.stats()['num_entries']
        assert total_count == 2  # Total of 2 entries

        corruption_pct = corruption_count / total_count * 100
        assert corruption_pct == 50.0  # 50% corrupted


def test_cache_key_consistency(temp_cache_dir, sample_pdf_path, cache_params):
    """Test that cache keys are computed consistently."""
    cache = DetectionCache(cache_dir=temp_cache_dir)

    # Compute key twice
    key1 = cache._compute_cache_key(sample_pdf_path, **cache_params)
    key2 = cache._compute_cache_key(sample_pdf_path, **cache_params)

    assert key1 == key2  # Keys should be identical

    # Different parameters should produce different keys
    params2 = cache_params.copy()
    params2["confidence"] = 0.2
    key3 = cache._compute_cache_key(sample_pdf_path, **params2)

    assert key1 != key3  # Different confidence = different key


def test_cache_compression(temp_cache_dir, sample_pdf_path, cache_params):
    """Test that compression works and reduces size."""
    cache = DetectionCache(cache_dir=temp_cache_dir)

    # Create large results
    large_results = [
        {
            "page_number": i,
            "boxes": [[j*10, j*10, j*20, j*20] for j in range(50)],  # 50 boxes per page
            "scores": [0.9] * 50,
            "class_ids": [0] * 50,
        }
        for i in range(100)  # 100 pages
    ]

    cache.set(sample_pdf_path, results=large_results, **cache_params)

    # Check compressed size vs uncompressed size
    row = cache.conn.query_one(
        "SELECT compressed_size FROM detection_cache WHERE pdf_name = ?",
        (sample_pdf_path.name,)
    )
    compressed_size = row[0]

    # Uncompressed JSON size
    uncompressed_size = len(json.dumps(large_results).encode('utf-8'))

    # Compression should reduce size significantly
    assert compressed_size < uncompressed_size
    compression_ratio = compressed_size / uncompressed_size
    assert compression_ratio < 0.5  # At least 50% compression
