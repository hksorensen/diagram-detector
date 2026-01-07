"""
Detection results cache with SQLite backend.

Supports thread-safe caching with proper cache keys that include
all detection parameters (model, confidence, iou, dpi).
"""

import sqlite3
import json
import gzip
import hashlib
from pathlib import Path
from threading import local
from contextlib import contextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime


class ThreadSafeConnection:
    """Thread-local SQLite connection manager."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = local()

    def _get_conn(self):
        """Get or create thread-local connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            # Enable WAL mode for better concurrency
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    @contextmanager
    def transaction(self):
        """Context manager for transactions with automatic commit/rollback."""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def execute(self, sql: str, params: tuple = ()):
        """Execute SQL with automatic transaction."""
        with self.transaction() as conn:
            return conn.execute(sql, params)

    def query_one(self, sql: str, params: tuple = ()) -> Optional[tuple]:
        """Query single row."""
        conn = self._get_conn()
        cursor = conn.execute(sql, params)
        return cursor.fetchone()

    def query_all(self, sql: str, params: tuple = ()) -> List[tuple]:
        """Query all matching rows."""
        conn = self._get_conn()
        cursor = conn.execute(sql, params)
        return cursor.fetchall()


class DetectionCache:
    """
    SQLite-based cache for diagram detection results.

    Features:
    - Thread-safe operations
    - Gzip compression (~70-90% space savings)
    - Proper cache keys (includes model + all detection params)
    - Automatic cleanup by size/age
    - LRU tracking for intelligent eviction

    Cache key components:
    - PDF identity: name + size + mtime
    - Detection params: model + confidence + iou + dpi

    This ensures different detection runs don't incorrectly reuse
    cached results from runs with different parameters.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_size_gb: float = 10.0,
        max_age_days: Optional[int] = 90,
        compression: bool = True,
        auto_cleanup: bool = True,
    ):
        """
        Initialize detection cache.

        Args:
            cache_dir: Cache directory (default: ~/.cache/diagram-detector)
            max_size_gb: Maximum cache size in GB (default: 10.0)
            max_age_days: Delete entries older than this (None = never)
            compression: Enable gzip compression (default: True)
            auto_cleanup: Auto-cleanup when max_size exceeded (default: True)
        """
        # Set cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "diagram-detector"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.cache_dir / "detection_cache.db"
        self.max_size_gb = max_size_gb
        self.max_age_days = max_age_days
        self.compression = compression
        self.auto_cleanup = auto_cleanup

        # Initialize connection and database
        self.conn = ThreadSafeConnection(self.db_path)
        self._init_db()

        # Run cleanup if enabled
        if self.auto_cleanup:
            self._auto_cleanup()

    def _init_db(self):
        """Initialize database schema."""
        schema = """
        CREATE TABLE IF NOT EXISTS detection_cache (
            cache_key TEXT PRIMARY KEY,

            -- PDF metadata
            pdf_name TEXT NOT NULL,
            pdf_size INTEGER NOT NULL,
            pdf_mtime REAL NOT NULL,

            -- Detection parameters (critical for cache validity!)
            model TEXT NOT NULL,
            confidence REAL NOT NULL,
            iou REAL NOT NULL,
            dpi INTEGER NOT NULL,
            imgsz INTEGER NOT NULL,

            -- Results
            num_pages INTEGER NOT NULL,
            results_compressed BLOB NOT NULL,
            compressed_size INTEGER NOT NULL,

            -- Cache metadata
            cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            access_count INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_pdf_name ON detection_cache(pdf_name);
        CREATE INDEX IF NOT EXISTS idx_model ON detection_cache(model);
        CREATE INDEX IF NOT EXISTS idx_cached_at ON detection_cache(cached_at);
        CREATE INDEX IF NOT EXISTS idx_last_accessed ON detection_cache(last_accessed);
        CREATE INDEX IF NOT EXISTS idx_access_count ON detection_cache(access_count);
        """

        with self.conn.transaction() as conn:
            conn.executescript(schema)

    def _compute_cache_key(
        self,
        pdf_path: Path,
        model: str,
        confidence: float,
        iou: float,
        dpi: int,
        imgsz: int,
    ) -> str:
        """
        Compute cache key including ALL detection parameters.

        This is critical! The cache key MUST include model, confidence,
        iou, dpi, and imgsz. Otherwise, different detection runs will
        incorrectly reuse cached results.

        Args:
            pdf_path: Path to PDF file
            model: Model name (e.g., 'v5', 'yolo11m')
            confidence: Confidence threshold (e.g., 0.20)
            iou: IoU threshold (e.g., 0.30)
            dpi: DPI for PDF conversion (e.g., 300)
            imgsz: Image size for preprocessing (e.g., 640)

        Returns:
            SHA-256 hash of all parameters
        """
        stat = pdf_path.stat()

        # Round floats to avoid floating point noise
        # (0.20000001 vs 0.20 should be same cache entry)
        conf_rounded = round(confidence, 3)
        iou_rounded = round(iou, 3)

        # Build key with ALL parameters
        key_data = (
            f"pdf:{pdf_path.name}"
            f"|size:{stat.st_size}"
            f"|mtime:{int(stat.st_mtime)}"
            f"|model:{model}"
            f"|conf:{conf_rounded}"
            f"|iou:{iou_rounded}"
            f"|dpi:{dpi}"
            f"|imgsz:{imgsz}"
        )

        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(
        self,
        pdf_path: Path,
        model: str,
        confidence: float,
        iou: float,
        dpi: int,
        imgsz: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached results for PDF with specific detection parameters.

        Returns None if not cached or parameters don't match.

        Args:
            pdf_path: Path to PDF file
            model: Model name
            confidence: Confidence threshold
            iou: IoU threshold
            dpi: DPI setting
            imgsz: Image size for preprocessing

        Returns:
            List of detection results (dicts) or None if not cached
        """
        cache_key = self._compute_cache_key(pdf_path, model, confidence, iou, dpi, imgsz)

        row = self.conn.query_one(
            "SELECT results_compressed, access_count FROM detection_cache WHERE cache_key = ?",
            (cache_key,)
        )

        if row is None:
            return None

        results_compressed, access_count = row

        # Update access tracking (LRU)
        self.conn.execute(
            "UPDATE detection_cache SET last_accessed = ?, access_count = ? WHERE cache_key = ?",
            (datetime.now().isoformat(), access_count + 1, cache_key)
        )

        # Decompress and deserialize
        if self.compression:
            json_bytes = gzip.decompress(results_compressed)
        else:
            json_bytes = results_compressed

        results = json.loads(json_bytes.decode('utf-8'))
        return results

    def set(
        self,
        pdf_path: Path,
        model: str,
        confidence: float,
        iou: float,
        dpi: int,
        imgsz: int,
        results: List[Dict[str, Any]],
    ):
        """
        Cache detection results for PDF with specific parameters.

        Args:
            pdf_path: Path to PDF file
            model: Model name
            confidence: Confidence threshold
            iou: IoU threshold
            dpi: DPI setting
            imgsz: Image size for preprocessing
            results: Detection results (list of dicts from DetectionResult.to_dict())
        """
        cache_key = self._compute_cache_key(pdf_path, model, confidence, iou, dpi, imgsz)
        stat = pdf_path.stat()

        # Serialize and compress
        json_bytes = json.dumps(results, separators=(",", ":")).encode('utf-8')

        if self.compression:
            results_compressed = gzip.compress(json_bytes, compresslevel=6)
        else:
            results_compressed = json_bytes

        compressed_size = len(results_compressed)
        num_pages = len(results)

        # Insert or replace
        self.conn.execute(
            """
            INSERT OR REPLACE INTO detection_cache (
                cache_key, pdf_name, pdf_size, pdf_mtime,
                model, confidence, iou, dpi, imgsz,
                num_pages, results_compressed, compressed_size,
                cached_at, last_accessed, access_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                pdf_path.name,
                stat.st_size,
                stat.st_mtime,
                model,
                round(confidence, 3),
                round(iou, 3),
                dpi,
                imgsz,
                num_pages,
                results_compressed,
                compressed_size,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                0,  # access_count starts at 0
            )
        )

    def has(
        self,
        pdf_path: Path,
        model: str,
        confidence: float,
        iou: float,
        dpi: int,
        imgsz: int,
    ) -> bool:
        """Check if results are cached for given PDF and parameters."""
        cache_key = self._compute_cache_key(pdf_path, model, confidence, iou, dpi, imgsz)

        row = self.conn.query_one(
            "SELECT 1 FROM detection_cache WHERE cache_key = ?",
            (cache_key,)
        )

        return row is not None

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        rows = self.conn.query_all(
            """
            SELECT
                COUNT(*) as num_entries,
                COUNT(DISTINCT pdf_name) as num_pdfs,
                SUM(num_pages) as total_pages,
                SUM(compressed_size) as total_bytes,
                AVG(access_count) as avg_access_count,
                COUNT(DISTINCT model) as num_models
            FROM detection_cache
            """
        )

        if not rows or rows[0][0] == 0:
            return {
                "num_entries": 0,
                "num_pdfs": 0,
                "total_pages": 0,
                "size_mb": 0.0,
                "avg_access_count": 0.0,
                "num_models": 0,
            }

        row = rows[0]
        return {
            "num_entries": row[0] or 0,
            "num_pdfs": row[1] or 0,
            "total_pages": row[2] or 0,
            "size_mb": (row[3] or 0) / 1024 / 1024,
            "avg_access_count": row[4] or 0.0,
            "num_models": row[5] or 0,
        }

    def clear(self):
        """Clear entire cache."""
        self.conn.execute("DELETE FROM detection_cache")

    def _auto_cleanup(self):
        """
        Automatic cleanup based on size and age limits.

        Strategy:
        1. Delete entries older than max_age_days
        2. If still over size limit, delete least recently used (LRU)
        """
        # Delete old entries
        if self.max_age_days is not None:
            cutoff_date = datetime.now().timestamp() - (self.max_age_days * 24 * 3600)
            self.conn.execute(
                "DELETE FROM detection_cache WHERE strftime('%s', cached_at) < ?",
                (int(cutoff_date),)
            )

        # Check size
        stats = self.stats()
        size_gb = stats["size_mb"] / 1024

        if size_gb > self.max_size_gb:
            # Delete LRU entries until under limit
            target_size = self.max_size_gb * 0.8  # Leave 20% buffer

            # Delete least recently used first
            self.conn.execute(
                """
                DELETE FROM detection_cache
                WHERE cache_key IN (
                    SELECT cache_key FROM detection_cache
                    ORDER BY last_accessed ASC
                    LIMIT (SELECT COUNT(*) * 0.2 FROM detection_cache)
                )
                """
            )

    def cleanup(self, strategy: str = "lru"):
        """
        Manual cleanup with specified strategy.

        Args:
            strategy: 'lru' (least recently used), 'oldest', or 'largest'
        """
        if strategy == "lru":
            # Delete least recently accessed 20%
            self.conn.execute(
                """
                DELETE FROM detection_cache
                WHERE cache_key IN (
                    SELECT cache_key FROM detection_cache
                    ORDER BY last_accessed ASC
                    LIMIT (SELECT COUNT(*) * 0.2 FROM detection_cache)
                )
                """
            )
        elif strategy == "oldest":
            # Delete oldest 20%
            self.conn.execute(
                """
                DELETE FROM detection_cache
                WHERE cache_key IN (
                    SELECT cache_key FROM detection_cache
                    ORDER BY cached_at ASC
                    LIMIT (SELECT COUNT(*) * 0.2 FROM detection_cache)
                )
                """
            )
        elif strategy == "largest":
            # Delete largest 20%
            self.conn.execute(
                """
                DELETE FROM detection_cache
                WHERE cache_key IN (
                    SELECT cache_key FROM detection_cache
                    ORDER BY compressed_size DESC
                    LIMIT (SELECT COUNT(*) * 0.2 FROM detection_cache)
                )
                """
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def delete_by_model(self, model: str):
        """Delete all cache entries for specific model."""
        self.conn.execute(
            "DELETE FROM detection_cache WHERE model = ?",
            (model,)
        )

    def list_models(self) -> List[str]:
        """List all models in cache."""
        rows = self.conn.query_all(
            "SELECT DISTINCT model FROM detection_cache ORDER BY model"
        )
        return [row[0] for row in rows]

    def export(self, output_file: Path):
        """Export cache to JSON for debugging/backup."""
        rows = self.conn.query_all(
            """
            SELECT
                cache_key, pdf_name, model, confidence, iou, dpi, imgsz,
                num_pages, cached_at, last_accessed, access_count
            FROM detection_cache
            ORDER BY cached_at DESC
            """
        )

        entries = []
        for row in rows:
            entries.append({
                "cache_key": row[0],
                "pdf_name": row[1],
                "model": row[2],
                "confidence": row[3],
                "iou": row[4],
                "dpi": row[5],
                "imgsz": row[6],
                "num_pages": row[7],
                "cached_at": row[8],
                "last_accessed": row[9],
                "access_count": row[10],
            })

        with open(output_file, 'w') as f:
            json.dump({
                "stats": self.stats(),
                "entries": entries
            }, f, indent=2)
