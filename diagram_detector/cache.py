"""
SQLite-based Results Cache with Compression

Thread-safe caching for PDF processing results.
Uses SQLite for robust concurrent access and gzip for large results.
"""

import sqlite3
import json
import gzip
from pathlib import Path
from typing import Optional, List, Dict
import threading
from contextlib import contextmanager
from dataclasses import dataclass

from .models import DetectionResult, DiagramDetection


@dataclass
class CacheEntry:
    """Cache entry metadata."""
    pdf_name: str
    pdf_size: int
    pdf_mtime: float
    cache_key: str
    num_pages: int
    compressed_size: int
    cached_at: str


class SQLiteResultsCache:
    """
    Thread-safe SQLite-based cache for PDF results.

    Features:
    - Thread-safe (multiple processes can cache simultaneously)
    - Gzip compression (saves 70-90% space for JSON)
    - Fast lookups (indexed by cache key)
    - Metadata tracking
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache.

        Args:
            cache_dir: Cache directory (None = use default)
        """
        if cache_dir is None:
            cache_dir = (
                Path.home() / '.cache' / 'diagram-detector' / 'remote-results'
            )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.cache_dir / 'results.db'
        self._local = threading.local()

        # Initialize database
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._transaction() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pdf_results (
                    cache_key TEXT PRIMARY KEY,
                    pdf_name TEXT NOT NULL,
                    pdf_size INTEGER NOT NULL,
                    pdf_mtime REAL NOT NULL,
                    num_pages INTEGER NOT NULL,
                    results_compressed BLOB NOT NULL,
                    compressed_size INTEGER NOT NULL,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_pdf_name (pdf_name),
                    INDEX idx_cached_at (cached_at)
                )
            """)

    def _compute_cache_key(self, pdf_path: Path) -> str:
        """
        Compute cache key for PDF.

        Based on: name + size + mtime
        Fast check without reading file content.
        """
        stat = pdf_path.stat()
        import hashlib
        key_data = f"{pdf_path.name}_{stat.st_size}_{int(stat.st_mtime)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _compress_results(self, results: List[DetectionResult]) -> bytes:
        """Compress results to gzipped JSON."""
        data = [r.to_dict() for r in results]
        json_bytes = json.dumps(data, separators=(',', ':')).encode('utf-8')
        return gzip.compress(json_bytes, compresslevel=6)

    def _decompress_results(self, compressed: bytes) -> List[DetectionResult]:
        """Decompress results from gzipped JSON."""
        json_bytes = gzip.decompress(compressed)
        data = json.loads(json_bytes.decode('utf-8'))

        results = []
        for item in data:
            detections = [
                DiagramDetection(
                    bbox=tuple(d['bbox']),
                    confidence=d['confidence'],
                    class_name=d.get('class', 'diagram')
                )
                for d in item.get('detections', [])
            ]

            result = DetectionResult(
                filename=item['filename'],
                page_number=item.get('page_number'),
                detections=detections,
                image_width=item.get('image_width', 0),
                image_height=item.get('image_height', 0),
            )

            results.append(result)

        return results

    def get(self, pdf_path: Path) -> Optional[List[DetectionResult]]:
        """
        Get cached results for PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Cached results or None if not found/outdated
        """
        cache_key = self._compute_cache_key(pdf_path)

        with self._transaction() as conn:
            cursor = conn.execute(
                "SELECT results_compressed FROM pdf_results "
                "WHERE cache_key = ?",
                (cache_key,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            try:
                return self._decompress_results(row['results_compressed'])
            except Exception:
                # Corrupted cache entry - remove it
                conn.execute(
                    "DELETE FROM pdf_results WHERE cache_key = ?",
                    (cache_key,)
                )
                return None

    def set(self, pdf_path: Path, results: List[DetectionResult]) -> None:
        """
        Cache results for PDF.

        Args:
            pdf_path: Path to PDF file
            results: Detection results
        """
        cache_key = self._compute_cache_key(pdf_path)
        stat = pdf_path.stat()
        compressed = self._compress_results(results)

        with self._transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO pdf_results
                (cache_key, pdf_name, pdf_size, pdf_mtime, num_pages,
                 results_compressed, compressed_size)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key,
                pdf_path.name,
                stat.st_size,
                stat.st_mtime,
                len(results),
                compressed,
                len(compressed)
            ))

    def has(self, pdf_path: Path) -> bool:
        """Check if PDF is cached."""
        cache_key = self._compute_cache_key(pdf_path)

        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT 1 FROM pdf_results WHERE cache_key = ?",
            (cache_key,)
        )
        return cursor.fetchone() is not None

    def delete(self, pdf_path: Path) -> bool:
        """
        Delete cached results for PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if deleted, False if not found
        """
        cache_key = self._compute_cache_key(pdf_path)

        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM pdf_results WHERE cache_key = ?",
                (cache_key,)
            )
            return cursor.rowcount > 0

    def clear(self) -> int:
        """
        Clear entire cache.

        Returns:
            Number of entries deleted
        """
        with self._transaction() as conn:
            cursor = conn.execute("DELETE FROM pdf_results")
            return cursor.rowcount

    def stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        conn = self._get_connection()

        # Count entries
        cursor = conn.execute("SELECT COUNT(*) as count FROM pdf_results")
        num_cached = cursor.fetchone()['count']

        # Total compressed size
        cursor = conn.execute(
            "SELECT SUM(compressed_size) as total FROM pdf_results"
        )
        total_compressed = cursor.fetchone()['total'] or 0

        # Total pages
        cursor = conn.execute(
            "SELECT SUM(num_pages) as total FROM pdf_results"
        )
        total_pages = cursor.fetchone()['total'] or 0

        # Database file size
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            'num_cached_pdfs': num_cached,
            'total_pages': total_pages,
            'compressed_size_mb': total_compressed / (1024**2),
            'db_size_mb': db_size / (1024**2),
            'cache_dir': str(self.cache_dir),
            'avg_pages_per_pdf': (
                total_pages / num_cached if num_cached > 0 else 0
            ),
        }

    def list_cached(self, limit: int = 100) -> List[CacheEntry]:
        """
        List cached PDFs.

        Args:
            limit: Maximum number to return

        Returns:
            List of cache entries
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT
                cache_key, pdf_name, pdf_size, pdf_mtime,
                num_pages, compressed_size, cached_at
            FROM pdf_results
            ORDER BY cached_at DESC
            LIMIT ?
        """, (limit,))

        entries = []
        for row in cursor.fetchall():
            entries.append(CacheEntry(
                cache_key=row['cache_key'],
                pdf_name=row['pdf_name'],
                pdf_size=row['pdf_size'],
                pdf_mtime=row['pdf_mtime'],
                num_pages=row['num_pages'],
                compressed_size=row['compressed_size'],
                cached_at=row['cached_at'],
            ))

        return entries

    def vacuum(self) -> None:
        """Optimize database (reclaim space after deletions)."""
        conn = self._get_connection()
        conn.execute("VACUUM")
        conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')
