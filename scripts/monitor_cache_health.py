#!/usr/bin/env python3
"""
Monitor cache health and alert on corruption.

Run daily via cron:
  0 2 * * * cd ~/Documents/dh4pmp/packages/diagram-detector && python scripts/monitor_cache_health.py
"""

import sqlite3
from pathlib import Path
from datetime import datetime


def check_cache_health(cache_db: Path) -> dict:
    """
    Check cache health by counting corrupted entries.

    Args:
        cache_db: Path to detection cache database

    Returns:
        Dictionary with health statistics
    """
    conn = sqlite3.connect(str(cache_db))
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN compressed_size = 22 THEN 1 END) as empty_arrays,
            COUNT(CASE WHEN total_pages IS NULL THEN 1 END) as missing_pages
        FROM detection_cache
        WHERE model = 'diagram-detector-v5'
    """)

    total, empty, missing = cursor.fetchone()
    conn.close()

    corruption_pct = (empty / total * 100) if total > 0 else 0

    return {
        'total': total,
        'empty_arrays': empty,
        'missing_pages': missing,
        'corruption_pct': corruption_pct,
        'healthy': corruption_pct < 5
    }


def main():
    """Main entry point."""
    cache_db = Path.home() / ".cache" / "diagram-detector" / "detection_cache.db"

    if not cache_db.exists():
        print("Cache database not found")
        return

    health = check_cache_health(cache_db)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Cache Health Report")
    print(f"  Total entries: {health['total']:,}")
    print(f"  Empty arrays:  {health['empty_arrays']:,} ({health['corruption_pct']:.1f}%)")
    print(f"  Missing pages: {health['missing_pages']:,}")
    print(f"  Status: {'✓ HEALTHY' if health['healthy'] else '✗ CORRUPTED'}")

    if not health['healthy']:
        print("\n⚠ WARNING: Cache corruption detected!")
        print("  Consider clearing corrupted entries:")
        print(f"  sqlite3 {cache_db} 'DELETE FROM detection_cache WHERE compressed_size = 22;'")


if __name__ == "__main__":
    main()
