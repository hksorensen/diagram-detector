#!/usr/bin/env python3
"""
Test Batched Detection Methods

Tests both local batched detection and remote detection to verify they work correctly.
"""

import sys
from pathlib import Path
from diagram_detector import DiagramDetector
from diagram_detector.remote_pdf import PDFRemoteDetector
from diagram_detector.remote_ssh import RemoteConfig

def test_local_batched_detection():
    """Test local batched detection (detect_pdf called multiple times)."""
    print("=" * 70)
    print("TEST 1: LOCAL BATCHED DETECTION (Sequential)")
    print("=" * 70)

    # Find some test PDFs
    test_dir = Path("/Users/fvb832/Documents/dh4pmp/research/diagrams_in_arxiv/data/pdf_downloads")
    pdfs = sorted(list(test_dir.glob("*.pdf")))[:3]  # Test with 3 PDFs

    if not pdfs:
        print("⚠ No PDFs found for testing")
        return False

    print(f"\nTesting with {len(pdfs)} PDFs:")
    for pdf in pdfs:
        print(f"  - {pdf.name}")

    try:
        # Initialize detector with optimal parameters from grid search
        detector = DiagramDetector(
            model="v5",      # Use cached v5 model
            confidence=0.1,  # Optimal from grid search
            iou=0.3,         # Optimal from grid search
            device="auto",
            verbose=True
        )

        print(f"\n✓ Detector initialized")
        print(f"  Model: {detector.model_name}")
        print(f"  Confidence: {detector.confidence}")
        print(f"  IoU: {detector.iou}")
        print(f"  Device: {detector.device}")
        print(f"  Batch size: {detector.batch_size}")

        # Process each PDF
        all_results = {}
        for pdf_path in pdfs:
            print(f"\n--- Processing: {pdf_path.name} ---")
            results = detector.detect_pdf(pdf_path, dpi=200)

            diagrams = sum(r.count for r in results)
            pages_with_diagrams = sum(1 for r in results if r.has_diagram)

            print(f"✓ {len(results)} pages processed")
            print(f"  Diagrams found: {diagrams}")
            print(f"  Pages with diagrams: {pages_with_diagrams}/{len(results)}")

            all_results[pdf_path.name] = results

        # Summary
        total_pages = sum(len(results) for results in all_results.values())
        total_diagrams = sum(sum(r.count for r in results) for results in all_results.values())

        print(f"\n{'='*70}")
        print("LOCAL BATCHED DETECTION - SUMMARY")
        print(f"{'='*70}")
        print(f"✓ PDFs processed: {len(all_results)}")
        print(f"✓ Total pages: {total_pages}")
        print(f"✓ Total diagrams: {total_diagrams}")
        print(f"{'='*70}\n")

        return True

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_remote_batched_detection():
    """Test remote batched detection (PDFRemoteDetector.detect_pdfs)."""
    print("\n" + "=" * 70)
    print("TEST 2: REMOTE BATCHED DETECTION (PDFRemoteDetector)")
    print("=" * 70)

    # Check if remote is available (try both VPN port 8022 and local port 22)
    try:
        import socket

        # Try VPN port first (8022)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('thinkcentre.local', 8022))
        sock.close()

        # If VPN port fails, try local network port (22)
        if result != 0:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('thinkcentre.local', 22))
            sock.close()

        if result != 0:
            print("\n⚠ Remote server not reachable at thinkcentre.local:8022 or :22")
            print("  Skipping remote test (this is OK if you don't have remote setup)")
            return None  # Not a failure, just skipped
    except Exception as e:
        print(f"\n⚠ Cannot check remote connectivity: {e}")
        print("  Skipping remote test")
        return None

    # Find test PDFs
    test_dir = Path("/Users/fvb832/Documents/dh4pmp/research/diagrams_in_arxiv/data/pdf_downloads")
    pdfs = sorted(list(test_dir.glob("*.pdf")))[:3]  # Test with 3 PDFs

    if not pdfs:
        print("⚠ No PDFs found for testing")
        return False

    print(f"\nTesting with {len(pdfs)} PDFs:")
    for pdf in pdfs:
        print(f"  - {pdf.name}")

    try:
        # Initialize remote detector (uses default port 8022 for VPN)
        config = RemoteConfig()  # Defaults: port=8022, python_path="python3"

        detector = PDFRemoteDetector(
            config=config,
            batch_size=3,  # Process all 3 PDFs in one batch
            model="v5",    # Use cached v5 model
            confidence=0.1,  # Optimal from grid search
            dpi=200,
            imgsz=640,
            verbose=True
        )

        print(f"\n✓ Remote detector initialized")
        print(f"  Remote: {config.ssh_target}")
        print(f"  Model: {detector.model}")
        print(f"  Batch size: {detector.batch_size} PDFs/batch")

        # Process all PDFs in batch
        print(f"\n--- Processing batch of {len(pdfs)} PDFs ---")
        results_dict = detector.detect_pdfs(
            pdfs,
            use_cache=True,  # Enable caching
            force_reprocess=False
        )

        # Summary (handle both DetectionResult objects and dicts from cache)
        total_pages = sum(len(results) for results in results_dict.values())
        total_diagrams = sum(
            sum(r['count'] if isinstance(r, dict) else r.count for r in results)
            for results in results_dict.values()
        )
        pages_with_diagrams = sum(
            sum(1 for r in results if (r['has_diagram'] if isinstance(r, dict) else r.has_diagram))
            for results in results_dict.values()
        )

        print(f"\n{'='*70}")
        print("REMOTE BATCHED DETECTION - SUMMARY")
        print(f"{'='*70}")
        print(f"✓ PDFs processed: {len(results_dict)}")
        print(f"✓ Total pages: {total_pages}")
        print(f"✓ Pages with diagrams: {pages_with_diagrams}")
        print(f"✓ Total diagrams: {total_diagrams}")
        print(f"{'='*70}\n")

        return True

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("BATCHED DETECTION VERIFICATION TESTS")
    print("=" * 70)
    print()

    results = {}

    # Test 1: Local batched (sequential)
    results['local'] = test_local_batched_detection()

    # Test 2: Remote batched (parallel)
    results['remote'] = test_remote_batched_detection()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    if results['local']:
        print("✓ Local batched detection: WORKS")
    else:
        print("✗ Local batched detection: FAILED")

    if results['remote'] is True:
        print("✓ Remote batched detection: WORKS")
    elif results['remote'] is None:
        print("⊙ Remote batched detection: SKIPPED (no remote server)")
    else:
        print("✗ Remote batched detection: FAILED")

    print("=" * 70)
    print()

    # Return exit code
    if results['local'] and (results['remote'] is None or results['remote']):
        print("✓ All tests passed (or skipped)")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
