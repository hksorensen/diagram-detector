"""
Diagnostic logging setup for debugging contamination and ordering issues.

Provides a separate logger for diagnostic messages (checkpoints, order verification, etc.)
that can be enabled/disabled independently from regular application logging.

Usage:
    from diagram_detector.diagnostic_logger import diagnostic

    diagnostic.info("CHECKPOINT 1: PASS - order preserved")
    diagnostic.warning("CHECKPOINT 2: FAIL - order mismatch detected")
"""

import logging
from pathlib import Path
from typing import Optional


def setup_diagnostic_logger(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = False,
) -> logging.Logger:
    """
    Set up diagnostic logger with file and optional console output.

    Args:
        log_file: Path to diagnostic log file (None = disabled)
        level: Logging level (default: INFO)
        console: Also log to console (default: False, file only)

    Returns:
        Configured diagnostic logger

    Example:
        # Enable diagnostics to file only
        diagnostic = setup_diagnostic_logger(Path('diagnostics.log'))

        # Enable diagnostics to console and file
        diagnostic = setup_diagnostic_logger(Path('diagnostics.log'), console=True)

        # Disable diagnostics
        diagnostic = setup_diagnostic_logger(None)
    """
    logger = logging.getLogger('diagnostic')

    # Clear any existing handlers
    logger.handlers.clear()
    logger.setLevel(level)

    # Prevent propagation to root logger (avoid duplicate messages)
    logger.propagate = False

    if log_file is None:
        # Disabled - add null handler to prevent warnings
        logger.addHandler(logging.NullHandler())
        return logger

    # File handler - always enabled when log_file provided
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(level)

    # Detailed format for file (includes timestamp, level, location)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Optional console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Simpler format for console (no timestamp, shorter)
        console_formatter = logging.Formatter(
            '[DIAGNOSTIC] %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


# Global diagnostic logger instance (disabled by default)
diagnostic = logging.getLogger('diagnostic')
diagnostic.addHandler(logging.NullHandler())
diagnostic.propagate = False


def enable_diagnostics(log_file: Path, console: bool = False, level: int = logging.INFO):
    """
    Enable diagnostic logging (convenience function).

    Args:
        log_file: Path to diagnostic log file
        console: Also log to console (default: False)
        level: Logging level (default: INFO)
    """
    global diagnostic
    diagnostic = setup_diagnostic_logger(log_file, level=level, console=console)


def disable_diagnostics():
    """Disable diagnostic logging (convenience function)."""
    global diagnostic
    diagnostic = setup_diagnostic_logger(None)
