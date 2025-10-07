"""
Test cases for logging setup functionality.
"""

import pytest
import os
import logging
import tempfile
import shutil
from pathlib import Path


# Import the setup_logging function
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from main import setup_logging


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before and after each test."""
    # Clear existing handlers before test
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)

    yield

    # Clear handlers after test
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)

    # Reset logging level
    logging.root.setLevel(logging.WARNING)


def test_setup_logging_creates_missing_directory(temp_dir):
    """Test that setup_logging creates parent directory if it doesn't exist."""
    # Create a log file path in a non-existent subdirectory
    non_existent_dir = os.path.join(temp_dir, "logs", "subdir")
    log_file = os.path.join(non_existent_dir, "test.log")

    # Verify directory doesn't exist
    assert not os.path.exists(non_existent_dir)

    # Call setup_logging
    setup_logging("INFO", log_file)

    # Verify directory was created
    assert os.path.exists(non_existent_dir)

    # Verify logging works (this creates the file)
    logging.info("Test message")

    # Flush handlers to ensure file is written
    for handler in logging.root.handlers:
        handler.flush()

    # Verify log file was created
    assert os.path.exists(log_file)

    # Check file contains the message
    with open(log_file, 'r') as f:
        content = f.read()
        assert "Test message" in content


def test_setup_logging_with_existing_directory(temp_dir):
    """Test that setup_logging works when directory already exists."""
    # Create a log directory
    log_dir = os.path.join(temp_dir, "logs")
    os.makedirs(log_dir)
    log_file = os.path.join(log_dir, "test.log")

    # Call setup_logging
    setup_logging("DEBUG", log_file)

    # Verify logging works
    logging.debug("Debug message")

    # Flush handlers to ensure file is written
    for handler in logging.root.handlers:
        handler.flush()

    # Verify log file was created
    assert os.path.exists(log_file)

    # Check file contains the message
    with open(log_file, 'r') as f:
        content = f.read()
        assert "Debug message" in content


def test_setup_logging_no_directory_in_path(temp_dir):
    """Test setup_logging with a log file in the current directory (no subdirs)."""
    # Create a log file path without subdirectory
    log_file = os.path.join(temp_dir, "test.log")

    # Call setup_logging
    setup_logging("WARNING", log_file)

    # Verify logging works
    logging.warning("Warning message")

    # Flush handlers to ensure file is written
    for handler in logging.root.handlers:
        handler.flush()

    # Verify log file was created
    assert os.path.exists(log_file)

    # Check file contains the message
    with open(log_file, 'r') as f:
        content = f.read()
        assert "Warning message" in content


def test_setup_logging_creates_nested_directories(temp_dir):
    """Test that setup_logging creates deeply nested directory structures."""
    # Create a log file path with multiple nested subdirectories
    nested_path = os.path.join(temp_dir, "logs", "year", "month", "day")
    log_file = os.path.join(nested_path, "test.log")

    # Verify none of the directories exist
    assert not os.path.exists(nested_path)

    # Call setup_logging
    setup_logging("ERROR", log_file)

    # Verify all directories were created
    assert os.path.exists(nested_path)

    # Verify logging works
    logging.error("Error message")

    # Flush handlers to ensure file is written
    for handler in logging.root.handlers:
        handler.flush()

    # Verify log file was created
    assert os.path.exists(log_file)

    # Check file contains the message
    with open(log_file, 'r') as f:
        content = f.read()
        assert "Error message" in content


def test_setup_logging_log_levels(temp_dir):
    """Test that different log levels work correctly."""
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

    for level in log_levels:
        # Create a unique log file for each level
        log_file = os.path.join(temp_dir, f"test_{level.lower()}.log")

        # Reset logging configuration
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)

        # Setup logging with this level
        setup_logging(level, log_file)

        # Verify the logger has the correct level
        assert logging.root.level == getattr(logging, level)

        # Log a message at this level
        getattr(logging, level.lower())(f"Test {level} message")

        # Flush handlers to ensure file is written
        for handler in logging.root.handlers:
            handler.flush()

        # Verify the message was logged
        with open(log_file, 'r') as f:
            content = f.read()
            assert f"Test {level} message" in content


def test_setup_logging_file_overwrite(temp_dir):
    """Test that setup_logging overwrites existing log file."""
    log_file = os.path.join(temp_dir, "test.log")

    # Create an existing log file with content
    with open(log_file, 'w') as f:
        f.write("Old content\n")

    # Verify old content exists
    with open(log_file, 'r') as f:
        assert "Old content" in f.read()

    # Call setup_logging (should overwrite with mode='w')
    setup_logging("INFO", log_file)

    # Log a new message
    logging.info("New message")

    # Flush handlers to ensure file is written
    for handler in logging.root.handlers:
        handler.flush()

    # Verify old content is gone and new content is present
    with open(log_file, 'r') as f:
        content = f.read()
        assert "Old content" not in content
        assert "New message" in content
