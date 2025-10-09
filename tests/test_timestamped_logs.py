"""
Test cases for timestamped log file and font health file generation.
"""

import pytest
import subprocess
import os
import shutil
import glob
import json
import time
from pathlib import Path





def test_timestamped_log_file_creation(test_environment):
    """Test that a timestamped log file is created in the log directory."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    # Record the time before running the script
    before_time = time.strftime('%Y-%m-%d_%H-%M')

    # Run the script
    command = [
        "python3",
        str(script_path),
        "--text-file", test_environment["text_file"],
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--log-dir", test_environment["log_dir"],
        "--num-images", "1"
    ]

    result = subprocess.run(command, text=True, check=False, capture_output=True)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    # Check that a log file was created with timestamp
    log_dir = Path(test_environment["log_dir"])
    log_files = list(log_dir.glob("generation_*.log"))

    assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"

    log_file = log_files[0]
    # Verify the filename format
    assert log_file.name.startswith("generation_"), "Log file should start with 'generation_'"
    assert log_file.name.endswith(".log"), "Log file should end with '.log'"

    # Verify the timestamp is in the expected format (YYYY-MM-DD_HH-MM-SS)
    # Extract timestamp from filename: generation_YYYY-MM-DD_HH-MM-SS.log
    timestamp_part = log_file.stem.replace("generation_", "")
    parts = timestamp_part.split("_")
    assert len(parts) == 2, f"Expected timestamp format YYYY-MM-DD_HH-MM-SS, got {timestamp_part}"
    date_part, time_part = parts
    assert len(date_part.split("-")) == 3, "Date should have 3 parts (YYYY-MM-DD)"
    assert len(time_part.split("-")) == 3, "Time should have 3 parts (HH-MM-SS)"

    # Verify the log file contains expected content
    with open(log_file, 'r') as f:
        content = f.read()
        assert "Script started." in content
        assert "Script finished." in content


def test_timestamped_font_health_file_creation(test_environment):
    """Test that a timestamped font_health.json file is created in the log directory."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    # Run the script
    command = [
        "python3",
        str(script_path),
        "--text-file", test_environment["text_file"],
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--log-dir", test_environment["log_dir"],
        "--num-images", "2"
    ]

    result = subprocess.run(command, text=True, check=False, capture_output=True)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    # Check that a font_health.json file was created with timestamp
    log_dir = Path(test_environment["log_dir"])
    health_files = list(log_dir.glob("font_health_*.json"))

    assert len(health_files) == 1, f"Expected 1 font health file, found {len(health_files)}"

    health_file = health_files[0]
    # Verify the filename format
    assert health_file.name.startswith("font_health_"), "Font health file should start with 'font_health_'"
    assert health_file.name.endswith(".json"), "Font health file should end with '.json'"

    # Verify it's valid JSON
    with open(health_file, 'r') as f:
        health_data = json.load(f)
        assert "fonts" in health_data
        assert "metadata" in health_data


def test_matching_timestamps_for_log_and_font_health(test_environment):
    """Test that log file and font_health.json have the same timestamp."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    # Run the script
    command = [
        "python3",
        str(script_path),
        "--text-file", test_environment["text_file"],
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--log-dir", test_environment["log_dir"],
        "--num-images", "1"
    ]

    result = subprocess.run(command, text=True, check=False, capture_output=True)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    # Get the log file and font health file
    log_dir = Path(test_environment["log_dir"])
    log_files = list(log_dir.glob("generation_*.log"))
    health_files = list(log_dir.glob("font_health_*.json"))

    assert len(log_files) == 1, "Expected exactly 1 log file"
    assert len(health_files) == 1, "Expected exactly 1 font health file"

    # Extract timestamps from both files
    log_timestamp = log_files[0].stem.replace("generation_", "")
    health_timestamp = health_files[0].stem.replace("font_health_", "")

    # Timestamps should match exactly
    assert log_timestamp == health_timestamp, \
        f"Log timestamp ({log_timestamp}) should match font health timestamp ({health_timestamp})"


def test_multiple_runs_create_separate_files(test_environment):
    """Test that multiple runs create separate timestamped files."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    # Run the script twice with a small delay
    for i in range(2):
        command = [
            "python3",
            str(script_path),
            "--text-file", test_environment["text_file"],
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"],
            "--log-dir", test_environment["log_dir"],
            "--num-images", "1"
        ]

        result = subprocess.run(command, text=True, check=False, capture_output=True)
        assert result.returncode == 0, f"Script run {i+1} failed with error:\n{result.stderr}"

        # Small delay to ensure different timestamps
        if i == 0:
            time.sleep(1.5)

    # Check that two separate log files were created
    log_dir = Path(test_environment["log_dir"])
    log_files = list(log_dir.glob("generation_*.log"))
    health_files = list(log_dir.glob("font_health_*.json"))

    assert len(log_files) == 2, f"Expected 2 log files, found {len(log_files)}"
    assert len(health_files) == 2, f"Expected 2 font health files, found {len(health_files)}"

    # Verify that the files have different timestamps
    log_timestamps = [f.stem.replace("generation_", "") for f in log_files]
    assert log_timestamps[0] != log_timestamps[1], "Log files should have different timestamps"


def test_log_dir_created_if_not_exists(test_environment):
    """Test that the log directory is created if it doesn't exist."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    # Use a non-existent log directory
    new_log_dir = test_environment["tmp_path"] / "new_logs"
    assert not new_log_dir.exists(), "Log directory should not exist yet"

    # Run the script
    command = [
        "python3",
        str(script_path),
        "--text-file", test_environment["text_file"],
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--log-dir", str(new_log_dir),
        "--num-images", "1"
    ]

    result = subprocess.run(command, text=True, check=False, capture_output=True)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    # Verify the log directory was created
    assert new_log_dir.exists(), "Log directory should have been created"
    assert new_log_dir.is_dir(), "Log directory should be a directory"

    # Verify files were created in the new directory
    log_files = list(new_log_dir.glob("generation_*.log"))
    health_files = list(new_log_dir.glob("font_health_*.json"))

    assert len(log_files) == 1, "Expected 1 log file in new directory"
    assert len(health_files) == 1, "Expected 1 font health file in new directory"


def test_default_log_dir_is_logs(test_environment):
    """Test that the default log directory is 'logs' when not specified."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    # Create a temporary working directory
    work_dir = test_environment["tmp_path"] / "work"
    work_dir.mkdir()

    # Run the script without specifying --log-dir
    command = [
        "python3",
        str(script_path),
        "--text-file", test_environment["text_file"],
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--num-images", "1"
    ]

    # Run from the work directory
    result = subprocess.run(
        command,
        text=True,
        check=False,
        capture_output=True,
        cwd=str(work_dir)
    )
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    # The default 'logs' directory should be created in the working directory
    default_log_dir = work_dir / "logs"
    assert default_log_dir.exists(), "Default 'logs' directory should have been created"

    # Verify files were created
    log_files = list(default_log_dir.glob("generation_*.log"))
    health_files = list(default_log_dir.glob("font_health_*.json"))

    assert len(log_files) == 1, "Expected 1 log file in default logs directory"
    assert len(health_files) == 1, "Expected 1 font health file in default logs directory"
