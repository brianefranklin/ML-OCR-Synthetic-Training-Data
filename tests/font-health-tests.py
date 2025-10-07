"""
Comprehensive test suite for the FontHealthManager system.

Tests cover:
- Basic health scoring mechanics
- Cooldown behavior
- Character coverage tracking
- Weighted font selection
- Persistence and recovery
- Integration with OCR generation
- Edge cases and error handling
"""

import unittest
import tempfile
import time
import os
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from font_health_manager import FontHealthManager, FontHealth


class TestFontHealth(unittest.TestCase):
    """Test the FontHealth dataclass."""
    
    def test_font_health_initialization(self):
        """Test FontHealth initializes with correct defaults."""
        font = FontHealth(font_path="/fonts/Arial.ttf")
        
        self.assertEqual(font.font_path, "/fonts/Arial.ttf")
        self.assertEqual(font.health_score, 100.0)
        self.assertEqual(font.success_count, 0)
        self.assertEqual(font.failure_count, 0)
        self.assertEqual(font.consecutive_failures, 0)
        self.assertIsNone(font.last_failure_time)
        self.assertIsNone(font.last_success_time)
        self.assertEqual(len(font.failure_reasons), 0)
        self.assertEqual(len(font.character_coverage), 0)
        self.assertIsNone(font.cooldown_until)
    
    def test_cooldown_check(self):
        """Test cooldown detection."""
        font = FontHealth(font_path="/fonts/Arial.ttf")
        
        # No cooldown initially
        self.assertFalse(font.is_in_cooldown())
        
        # Set future cooldown
        font.cooldown_until = time.time() + 10
        self.assertTrue(font.is_in_cooldown())
        
        # Set past cooldown
        font.cooldown_until = time.time() - 10
        self.assertFalse(font.is_in_cooldown())
    
    def test_serialization(self):
        """Test to_dict and from_dict methods."""
        font = FontHealth(font_path="/fonts/Arial.ttf")
        font.health_score = 75.0
        font.success_count = 10
        font.failure_count = 5
        font.character_coverage = {'a', 'b', 'c'}
        font.failure_reasons = {'render_error': 3, 'missing_glyphs': 2}
        
        # Serialize
        data = font.to_dict()
        
        # Deserialize
        font2 = FontHealth.from_dict(data)
        
        self.assertEqual(font.font_path, font2.font_path)
        self.assertEqual(font.health_score, font2.health_score)
        self.assertEqual(font.success_count, font2.success_count)
        self.assertEqual(font.failure_count, font2.failure_count)
        self.assertEqual(font.character_coverage, font2.character_coverage)
        self.assertEqual(font.failure_reasons, font2.failure_reasons)


class TestFontHealthManager(unittest.TestCase):
    """Test the FontHealthManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.temp_file.close()
        
        self.manager = FontHealthManager(
            health_file=self.temp_file.name,
            min_health_threshold=30.0,
            success_increment=2.0,
            failure_decrement=10.0,
            cooldown_base_seconds=1.0,  # Short for testing
            auto_save_interval=5
        )
        
        # Sample font paths
        self.fonts = [
            "/fonts/Arial.ttf",
            "/fonts/Times.ttf",
            "/fonts/Comic.ttf",
            "/fonts/KumarOne-Regular.ttf"  # Problematic font
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_register_font(self):
        """Test font registration."""
        self.assertEqual(len(self.manager.fonts), 0)
        
        self.manager.register_font(self.fonts[0])
        self.assertEqual(len(self.manager.fonts), 1)
        self.assertIn(self.fonts[0], self.manager.fonts)
        
        # Registering again shouldn't duplicate
        self.manager.register_font(self.fonts[0])
        self.assertEqual(len(self.manager.fonts), 1)
    
    def test_record_success(self):
        """Test recording successful operations."""
        font_path = self.fonts[0]
        text = "Hello World"
        
        self.manager.record_success(font_path, text)
        
        font = self.manager.fonts[font_path]
        self.assertEqual(font.success_count, 1)
        self.assertEqual(font.health_score, 100.0)  # Already at max
        self.assertEqual(font.consecutive_failures, 0)
        self.assertIsNotNone(font.last_success_time)
        self.assertEqual(font.character_coverage, set("Hello World"))
    
    def test_record_success_increases_health(self):
        """Test that success increases health when below max."""
        font_path = self.fonts[0]
        
        # Manually lower health
        self.manager.register_font(font_path)
        self.manager.fonts[font_path].health_score = 50.0
        
        self.manager.record_success(font_path, "test")
        
        font = self.manager.fonts[font_path]
        self.assertEqual(font.health_score, 52.0)  # 50 + 2 (success_increment)
    
    def test_record_failure(self):
        """Test recording failed operations."""
        font_path = self.fonts[3]  # Problematic font
        
        self.manager.record_failure(font_path, reason="render_error")
        
        font = self.manager.fonts[font_path]
        self.assertEqual(font.failure_count, 1)
        self.assertEqual(font.health_score, 90.0)  # 100 - 10
        self.assertEqual(font.consecutive_failures, 1)
        self.assertIsNotNone(font.last_failure_time)
        self.assertEqual(font.failure_reasons["render_error"], 1)
    
    def test_consecutive_failures_trigger_cooldown(self):
        """Test that consecutive failures trigger cooldown."""
        font_path = self.fonts[3]
        
        # First two failures don't trigger cooldown
        self.manager.record_failure(font_path, "error1")
        self.assertIsNone(self.manager.fonts[font_path].cooldown_until)
        
        self.manager.record_failure(font_path, "error2")
        self.assertIsNone(self.manager.fonts[font_path].cooldown_until)
        
        # Third consecutive failure triggers cooldown
        self.manager.record_failure(font_path, "error3")
        self.assertIsNotNone(self.manager.fonts[font_path].cooldown_until)
        self.assertTrue(self.manager.fonts[font_path].is_in_cooldown())
    
    def test_success_clears_cooldown(self):
        """Test that success clears cooldown."""
        font_path = self.fonts[3]
        
        # Put font in cooldown
        for _ in range(3):
            self.manager.record_failure(font_path, "error")
        
        self.assertTrue(self.manager.fonts[font_path].is_in_cooldown())
        
        # Success should clear cooldown
        self.manager.record_success(font_path, "test")
        self.assertFalse(self.manager.fonts[font_path].is_in_cooldown())
        self.assertEqual(self.manager.fonts[font_path].consecutive_failures, 0)
    
    def test_get_available_fonts(self):
        """Test getting available fonts."""
        # Register all fonts
        for font in self.fonts:
            self.manager.register_font(font)
        
        # All should be available initially
        available = self