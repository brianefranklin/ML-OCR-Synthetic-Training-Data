"""
Font Health Management System

This module provides an adaptive font reliability tracking system that learns
which fonts work well and which ones are problematic. It replaces hardcoded
font blacklists with a dynamic scoring system.

Key Features:
- Health scoring (0-100) based on success/failure rates
- Exponential backoff cooldowns for problematic fonts
- Character coverage tracking per font
- Weighted selection favoring reliable fonts
- Persistent state across runs
"""

import os
import time
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional
from collections import defaultdict


@dataclass
class FontHealth:
    """
    Tracks health metrics for a single font.

    Attributes:
        font_path: Path to the font file
        health_score: Current health score (0-100)
        success_count: Total successful generations
        failure_count: Total failed generations
        consecutive_failures: Number of consecutive failures
        last_failure_time: Timestamp of last failure
        last_success_time: Timestamp of last success
        failure_reasons: Dict mapping failure reasons to counts
        character_coverage: Set of characters successfully rendered
        cooldown_until: Timestamp when cooldown expires (None if not in cooldown)
    """
    font_path: str
    health_score: float = 100.0
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    failure_reasons: Dict[str, int] = field(default_factory=dict)
    character_coverage: Set[str] = field(default_factory=set)
    cooldown_until: Optional[float] = None

    def is_in_cooldown(self) -> bool:
        """Check if font is currently in cooldown."""
        if self.cooldown_until is None:
            return False
        return time.time() < self.cooldown_until

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            'font_path': self.font_path,
            'health_score': self.health_score,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'consecutive_failures': self.consecutive_failures,
            'last_failure_time': self.last_failure_time,
            'last_success_time': self.last_success_time,
            'failure_reasons': self.failure_reasons,
            'character_coverage': list(self.character_coverage),
            'cooldown_until': self.cooldown_until
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FontHealth':
        """Deserialize from dictionary."""
        data = data.copy()
        # Convert character_coverage list back to set
        if 'character_coverage' in data:
            data['character_coverage'] = set(data['character_coverage'])
        return cls(**data)


class FontHealthManager:
    """
    Manages health tracking and selection for fonts.

    This class maintains health scores for fonts and provides methods to:
    - Track successes and failures
    - Apply cooldowns to problematic fonts
    - Select fonts based on health scores
    - Persist state across runs
    """

    def __init__(self,
                 health_file: str = "font_health.json",
                 min_health_threshold: float = 30.0,
                 success_increment: float = 1.0,
                 failure_decrement: float = 10.0,
                 cooldown_base_seconds: float = 300.0,
                 auto_save_interval: int = 50):
        """
        Initialize the font health manager.

        Args:
            health_file: Path to JSON file for persisting health data
            min_health_threshold: Minimum health score for font to be used
            success_increment: Points to add on success
            failure_decrement: Points to subtract on failure
            cooldown_base_seconds: Base cooldown duration (doubles per consecutive failure)
            auto_save_interval: Save state every N operations
        """
        self.health_file = health_file
        self.min_health_threshold = min_health_threshold
        self.success_increment = success_increment
        self.failure_decrement = failure_decrement
        self.cooldown_base_seconds = cooldown_base_seconds
        self.auto_save_interval = auto_save_interval

        self.fonts: Dict[str, FontHealth] = {}
        self.operation_count = 0

        # Load existing state if available
        self.load_state()

    def register_font(self, font_path: str) -> None:
        """Register a font if not already tracked."""
        if font_path not in self.fonts:
            self.fonts[font_path] = FontHealth(font_path=font_path)
            logging.debug(f"Registered new font: {os.path.basename(font_path)}")

    def record_success(self, font_path: str, text: str = "") -> None:
        """
        Record a successful generation with this font.

        Args:
            font_path: Path to the font
            text: Text that was successfully rendered (for coverage tracking)
        """
        self.register_font(font_path)
        font = self.fonts[font_path]

        # Update metrics
        font.success_count += 1
        font.consecutive_failures = 0
        font.last_success_time = time.time()

        # Increase health (capped at 100)
        font.health_score = min(100.0, font.health_score + self.success_increment)

        # Track character coverage
        if text:
            font.character_coverage.update(text)

        # Clear cooldown on success
        font.cooldown_until = None

        self._auto_save()
        logging.debug(f"Font {os.path.basename(font_path)} success: health={font.health_score:.1f}")

    def record_failure(self, font_path: str, reason: str = "unknown") -> None:
        """
        Record a failed generation with this font.

        Args:
            font_path: Path to the font
            reason: Reason for failure (e.g., "freetype_error", "render_error")
        """
        self.register_font(font_path)
        font = self.fonts[font_path]

        # Update metrics
        font.failure_count += 1
        font.consecutive_failures += 1
        font.last_failure_time = time.time()

        # Track failure reason
        if reason not in font.failure_reasons:
            font.failure_reasons[reason] = 0
        font.failure_reasons[reason] += 1

        # Decrease health (min 0)
        font.health_score = max(0.0, font.health_score - self.failure_decrement)

        # Apply cooldown after 3 consecutive failures
        if font.consecutive_failures >= 3:
            # Exponential backoff: 2^(failures-3) * base_seconds
            cooldown_multiplier = 2 ** (font.consecutive_failures - 3)
            cooldown_duration = self.cooldown_base_seconds * cooldown_multiplier
            font.cooldown_until = time.time() + cooldown_duration
            logging.warning(
                f"Font {os.path.basename(font_path)} entered cooldown for "
                f"{cooldown_duration:.0f}s after {font.consecutive_failures} failures"
            )

        self._auto_save()
        logging.debug(
            f"Font {os.path.basename(font_path)} failure ({reason}): "
            f"health={font.health_score:.1f}, consecutive={font.consecutive_failures}"
        )

    def get_available_fonts(self,
                           font_paths: List[str],
                           required_chars: Optional[Set[str]] = None) -> List[str]:
        """
        Get list of healthy fonts that can render the required characters.

        Args:
            font_paths: List of font paths to filter
            required_chars: Optional set of characters that must be supported

        Returns:
            List of available font paths, sorted by health score (best first)
        """
        available = []

        for font_path in font_paths:
            self.register_font(font_path)
            font = self.fonts[font_path]

            # Skip if in cooldown
            if font.is_in_cooldown():
                continue

            # Skip if health too low
            if font.health_score < self.min_health_threshold:
                continue

            # Check character coverage if required
            if required_chars and font.character_coverage:
                # Only use this font if it has successfully rendered these chars before
                if not required_chars.issubset(font.character_coverage):
                    continue

            available.append(font_path)

        # Sort by health score (descending)
        available.sort(key=lambda p: self.fonts[p].health_score, reverse=True)

        return available

    def select_font_weighted(self,
                            font_paths: List[str],
                            required_chars: Optional[Set[str]] = None) -> Optional[str]:
        """
        Select a font using weighted random selection based on health scores.

        Fonts with higher health scores are more likely to be selected.
        Uses quadratic weighting (health²) to strongly prefer reliable fonts.

        Args:
            font_paths: List of candidate font paths
            required_chars: Optional set of required characters

        Returns:
            Selected font path or None if no suitable fonts
        """
        available = self.get_available_fonts(font_paths, required_chars)

        if not available:
            return None

        # Calculate weights (health_score²)
        weights = []
        for font_path in available:
            health = self.fonts[font_path].health_score
            weight = health ** 2  # Quadratic weighting
            weights.append(weight)

        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            # All fonts have 0 health, just pick randomly
            return random.choice(available)

        rand_val = random.uniform(0, total_weight)
        cumulative = 0
        for font_path, weight in zip(available, weights):
            cumulative += weight
            if rand_val <= cumulative:
                return font_path

        # Fallback (shouldn't happen)
        return available[0]

    def save_state(self) -> None:
        """Save current state to JSON file."""
        try:
            data = {
                'fonts': {path: font.to_dict() for path, font in self.fonts.items()},
                'metadata': {
                    'last_save': time.time(),
                    'total_fonts': len(self.fonts)
                }
            }

            with open(self.health_file, 'w') as f:
                json.dump(data, f, indent=2)

            logging.debug(f"Saved font health state to {self.health_file}")
        except Exception as e:
            logging.error(f"Failed to save font health state: {e}")

    def load_state(self) -> None:
        """Load state from JSON file if it exists."""
        if not os.path.exists(self.health_file):
            logging.info("No existing font health file, starting fresh")
            return

        try:
            with open(self.health_file, 'r') as f:
                data = json.load(f)

            # Load font data
            for font_path, font_data in data.get('fonts', {}).items():
                self.fonts[font_path] = FontHealth.from_dict(font_data)

            logging.info(f"Loaded font health for {len(self.fonts)} fonts from {self.health_file}")
        except Exception as e:
            logging.error(f"Failed to load font health state: {e}, starting fresh")

    def get_summary_report(self) -> dict:
        """Get summary statistics about font health."""
        if not self.fonts:
            return {'total_fonts': 0}

        total_fonts = len(self.fonts)
        healthy_fonts = sum(1 for f in self.fonts.values()
                          if f.health_score >= self.min_health_threshold and not f.is_in_cooldown())
        in_cooldown = sum(1 for f in self.fonts.values() if f.is_in_cooldown())

        health_scores = [f.health_score for f in self.fonts.values()]
        avg_health = sum(health_scores) / len(health_scores) if health_scores else 0

        total_successes = sum(f.success_count for f in self.fonts.values())
        total_failures = sum(f.failure_count for f in self.fonts.values())

        return {
            'total_fonts': total_fonts,
            'healthy_fonts': healthy_fonts,
            'in_cooldown': in_cooldown,
            'average_health': round(avg_health, 1),
            'total_successes': total_successes,
            'total_failures': total_failures,
            'success_rate': round(total_successes / max(1, total_successes + total_failures) * 100, 1)
        }

    def reset_font(self, font_path: str) -> None:
        """Reset a font's health to defaults."""
        if font_path in self.fonts:
            self.fonts[font_path] = FontHealth(font_path=font_path)
            logging.info(f"Reset health for font: {os.path.basename(font_path)}")

    def cleanup_stale_cooldowns(self) -> int:
        """Remove expired cooldowns and return count of cleanups."""
        count = 0
        for font in self.fonts.values():
            if font.cooldown_until and not font.is_in_cooldown():
                font.cooldown_until = None
                count += 1
        return count

    def _get_font_status(self, font: FontHealth) -> str:
        """Get human-readable status string for a font."""
        if font.is_in_cooldown():
            return "cooldown"
        elif font.health_score < self.min_health_threshold:
            return "unhealthy"
        elif font.health_score >= 80:
            return "healthy"
        else:
            return "fair"

    def _auto_save(self) -> None:
        """Auto-save state periodically."""
        self.operation_count += 1
        if self.operation_count >= self.auto_save_interval:
            self.save_state()
            self.operation_count = 0
