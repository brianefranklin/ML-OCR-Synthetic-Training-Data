"""
Integration module for FontHealthManager with OCRDataGenerator

This module shows how to integrate the adaptive font health system
into the existing OCR data generation pipeline.
"""

import logging
from typing import List, Optional, Set
from PIL import ImageFont
from font_health_manager import FontHealthManager


def integrate_font_health_into_main(generator_main_module):
    """
    Monkey-patch or extend the main generation functions to use FontHealthManager.
    This function shows the integration points.
    """
    
    # Initialize global font health manager
    font_health_manager = FontHealthManager(
        health_file="font_health.json",
        min_health_threshold=30.0,
        success_increment=1.0,
        failure_decrement=10.0
    )
    
    # Store reference in module for access
    generator_main_module._font_health_manager = font_health_manager
    
    # Original can_font_render_text function
    original_can_font_render = generator_main_module.can_font_render_text
    
    def enhanced_can_font_render_text(font_path, text, character_set):
        """Enhanced version that tracks font health."""
        font_name = os.path.basename(font_path)
        
        # Check if font is in cooldown or unhealthy
        font_health_manager.register_font(font_path)
        font = font_health_manager.fonts[font_path]
        
        if font.is_in_cooldown():
            logging.debug(f"Font {font_name} is in cooldown, skipping")
            return False
        
        if font.health_score < font_health_manager.min_health_threshold:
            logging.debug(f"Font {font_name} health too low ({font.health_score:.1f}), skipping")
            return False
        
        # Try original validation
        try:
            result = original_can_font_render(font_path, text, character_set)
            if result:
                # Track successful validation (lightweight success)
                # Don't increment full score, just track coverage
                font.character_coverage.update(character_set)
            return result
        except Exception as e:
            # Record validation failure
            font_health_manager.record_failure(font_path, reason="validation_error")
            logging.warning(f"Font {font_name} validation failed: {e}")
            return False
    
    # Replace the function
    generator_main_module.can_font_render_text = enhanced_can_font_render_text
    
    return font_health_manager


def enhance_generate_with_batches(batch_generation_func, font_health_manager):
    """
    Wrapper for the generate_with_batches function that integrates font health tracking.
    """
    
    def wrapped_generate_with_batches(batch_config, font_files, background_images, args):
        # Filter available fonts before starting
        available_fonts = font_health_manager.get_available_fonts(font_files)
        
        if not available_fonts:
            logging.error("No healthy fonts available for generation")
            return
        
        logging.info(f"Starting generation with {len(available_fonts)}/{len(font_files)} healthy fonts")
        
        # Store original generate_image method
        from main import OCRDataGenerator
        original_generate_image = OCRDataGenerator.generate_image
        
        def tracked_generate_image(self, text, font_path, *args, **kwargs):
            """Wrapped generate_image that tracks success/failure."""
            try:
                result = original_generate_image(self, text, font_path, *args, **kwargs)
                # Record success
                font_health_manager.record_success(font_path, text)
                return result
            except OSError as e:
                if "execution context too long" in str(e):
                    font_health_manager.record_failure(font_path, reason="freetype_error")
                else:
                    font_health_manager.record_failure(font_path, reason="os_error")
                raise
            except Exception as e:
                font_health_manager.record_failure(font_path, reason=type(e).__name__)
                raise
        
        # Temporarily replace the method
        OCRDataGenerator.generate_image = tracked_generate_image
        
        try:
            # Call original function with filtered fonts
            result = batch_generation_func(batch_config, available_fonts, background_images, args)
            
            # Print health report after generation
            report = font_health_manager.get_summary_report()
            logging.info(f"Font health summary: {report}")
            
            return result
        finally:
            # Restore original method
            OCRDataGenerator.generate_image = original_generate_image
            # Save final state
            font_health_manager.save_state()
    
    return wrapped_generate_with_batches


def select_font_with_health(font_health_manager: FontHealthManager,
                           font_files: List[str],
                           text: str,
                           use_weighted: bool = True) -> Optional[str]:
    """
    Select a font using the health manager.
    
    Args:
        font_health_manager: The font health manager instance
        font_files: List of available font paths
        text: Text that will be rendered (for character checking)
        use_weighted: Use weighted selection (True) or just get best available (False)
    
    Returns:
        Selected font path or None if no suitable fonts
    """
    # Extract unique characters from text
    required_chars = set(text)
    
    if use_weighted:
        return font_health_manager.select_font_weighted(font_files, required_chars)
    else:
        available = font_health_manager.get_available_fonts(font_files, required_chars)
        return available[0] if available else None


# Modified section of main.py's generate_with_batches function
def enhanced_font_selection_snippet():
    """
    This shows how to modify the font selection in generate_with_batches
    to use the health manager.
    """
    # This would replace the section in generate_with_batches where font is selected
    
    # Instead of:
    # font_path = task['font_path']
    
    # Use:
    """
    # Get font using health-aware selection
    candidate_fonts = [task['font_path']]  # or task['available_fonts']
    font_path = select_font_with_health(
        font_health_manager,
        candidate_fonts,
        text_line,
        use_weighted=True
    )
    
    if not font_path:
        logging.warning(f"No healthy fonts available for batch '{batch_name}'")
        failed_attempts += 1
        continue
    """