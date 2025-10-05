"""
Glyph Overlap Module for OCR Synthetic Data Generator
Implements realistic text rendering with kerning, ligatures, and overlap effects.
"""

import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np


@dataclass
class GlyphOverlapConfig:
    """Configuration for glyph overlap features."""
    # Kerning
    enable_kerning: bool = True
    kerning_intensity: float = 0.3  # 0.0 to 1.0, how aggressive kerning is
    
    # Ligatures
    enable_ligatures: bool = True
    ligature_probability: float = 0.4  # Probability of using ligatures when available
    
    # Ink effects
    enable_ink_bleed: bool = True
    ink_bleed_intensity: float = 0.2  # 0.0 to 1.0
    
    # Character connection (for cursive/script fonts)
    enable_connections: bool = True
    connection_probability: float = 0.3


class GlyphOverlapRenderer:
    """Handles realistic glyph rendering with overlap effects."""
    
    # Common kerning pairs in English
    KERN_PAIRS = {
        ('A', 'V'): -0.15, ('A', 'W'): -0.12, ('A', 'Y'): -0.18,
        ('A', 'v'): -0.08, ('A', 'w'): -0.08, ('A', 'y'): -0.10,
        ('F', 'a'): -0.08, ('F', 'e'): -0.08, ('F', 'o'): -0.08,
        ('L', 'T'): -0.10, ('L', 'V'): -0.12, ('L', 'W'): -0.10,
        ('L', 'Y'): -0.14, ('P', 'a'): -0.05, ('P', 'e'): -0.05,
        ('T', 'a'): -0.10, ('T', 'e'): -0.10, ('T', 'o'): -0.10,
        ('T', 'r'): -0.08, ('T', 'u'): -0.08, ('T', 'w'): -0.08,
        ('T', 'y'): -0.08, ('V', 'a'): -0.12, ('V', 'e'): -0.10,
        ('V', 'o'): -0.10, ('V', 'u'): -0.08, ('W', 'a'): -0.10,
        ('W', 'e'): -0.08, ('W', 'o'): -0.08, ('Y', 'a'): -0.14,
        ('Y', 'e'): -0.12, ('Y', 'o'): -0.12, ('Y', 'u'): -0.10,
        ('a', 'v'): -0.03, ('a', 'w'): -0.03, ('a', 'y'): -0.03,
        ('o', 'v'): -0.03, ('o', 'w'): -0.03, ('o', 'x'): -0.03,
        ('o', 'y'): -0.03, ('r', 'a'): -0.03, ('r', 'o'): -0.03,
        ('v', 'a'): -0.05, ('v', 'e'): -0.03, ('v', 'o'): -0.03,
        ('w', 'a'): -0.03, ('w', 'e'): -0.03, ('w', 'o'): -0.03,
        ('y', 'a'): -0.05, ('y', 'e'): -0.03, ('y', 'o'): -0.03,
    }
    
    # Ligature replacements (character sequences to ligature character)
    # Note: Actual ligatures depend on font support
    LIGATURES = {
        'fi': 'ﬁ', 'fl': 'ﬂ', 'ff': 'ﬀ', 'ffi': 'ﬃ', 'ffl': 'ﬄ',
        'st': 'ﬆ', 'ct': None,  # Some fonts have ct ligature
    }
    
    def __init__(self, config: GlyphOverlapConfig = None):
        """Initialize the glyph overlap renderer."""
        self.config = config or GlyphOverlapConfig()
    
    def apply_kerning(self, 
                     char1: str, 
                     char2: str, 
                     base_spacing: float,
                     font: ImageFont.FreeTypeFont) -> float:
        """
        Calculate kerning adjustment for a character pair.
        
        Args:
            char1: First character
            char2: Second character
            base_spacing: Base spacing between characters
            font: Font being used
            
        Returns:
            Adjusted spacing value
        """
        if not self.config.enable_kerning:
            return base_spacing
        
        # Check for kerning pair
        kern_pair = (char1, char2)
        if kern_pair in self.KERN_PAIRS:
            # Apply kerning with intensity factor
            kern_value = self.KERN_PAIRS[kern_pair] * self.config.kerning_intensity
            
            # Get character width for proportional adjustment
            temp_img = Image.new('RGBA', (1, 1))
            temp_draw = ImageDraw.Draw(temp_img)
            char_width = temp_draw.textlength(char1, font=font)
            
            # Apply kerning as proportion of character width
            adjustment = char_width * kern_value
            return max(0, base_spacing + adjustment)  # Ensure non-negative
        
        return base_spacing
    
    def check_ligature(self, text: str, position: int) -> Tuple[Optional[str], int]:
        """
        Check if a ligature should be applied at the current position.
        
        Args:
            text: Full text string
            position: Current position in text
            
        Returns:
            Tuple of (ligature_char or None, characters_consumed)
        """
        if not self.config.enable_ligatures:
            return None, 0
        
        # Check for multi-character ligatures first (longer sequences)
        for seq_len in [3, 2]:  # Check longer sequences first
            if position + seq_len <= len(text):
                sequence = text[position:position + seq_len]
                if sequence in self.LIGATURES:
                    # Apply with probability
                    if random.random() < self.config.ligature_probability:
                        ligature = self.LIGATURES[sequence]
                        if ligature:  # Some ligatures might not be available
                            return ligature, seq_len
        
        return None, 0
    
    def render_with_overlap(self,
                          text: str,
                          font: ImageFont.FreeTypeFont,
                          direction: str = 'left_to_right') -> Tuple[Image.Image, List[List[float]]]:
        """
        Render text with realistic glyph overlap effects.
        
        Args:
            text: Text to render
            font: Font to use
            direction: Text direction
            
        Returns:
            Tuple of (rendered_image, character_bboxes)
        """
        if direction != 'left_to_right':
            # For now, only implement for LTR
            # You can extend this for other directions
            return self._render_standard(text, font, direction)
        
        # Calculate initial dimensions
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        
        # Process text for ligatures and measure
        processed_chars = []
        char_mappings = []  # Maps processed chars to original positions
        i = 0
        while i < len(text):
            ligature, consumed = self.check_ligature(text, i)
            if ligature:
                processed_chars.append(ligature)
                char_mappings.append((i, i + consumed))
                i += consumed
            else:
                processed_chars.append(text[i])
                char_mappings.append((i, i + 1))
                i += 1
        
        # Measure characters with kerning
        char_positions = []
        x_offset = 20
        
        for i, char in enumerate(processed_chars):
            # Apply kerning if not first character
            if i > 0:
                prev_char = processed_chars[i-1]
                base_spacing = 0  # Start with no spacing
                adjusted_spacing = self.apply_kerning(prev_char, char, base_spacing, font)
                x_offset += adjusted_spacing
            
            # Store position
            char_positions.append(x_offset)
            
            # Add character width
            char_width = temp_draw.textlength(char, font=font)
            x_offset += char_width
        
        # Calculate image dimensions
        total_width = x_offset + 20
        text_bbox = temp_draw.textbbox((0, 0), ''.join(processed_chars), font=font)
        img_height = (text_bbox[3] - text_bbox[1]) + 30
        
        # Create actual image
        image = Image.new('RGB', (int(total_width), img_height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Render characters with overlap
        char_boxes = []
        y_offset = 15
        
        for i, (char, x_pos) in enumerate(zip(processed_chars, char_positions)):
            # Render character
            draw.text((x_pos, y_offset), char, font=font, fill='black')
            
            # Calculate bounding box
            char_bbox = draw.textbbox((x_pos, y_offset), char, font=font)
            
            # Store bbox for each original character
            # (handle ligatures by duplicating bbox)
            start_idx, end_idx = char_mappings[i]
            for _ in range(end_idx - start_idx):
                char_boxes.append(list(char_bbox))
        
        # Apply ink bleed effect if enabled
        if self.config.enable_ink_bleed and random.random() < 0.5:
            image = self.apply_ink_bleed(image)
        
        # Apply character connections for script fonts
        if self.config.enable_connections and self._is_script_font(font):
            image = self.apply_connections(image, char_boxes)
        
        return image, char_boxes
    
    def apply_ink_bleed(self, image: Image.Image) -> Image.Image:
        """
        Simulate ink bleeding effect.
        
        Args:
            image: Input image
            
        Returns:
            Image with ink bleed effect
        """
        if not self.config.enable_ink_bleed:
            return image
        
        # Convert to grayscale for processing
        gray = image.convert('L')
        
        # Apply slight blur to simulate ink spread
        blur_radius = random.uniform(0.3, 0.8) * self.config.ink_bleed_intensity
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Apply threshold to maintain text darkness
        np_img = np.array(blurred)
        threshold = 200  # Adjust for more/less aggressive effect
        np_img = np.where(np_img < threshold, np_img * 0.9, np_img)
        
        # Add slight morphological dilation for ink spread
        if self.config.ink_bleed_intensity > 0.3:
            from scipy import ndimage
            struct = np.ones((2, 2))
            np_img = ndimage.grey_dilation(np_img, footprint=struct)
        
        result = Image.fromarray(np_img.astype(np.uint8))
        return result.convert('RGB')
    
    def apply_connections(self, 
                        image: Image.Image, 
                        char_boxes: List[List[float]]) -> Image.Image:
        """
        Add connecting strokes between characters (for cursive/script styles).
        
        Args:
            image: Input image
            char_boxes: Character bounding boxes
            
        Returns:
            Image with character connections
        """
        if not self.config.enable_connections or len(char_boxes) < 2:
            return image
        
        draw = ImageDraw.Draw(image)
        
        for i in range(len(char_boxes) - 1):
            if random.random() < self.config.connection_probability:
                # Get connection points
                box1 = char_boxes[i]
                box2 = char_boxes[i + 1]
                
                # Connect from right-middle of first char to left-middle of second
                start_x = box1[2]  # Right edge
                start_y = (box1[1] + box1[3]) / 2  # Middle height
                end_x = box2[0]  # Left edge
                end_y = (box2[1] + box2[3]) / 2  # Middle height
                
                # Draw a slight curve connection
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2 - random.uniform(0, 3)
                
                # Simple bezier approximation with lines
                points = [
                    (start_x, start_y),
                    (mid_x, mid_y),
                    (end_x, end_y)
                ]
                draw.line(points, fill='black', width=1)
        
        return image
    
    def _render_standard(self, 
                        text: str, 
                        font: ImageFont.FreeTypeFont,
                        direction: str) -> Tuple[Image.Image, List[List[float]]]:
        """
        Fallback standard rendering (placeholder for other directions).
        """
        # This would call the original rendering methods
        # You'd integrate this with your existing OCRDataGenerator methods
        pass
    
    def _is_script_font(self, font: ImageFont.FreeTypeFont) -> bool:
        """
        Heuristic to determine if a font is script/cursive style.
        """
        # You could check font name or test character connectivity
        # This is a simplified version
        return random.random() < 0.2  # 20% chance for demo


# Integration function for your existing OCRDataGenerator
def enhance_generator_with_overlap(generator_instance):
    """
    Enhance an existing OCRDataGenerator instance with overlap capabilities.
    
    Usage:
        generator = OCRDataGenerator(font_files, background_images)
        generator = enhance_generator_with_overlap(generator)
    """
    overlap_renderer = GlyphOverlapRenderer()
    
    # Store original render method
    original_render = generator_instance.render_left_to_right
    
    # Create enhanced render method
    def enhanced_render_left_to_right(text, font):
        # Use overlap renderer with probability
        if random.random() < 0.7:  # 70% chance of using overlap features
            return overlap_renderer.render_with_overlap(text, font, 'left_to_right')
        else:
            return original_render(text, font)
    
    # Replace method
    generator_instance.render_left_to_right = enhanced_render_left_to_right
    
    return generator_instance