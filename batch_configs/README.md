# Batch Configuration Examples

This directory contains example YAML batch configuration files for OCR synthetic data generation.

## Quick Start

Generate images using a batch configuration:

```bash
python3 src/main.py \
  --batch-config batch_configs/simple_example.yaml \
  --output-dir output \
  --fonts-dir data/fonts \
  --text-file data/raw_text/corpus.txt
```

## Configuration Format

### Basic Structure

```yaml
total_images: 1000        # Total number of images to generate
seed: 42                  # Optional: random seed for reproducibility

batches:
  - name: batch_name      # Identifier for this batch
    proportion: 0.6       # Proportion of total images (0.0-1.0)
    # ... batch-specific parameters
```

### Batch Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | required | Unique identifier for the batch |
| `proportion` | float | required | Fraction of total images (will be normalized if sum ≠ 1.0) |
| `text_direction` | string | "left_to_right" | Direction: "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top", or "random" |
| `corpus_file` | string | CLI default | Path to text corpus file |
| `font_filter` | string | "*.{ttf,otf}" | Glob pattern to match fonts |
| `font_weights` | dict | {} | Font weight multipliers (pattern: weight) |
| `min_text_length` | int | 5 | Minimum text length in characters |
| `max_text_length` | int | 25 | Maximum text length in characters |
| `curve_type` | string | "none" | Text curvature: "none", "arc", "sine", or "random" |
| `curve_intensity` | float | 0.0 | Curvature strength (0.0-1.0) |
| `overlap_intensity` | float | 0.0 | Glyph overlap amount (0.0-1.0) |
| `ink_bleed_intensity` | float | 0.0 | Ink bleed effect strength (0.0-1.0) |
| `effect_type` | string | "none" | 3D effect type: "none", "raised", "embossed", or "engraved" |
| `effect_depth` | float | 0.5 | 3D effect depth intensity (0.0-1.0) |
| `light_azimuth` | float | 135.0 | Light direction angle (0-360 degrees) |
| `light_elevation` | float | 45.0 | Light elevation angle (0-90 degrees) |
| `text_color_mode` | string | "uniform" | Color mode: "uniform", "per_glyph", "gradient", or "random" |
| `color_palette` | string | "realistic_dark" | Color palette: "realistic_dark", "realistic_light", "vibrant", or "pastels" |
| `custom_colors` | list | [] | List of custom RGB tuples (e.g., [[255, 0, 0], [0, 255, 0]]) |
| `background_color` | string/list | "auto" | Background color: RGB list, "auto" (contrasting), or color string |

### Font Filters

Use glob patterns to select fonts:

```yaml
font_filter: "*.ttf"                           # All TTF fonts
font_filter: "*{arabic,amiri}*.ttf"           # Arabic fonts
font_filter: "*{cjk,noto*jp}*.{ttf,otf}"      # CJK fonts (TTF or OTF)
font_filter: "*{sans,mono}*regular*.ttf"       # Sans/mono regular weights
```

### Font Weights

Assign weights to prioritize certain fonts:

```yaml
font_weights:
  "*noto*": 2.0          # 2x weight for Noto fonts
  "*bold*": 0.5          # 0.5x weight (de-emphasize bold)
  "*amiri*": 3.0         # 3x weight for Amiri fonts
```

Higher weights = more likely to be selected.

### Text Curvature

Add realistic curves to text for more varied training data:

```yaml
# No curvature (straight line)
curve_type: none
curve_intensity: 0.0

# Circular arc (like text on a curved surface)
curve_type: arc
curve_intensity: 0.3  # 0.0-1.0, higher = more curve

# Sine wave (wavy text)
curve_type: sine
curve_intensity: 0.4  # 0.0-1.0, higher = more wavy

# Random selection per image
curve_type: random  # Chooses none/arc/sine randomly
curve_intensity: 0.3
```

**Curvature Types:**
- `none` - Straight baseline (default)
- `arc` - Circular arc path (realistic for curved surfaces, signs)
- `sine` - Sine wave pattern (wavy/distorted text)
- `random` - Randomly chooses arc or sine for each image

**Intensity Guidelines:**
- `0.0-0.2` - Subtle curve (realistic documents)
- `0.2-0.4` - Moderate curve (street signs, labels)
- `0.4-0.6` - Strong curve (artistic/distorted)
- `0.6-1.0` - Extreme curve (special effects)

**Supported Directions:** Curvature works for all text directions:
- `left_to_right` - Arc and sine curves
- `right_to_left` - Arc and sine curves (mirrored)
- `top_to_bottom` - Arc and sine curves (vertical)
- `bottom_to_top` - Arc and sine curves (vertical, reversed)

### Glyph Overlap

Simulate realistic handwriting and ancient text with character overlap:

```yaml
# No overlap (default)
overlap_intensity: 0.0

# Subtle overlap (typical handwriting)
overlap_intensity: 0.25  # 20% overlap

# Moderate overlap (dense handwriting, ancient scripts)
overlap_intensity: 0.5   # 40% overlap

# Heavy overlap (cursive, calligraphy)
overlap_intensity: 0.75  # 60% overlap
```

**Overlap Guidelines:**
- `0.0` - No overlap (default, typical printed text)
- `0.0-0.3` - Subtle overlap (realistic handwriting)
- `0.3-0.6` - Moderate overlap (dense text, ancient manuscripts)
- `0.6-1.0` - Heavy overlap (cursive, artistic calligraphy)

**Language-Agnostic:** The overlap feature uses spacing reduction rather than language-specific kerning tables, making it work with any script (Latin, Arabic, CJK, etc.).

### Ink Bleed Effect

Add realistic document scanning artifacts:

```yaml
# No ink bleed (default)
ink_bleed_intensity: 0.0

# Subtle bleed (good quality scan)
ink_bleed_intensity: 0.3

# Moderate bleed (typical old documents)
ink_bleed_intensity: 0.5

# Heavy bleed (aged/degraded documents)
ink_bleed_intensity: 0.8
```

**Combines well with overlap** for ancient manuscript simulation:
```yaml
overlap_intensity: 0.6      # Heavy character overlap
ink_bleed_intensity: 0.5    # Moderate ink bleeding
```

### 3D Text Effects

Add realistic depth and dimensionality to text:

```yaml
# No 3D effect (default)
effect_type: none
effect_depth: 0.5

# Raised text (drop shadow effect)
effect_type: raised
effect_depth: 0.6           # 0.0-1.0, higher = more pronounced
light_azimuth: 135.0        # Light angle (0-360°)
light_elevation: 45.0       # Light elevation (0-90°)

# Embossed text (raised with highlights and shadows)
effect_type: embossed
effect_depth: 0.5
light_azimuth: 135.0
light_elevation: 45.0

# Engraved/debossed text (carved into surface)
effect_type: engraved
effect_depth: 0.7
light_azimuth: 135.0
light_elevation: 45.0
```

**Effect Types:**
- `none` - No 3D effect (default, flat text)
- `raised` - Drop shadow effect for text that appears above the surface
- `embossed` - Raised text with edge highlights and shadows for realistic depth
- `engraved` - Text carved into the surface (inverted emboss)

**Depth Guidelines:**
- `0.0` - No effect
- `0.0-0.3` - Subtle depth (realistic documents)
- `0.3-0.6` - Moderate depth (signs, labels)
- `0.6-1.0` - Strong depth (artistic/decorative)

**Light Direction (Azimuth):**
- `0°` - Light from top (shadow below)
- `90°` - Light from right (shadow left)
- `135°` - Light from top-right (shadow bottom-left) - default
- `180°` - Light from bottom (shadow above)
- `270°` - Light from left (shadow right)

**Light Elevation:**
- `10-30°` - Low angle, long shadows (dramatic effect)
- `45°` - Moderate angle (default, balanced)
- `60-90°` - High angle, short shadows (subtle effect)

**Language-Agnostic:** Works with all scripts (Latin, Arabic, CJK, etc.)

**Combines well with other effects:**
```yaml
# Embossed ancient manuscript
effect_type: embossed
effect_depth: 0.5
overlap_intensity: 0.4      # Character overlap
ink_bleed_intensity: 0.3    # Ink bleed

# Engraved curved signage
effect_type: engraved
effect_depth: 0.6
curve_type: arc
curve_intensity: 0.3
```

### Text Color

Add realistic and artistic colors to text for varied training data:

```yaml
# Uniform color (all characters same color per line)
text_color_mode: uniform
color_palette: realistic_dark  # All text rendered in dark colors

# Per-glyph color (each character different color)
text_color_mode: per_glyph
color_palette: vibrant  # Each char gets a different vibrant color

# Gradient mode (gradient across text)
text_color_mode: gradient
color_palette: pastels  # Smooth gradient between first two palette colors

# Random mode (randomly choose uniform or per_glyph)
text_color_mode: random
color_palette: realistic_light

# Custom colors (define your own palette)
text_color_mode: uniform
custom_colors:
  - [255, 0, 0]      # Red
  - [0, 255, 0]      # Green
  - [0, 0, 255]      # Blue

# Background color (auto-contrast or custom)
background_color: auto               # Automatically contrasting (default)
background_color: [255, 255, 255]    # White background
background_color: [0, 0, 0]          # Black background
```

**Color Modes:**
- `uniform` - All characters same color per line (default)
- `per_glyph` - Each character gets a different color from palette
- `gradient` - Smooth gradient from first to second palette color
- `random` - Randomly chooses uniform or per_glyph per image

**Color Palettes:**
- `realistic_dark` - Black, navy, brown, dark grays (for typical documents)
- `realistic_light` - White, beige, light blues (for inverted/bright text)
- `vibrant` - Red, green, blue, yellow, magenta, cyan (high saturation)
- `pastels` - Light pink, light blue, peach, pale green (soft colors)

**Background Color:**
- `auto` - Automatically calculates contrasting background (default)
- RGB list like `[255, 255, 255]` - Custom background color
- System automatically ensures good contrast with text color

**Language-Agnostic:** Works with all scripts (Latin, Arabic, CJK, etc.)

**Combines well with other effects:**
```yaml
# Vibrant curved text
text_color_mode: per_glyph
color_palette: vibrant
curve_type: arc
curve_intensity: 0.3

# Dark realistic text with 3D emboss
text_color_mode: uniform
color_palette: realistic_dark
effect_type: embossed
effect_depth: 0.5

# Ancient manuscript with custom colors
text_color_mode: uniform
custom_colors: [[139, 69, 19]]  # Saddle brown
background_color: [245, 245, 220]  # Beige
overlap_intensity: 0.4
ink_bleed_intensity: 0.3
```

## Example Configurations

### 1. Simple Example (`simple_example.yaml`)
- 100 images total
- 60% English (LTR)
- 40% Arabic (RTL)
- Basic font filtering

### 2. Multilingual (`multilingual.yaml`)
- 1000 images total
- 35% English horizontal
- 25% Arabic RTL (with font weights)
- 20% Japanese vertical (top-to-bottom)
- 10% Japanese vertical (bottom-to-top)
- 10% Random mixed

### 3. Advanced (`advanced.yaml`)
- 500 images total
- Varied text lengths per batch
- Font weight customization
- Specialty font handling
- Direction randomization

### 4. Ancient Overlap (`ancient_overlap.yaml`)
- 100 images demonstrating overlap effects
- 25% modern printed (no overlap)
- 25% handwritten style (subtle overlap)
- 25% ancient manuscript (moderate overlap + ink bleed)
- 25% calligraphic (heavy overlap + bleed)

### 5. 3D Effects (`3d_effects_example.yaml`)
- 100 images demonstrating 3D text effects
- 25% flat text (no 3D effect)
- 25% raised text (drop shadow)
- 25% embossed text (raised with highlights)
- 25% engraved text (carved into surface)

### 6. Advanced 3D Combined (`3d_combined_example.yaml`)
- 100 images combining 3D effects with other features
- 30% embossed curved signage
- 25% engraved ancient manuscript (with overlap + ink bleed)
- 25% raised handwriting (with overlap + wavy curves)
- 20% embossed vertical text (top-to-bottom)

### 7. Color Examples (`color_samples/`)
- 10 color configuration examples in `batch_configs/color_samples/`
- **vibrant_per_glyph.yaml** - Each character different vibrant color
- **realistic_dark_random.yaml** - Randomly uniform or per-glyph dark colors
- **realistic_light_per_glyph.yaml** - Light colors with per-glyph variation
- **pastels_gradient.yaml** - Smooth gradient with pastel colors
- **custom_gradient.yaml** - Custom color gradient
- **custom_red_on_yellow.yaml** - Custom red text on yellow background
- **black_on_white.yaml** - Classic black text on white
- **white_on_black.yaml** - Inverted white text on black
- **auto_background.yaml** - Auto-contrasting backgrounds
- **mixed_color_modes.yaml** - Mix of all color modes

## Features

### Interleaved Generation
Images are generated in an interleaved fashion across all batches, ensuring balanced output even if generation is interrupted.

### Best-Effort Proportions
The system attempts to match specified proportions approximately. If a font cannot render a corpus (insufficient glyph coverage), it will skip that combination and continue.

### Font Validation
Each font is automatically validated against the corpus to ensure it can render the required characters (90% coverage threshold).

### Reproducibility
Set `seed` for reproducible image generation across runs.

## Tips

1. **Corpus Files**: Ensure each batch's `corpus_file` contains text in the appropriate script/language
2. **Font Filters**: Use specific patterns to target the right fonts (e.g., Arabic fonts for Arabic text)
3. **Proportions**: They will be normalized if they don't sum to 1.0
4. **Text Lengths**: Adjust per batch based on typical text patterns (Arabic often shorter, CJK can be longer)
5. **Direction**: Use "random" for varied training data within a batch

## Troubleshooting

**No fonts match filter:**
- Check font filter pattern syntax
- Verify fonts exist in fonts directory
- System falls back to all fonts if none match

**Low coverage warnings:**
- Font cannot render corpus characters
- Ensure corpus matches font capabilities (e.g., Arabic text needs Arabic fonts)

**Proportion mismatches:**
- Best-effort allocation - exact counts may vary slightly
- Check logs for skipped combinations
