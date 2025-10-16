# Conceptual: Curved Text Rendering

## Overview

The curved text feature enables text to be rendered along mathematical curves (arcs, sine waves, etc.) to simulate real-world conditions such as:
- Text on curved surfaces (bottles, signs, cylindrical objects)
- Artistic or handwritten text with natural curves
- Sloppily written text with uneven baselines
- Projected text on non-flat surfaces

Unlike augmentations that distort already-rendered straight text, curved text is rendered during the **text rendering stage**, ensuring more accurate character placement and bounding boxes.

## Curve Types

### `curve_type: "none"` (Default)
Standard straight-line text rendering. All curve parameters are present but set to zero values.

### `curve_type: "arc"`
Text follows a circular arc. Characters are positioned along the circumference of a circle and rotated to be tangent to the curve.

**Parameters:**
- `arc_radius`: Distance from the center of the circle to the text baseline
  - `0.0` = straight line (infinite radius)
  - Larger values = gentler curves
  - Typical range: 100-500 pixels
- `arc_concave`: Controls curve direction
  - `true` = concave (curves "inward")
  - `false` = convex (curves "outward")

### `curve_type: "sine"`
Text follows a sine wave pattern. The baseline oscillates vertically (for horizontal text) or horizontally (for vertical text).

**Parameters:**
- `sine_amplitude`: Height of the wave oscillation
  - `0.0` = straight line
  - Typical range: 5-30 pixels
- `sine_frequency`: How many wave cycles per unit length
  - `0.0` = straight line
  - Higher values = more oscillations
  - Typical range: 0.01-0.1
- `sine_phase`: Starting phase offset of the wave (in radians)
  - Allows wave to start at different points in its cycle
  - Range: 0.0 to 2π (6.28)

## Curve Direction/Orientation

### Arc Curves Explained

For **Left-to-Right (LTR) text:**

**Concave (arc_concave=true)** - Text curves **upward** like a smile ☺
```
   t  e  x  t
  (     )
 (       )
```

**Convex (arc_concave=false)** - Text curves **downward** like a frown ☹
```
(       )
 (     )
   t  e  x  t
```

For **Top-to-Bottom (TTB) text:**

**Concave** - Text curves **rightward** (bends away from left margin)
```
t (
e  (
x   (
t    (
```

**Convex** - Text curves **leftward** (bends toward left margin)
```
    ) t
   ) e
  ) x
 ) t
```

For **Right-to-Left (RTL)** and **Bottom-to-Top (BTT)**, the orientation is reversed accordingly.

### Arc Radius Visualization

```
Small radius (100):     Large radius (500):      Infinite radius (0):
    ___                      ___                    ___________
   /   \                    /   \
  | txt |                __|  txt |__              | t e x t |
   \___/
  (tight curve)          (gentle curve)            (straight line)
```

### Sine Wave Parameters

- **Amplitude**: How tall the waves are
- **Frequency**: How many waves appear
- **Phase**: Where the wave starts

```
Low amplitude (5px):     High amplitude (20px):
_/‾\_/‾\_/‾\_           _____/‾‾‾‾‾\_____
 text text               text    text

Low frequency (0.01):    High frequency (0.05):
___/‾‾‾‾‾\_____         _/‾\_/‾\_/‾\_/‾\_
 long  curve            many short waves
```

## Design Principles

### 1. Universal Direction Support
Curves work identically for all 4 text directions:
- `left_to_right`
- `right_to_left`
- `top_to_bottom`
- `bottom_to_top`

Vertical text is NOT treated as a special case - the curve algorithms are direction-agnostic.

### 2. ML Feature Consistency
**All curve parameters are always present** in the plan/JSON, even when `curve_type="none"`. This ensures:
- Consistent feature vectors for machine learning analysis
- Random Forest classifiers can correlate OCR performance with curve parameters
- Zero values (`arc_radius=0.0`, `sine_amplitude=0.0`) explicitly signal "no curve"

### 3. Separation from Augmentations
Curved text rendering happens **before** augmentations like rotation, perspective warp, etc. This allows:
- Curves to be applied, then augmentations to further distort
- Accurate bbox calculation at the rendering stage
- Realistic simulation of curved text that is then photographed at an angle

### 4. Extensibility
The architecture supports future curve types:
- **Bezier curves** (planned)
- **Random perturbations** (planned) - each character randomly offset
- **Circular** (planned) - full circle text

## Implementation Status

### Completed (Phase 0-2)
- ✅ Configuration system with all curve parameters
- ✅ Parameters integrated into `BatchSpecification`
- ✅ `plan_generation()` includes curve data in plans
- ✅ Arc text rendering with full direction support (LTR, RTL, TTB, BTT)
- ✅ Bounding box calculation for arc text using transform-based approach
- ✅ Sine wave text rendering with full direction support
- ✅ Automatic image cropping to content bounds
- ✅ Zero-value fallback to straight text (arc_radius=0, sine_amplitude=0)

### Planned (Phase 3+)
- 📋 Generic curve dispatcher for extensibility
- 📋 Bezier curves
- 📋 Random perturbation curves
- 📋 Circular (full circle) text

## Usage Example

```yaml
# batch_config.yaml
specifications:
  - name: "curved_label_text"
    proportion: 0.5
    text_direction: "left_to_right"
    corpus_file: "labels.txt"

    # Arc curve parameters
    curve_type: "arc"
    arc_radius_min: 150.0
    arc_radius_max: 300.0
    arc_concave: true

    # Sine parameters still present for ML (but unused when curve_type="arc")
    sine_amplitude_min: 0.0
    sine_amplitude_max: 0.0
    sine_frequency_min: 0.0
    sine_frequency_max: 0.0
```

The resulting JSON plan will contain specific values:
```json
{
  "text": "Sample Label",
  "curve_type": "arc",
  "arc_radius": 225.7,
  "arc_concave": true,
  "sine_amplitude": 0.0,
  "sine_frequency": 0.0,
  ...
}
```

## Mathematical Foundation

### Arc Rendering
For each character at position `i` along the text:
1. Calculate arc length position: `s = char_position / total_chars * arc_span`
2. Calculate angle: `θ = s / radius`
3. Calculate position: `x = radius * cos(θ)`, `y = radius * sin(θ)`
4. Calculate tangent angle for rotation: `rotation = θ + 90°`

### Sine Wave Rendering
For each character at horizontal position `x`:
1. Calculate vertical offset: `y_offset = amplitude * sin(frequency * x + phase)`
2. Calculate tangent angle: `rotation = atan(amplitude * frequency * cos(frequency * x + phase))`

## Bounding Box Strategy

Curved text uses **transform-based bounding box calculation**:
1. Each character is rendered onto a temporary surface
2. The character is rotated to match the curve tangent
3. The four corners of the character's original bbox are transformed through the rotation
4. The axis-aligned bounding rectangle of these transformed corners becomes the final bbox

This approach:
- Is fast and mathematically precise
- Matches the existing `apply_rotation()` augmentation strategy
- Provides consistent results across the pipeline
- May be 3-5% loose (slight padding) which can actually help OCR training
