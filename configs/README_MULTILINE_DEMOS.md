# Multi-Line Text Generation Demo Configurations

This directory contains demonstration configurations that showcase the new multi-line text generation features across all text directions.

## Overview

These demo configurations are designed to:
1. **Test backward compatibility** - Ensure single-line mode still works (old code path)
2. **Demonstrate new features** - Show all new multi-line capabilities
3. **Verify correctness** - Enable visual inspection of output images
4. **Test all directions** - Cover LTR, RTL, TTB, and BTT text

## Demo Configurations

### Left-to-Right (LTR)

#### `demo_multiline_ltr_stage1.yaml` - Basic Features (150 images)
- **Spec 1 (33%)**: Single-line mode - backward compatibility test
  - Uses OLD code path (min_lines=1, max_lines=1)
  - No multi-line features
  - Verifies existing functionality still works

- **Spec 2 (33%)**: Multi-line with word breaking
  - Tests NEW multi-line code path
  - Line break mode: "word"
  - 2-4 lines per image
  - Left alignment

- **Spec 3 (34%)**: Multi-line with character breaking
  - Tests NEW multi-line code path
  - Line break mode: "character"
  - 3-5 lines per image
  - Center alignment

#### `demo_multiline_ltr_stage2.yaml` - Intermediate Features (150 images)
- **Spec 1 (33%)**: Left alignment with effects
  - Effects applied uniformly across all lines
  - Tests blur and noise on multi-line text

- **Spec 2 (33%)**: Center alignment with variable spacing
  - Line spacing: 1.2-2.0x
  - Tests alignment and spacing variations

- **Spec 3 (34%)**: Right alignment with perspective warp
  - Character breaking mode
  - Tests geometric augmentations on multi-line

#### `demo_multiline_ltr_stage3.yaml` - Advanced Features (150 images)
- **Spec 1 (33%)**: Multi-line with arc curves
  - Each line curved independently
  - Same curve parameters across all lines

- **Spec 2 (33%)**: Multi-line with full augmentation pipeline
  - Multiple effects: rotation, noise, blur, perspective warp
  - All applied uniformly

- **Spec 3 (34%)**: Variable text length
  - Range: 1 character to 200 characters
  - Lines: 1 to 8
  - Tests extreme cases

### Right-to-Left (RTL)

#### `demo_multiline_rtl_stage1.yaml` - Basic RTL Features (150 images)
- **Spec 1 (33%)**: Single-line RTL (backward compatibility)
- **Spec 2 (33%)**: Multi-line RTL with word breaking
  - Right alignment (natural for RTL)
- **Spec 3 (34%)**: Multi-line RTL with character breaking
  - Center alignment

#### `demo_multiline_rtl_stage2.yaml` - Intermediate RTL Features (150 images)
- **Spec 1 (33%)**: Right alignment with effects
  - Effects applied uniformly across all lines
- **Spec 2 (33%)**: Center alignment with variable spacing
  - Line spacing: 1.5-2.5x
- **Spec 3 (34%)**: Left alignment with perspective warp
  - Character breaking mode

#### `demo_multiline_rtl_stage3.yaml` - Advanced RTL Features (150 images)
- **Spec 1 (33%)**: Multi-line RTL with arc curves
  - Each line curved independently
- **Spec 2 (33%)**: Multi-line RTL with full augmentation pipeline
  - Multiple effects applied uniformly
- **Spec 3 (34%)**: Variable text length
  - Range: 1-200 characters, 1-8 lines

### Top-to-Bottom (TTB)

#### `demo_multiline_ttb_stage1.yaml` - Basic TTB Features (150 images)
- **Spec 1 (33%)**: Single-line vertical (backward compatibility)
- **Spec 2 (33%)**: Multi-line vertical with word breaking
  - Top alignment (for vertical text)
  - Lines arranged horizontally (side by side)
- **Spec 3 (34%)**: Multi-line vertical with character breaking
  - Center alignment

#### `demo_multiline_ttb_stage2.yaml` - Intermediate TTB Features (150 images)
- **Spec 1 (33%)**: Top alignment with effects
  - Noise and blur applied uniformly
- **Spec 2 (33%)**: Center alignment with variable spacing
  - Line spacing: 1.5-2.5x
- **Spec 3 (34%)**: Bottom alignment with perspective warp
  - Character breaking mode

#### `demo_multiline_ttb_stage3.yaml` - Advanced TTB Features (150 images)
- **Spec 1 (33%)**: Multi-line vertical with arc curves
  - Concave arc curves for TTB
- **Spec 2 (33%)**: Multi-line vertical with full augmentation
  - Multiple effects applied uniformly
- **Spec 3 (34%)**: Variable text length
  - Range: 1-200 characters, 1-6 lines

### Bottom-to-Top (BTT)

#### `demo_multiline_btt_stage1.yaml` - Basic BTT Features (150 images)
- **Spec 1 (33%)**: Single-line BTT (backward compatibility)
- **Spec 2 (33%)**: Multi-line BTT with character breaking
  - Bottom alignment
- **Spec 3 (34%)**: Multi-line BTT with word breaking
  - Top alignment

#### `demo_multiline_btt_stage2.yaml` - Intermediate BTT Features (150 images)
- **Spec 1 (33%)**: Bottom alignment with effects
  - Noise and blur applied uniformly
- **Spec 2 (33%)**: Center alignment with variable spacing
  - Line spacing: 1.5-2.5x
- **Spec 3 (34%)**: Top alignment with perspective warp
  - Character breaking mode

#### `demo_multiline_btt_stage3.yaml` - Advanced BTT Features (150 images)
- **Spec 1 (33%)**: Multi-line BTT with arc curves
  - Convex arc curves for BTT
- **Spec 2 (33%)**: Multi-line BTT with full augmentation
  - Multiple effects applied uniformly
- **Spec 3 (34%)**: Variable text length
  - Range: 1-200 characters, 1-6 lines

## Running the Demos

### Run All Demos (Recommended)
```bash
./scripts/demo_multiline_all.sh
```
This generates **1800 images** covering all text directions and features (12 stages total).

### Run Individual Direction Demos
```bash
./scripts/demo_multiline_ltr.sh   # LTR - 450 images (3 stages)
./scripts/demo_multiline_rtl.sh   # RTL - 450 images (3 stages)
./scripts/demo_multiline_ttb.sh   # TTB - 450 images (3 stages)
./scripts/demo_multiline_btt.sh   # BTT - 450 images (3 stages)
```

## Verification Checklist

After running the demos, verify the following by inspecting generated images and JSON files:

### Visual Verification (Images)

**Single-Line Mode (Backward Compatibility)**:
- [ ] Single-line images render correctly
- [ ] No visual differences from pre-multi-line code
- [ ] All text directions work (LTR, RTL, TTB, BTT)

**Multi-Line Word Breaking**:
- [ ] Words are kept intact (not split across lines)
- [ ] Text is distributed reasonably across lines
- [ ] Works for all text directions

**Multi-Line Character Breaking**:
- [ ] Text breaks at any character position
- [ ] Characters distributed evenly across lines
- [ ] Works for all text directions

**Text Alignment**:
- [ ] Left alignment: lines aligned to left edge
- [ ] Center alignment: lines centered
- [ ] Right alignment: lines aligned to right edge
- [ ] Top alignment (vertical): lines aligned to top
- [ ] Bottom alignment (vertical): lines aligned to bottom

**Line Spacing**:
- [ ] Lines are properly spaced
- [ ] Spacing is consistent within each image
- [ ] Variable spacing works (different images have different spacing)

**Effects Application**:
- [ ] Blur is uniform across all lines
- [ ] Noise is uniform across all lines
- [ ] Rotation affects entire multi-line block
- [ ] Perspective warp affects entire block

**Curved Text**:
- [ ] Each line is curved individually
- [ ] All lines use same curve parameters
- [ ] Curves look natural

**Vertical Text**:
- [ ] TTB lines are arranged horizontally (left to right)
- [ ] BTT lines are arranged horizontally (left to right)
- [ ] Vertical alignment works correctly

### Truth Data Verification (JSON Files)

**Single-Line Mode**:
- [ ] JSON files do NOT contain `line_index` field in bboxes
- [ ] All existing fields are present and correct

**Multi-Line Mode**:
- [ ] JSON files contain `num_lines` field
- [ ] JSON files contain `lines` array
- [ ] JSON files contain `line_spacing` field
- [ ] JSON files contain `line_break_mode` field
- [ ] JSON files contain `text_alignment` field
- [ ] Each bbox has `line_index` field (0-indexed)
- [ ] Line indices are sequential (0, 1, 2, ...)
- [ ] Character order is preserved in bboxes

**Example Multi-Line JSON**:
```json
{
  "text": "Full text content",
  "lines": ["Line 1", "Line 2", "Line 3"],
  "num_lines": 3,
  "line_spacing": 1.2,
  "line_break_mode": "word",
  "text_alignment": "left",
  "bboxes": [
    {
      "char": "L",
      "x0": 10, "y0": 5, "x1": 20, "y1": 35,
      "line_index": 0
    },
    ...
  ]
}
```

## Feature Coverage Matrix

| Feature | LTR | RTL | TTB | BTT | Config |
|---------|-----|-----|-----|-----|--------|
| Single-line (compat) | ✓ | ✓ | ✓ | ✓ | All stage1 |
| Word breaking | ✓ | ✓ | ✓ | ✓ | All stage1 |
| Character breaking | ✓ | ✓ | ✓ | ✓ | All stage1 |
| Left alignment | ✓ | ✓ | - | - | ltr_stage1-3, rtl_stage2 |
| Center alignment | ✓ | ✓ | ✓ | ✓ | All stages |
| Right alignment | ✓ | ✓ | - | - | ltr_stage2, rtl_stage1-2 |
| Top alignment | - | - | ✓ | ✓ | ttb/btt_stage1-2 |
| Bottom alignment | - | - | ✓ | ✓ | ttb_stage2, btt_stage1-2 |
| Variable line spacing | ✓ | ✓ | ✓ | ✓ | All stage2-3 |
| Curved text (arc) | ✓ | ✓ | ✓ | ✓ | All stage3 |
| Effects (uniform) | ✓ | ✓ | ✓ | ✓ | All stage2-3 |
| Full augmentation | ✓ | ✓ | ✓ | ✓ | All stage3 |
| Variable length (1-200) | ✓ | ✓ | ✓ | ✓ | All stage3 |

## Troubleshooting

### Generation Errors

**"Invalid line_break_mode"**:
- Ensure only "word" or "character" is used
- Check YAML syntax

**"min_lines must be >= 1"**:
- Line count must be at least 1
- Check min_lines and max_lines values

**"line_spacing_min must be > 0"**:
- Line spacing cannot be zero or negative
- Use values >= 0.1

### Visual Issues

**Lines appear too close together**:
- Increase `line_spacing_min` and `line_spacing_max`

**Words split across lines**:
- Verify `line_break_mode: "word"` is set
- Check that corpus has whitespace-separated words

**Lines not aligned correctly**:
- Verify `text_alignment` value is appropriate for text direction
- For vertical text, use "top", "center", or "bottom"
- For horizontal text, use "left", "center", or "right"

## Output Structure

After running demos, output will be organized as:

```
output.nosync/demo_multiline/
├── ltr_stage1/
│   ├── images/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   └── labels/
│       ├── 000000.json
│       ├── 000001.json
│       └── ...
├── ltr_stage2/
├── ltr_stage3/
├── rtl_stage1/
├── rtl_stage2/
├── rtl_stage3/
├── ttb_stage1/
├── ttb_stage2/
├── ttb_stage3/
├── btt_stage1/
├── btt_stage2/
└── btt_stage3/
```

## Next Steps

1. Run the demos: `./scripts/demo_multiline_all.sh`
2. Visually inspect generated images
3. Verify JSON truth data
4. Check that single-line and multi-line modes both work
5. Confirm effects are applied uniformly
6. Test with your own text corpora and fonts

## Notes

- Demo configs use small batch sizes (150 images) for quick testing
- Clean images (minimal effects) in stage 1 for easy verification
- Incremental complexity across stages
- All configs use white backgrounds for clarity
- Font sizes are moderate (24-52px) for visibility
