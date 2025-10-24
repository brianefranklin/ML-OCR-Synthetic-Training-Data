# Multi-Line Text Generation - Demo Quick Start

This guide helps you quickly test the new multi-line text generation features.

## Quick Start (5 minutes)

### 1. Run a Small Test

Test just the LTR Stage 1 (150 images, ~30 seconds):

```bash
python3 -m src.main \
  --batch-config configs/demo_multiline_ltr_stage1.yaml \
  --output-dir output.nosync/demo_test \
  --font-dir data.nosync/fonts \
  --background-dir data.nosync/backgrounds/solid_white \
  --corpus-dir data.nosync/corpus_text \
  --generation-workers 2 \
  --workers 2 \
  --chunk-size 50 \
  --io-batch-size 25 \
  --log-level INFO
```

### 2. Inspect the Output

```bash
# View a few generated images
ls output.nosync/demo_test/images/ | head -10

# Check a JSON label file
cat output.nosync/demo_test/labels/000000.json | python3 -m json.tool
```

### 3. Verify Key Features

**Check for single-line mode (backward compatibility)**:
```bash
# Single-line images should NOT have "line_index" in bboxes
# Look for images from spec "ltr_single_line_compat"
grep -l "ltr_single_line_compat" output.nosync/demo_test/labels/*.json | head -1 | xargs cat | python3 -m json.tool | grep -A5 "bboxes"
```

**Check for multi-line mode**:
```bash
# Multi-line images SHOULD have "line_index" in bboxes
# Look for images from spec "ltr_multiline_word"
grep -l "ltr_multiline_word" output.nosync/demo_test/labels/*.json | head -1 | xargs cat | python3 -m json.tool | grep -A10 "bboxes"
```

## Full Demo Suite (30 minutes)

Run all demos across all text directions:

```bash
./scripts/demo_multiline_all.sh
```

This generates **1800 images** (12 stages) testing:
- ✓ All text directions (LTR, RTL, TTB, BTT)
- ✓ Single-line mode (backward compatibility)
- ✓ Word and character line breaking
- ✓ All text alignments
- ✓ Line spacing variations
- ✓ Effects applied uniformly
- ✓ Curved multi-line text
- ✓ Variable text length (1 char to paragraph)

## What to Look For

### Visual Inspection

1. **Single-Line Images** (from `*_single_line_compat` specs):
   - Should look like normal single-line text
   - No changes from previous behavior

2. **Multi-Line Word Breaking** (from `*_multiline_word` specs):
   - Words should NOT be split across lines
   - Lines should have varying lengths based on word boundaries
   - Example: "Hello world testing" → ["Hello world", "testing"]

3. **Multi-Line Character Breaking** (from `*_multiline_char` specs):
   - Text can break anywhere
   - Lines should have similar lengths
   - Example: "HelloWorld" → ["Hello", "World"]

4. **Text Alignment**:
   - **Left**: All lines start at same left position
   - **Center**: Lines are centered (ragged left and right)
   - **Right**: All lines end at same right position

5. **Effects Applied Uniformly**:
   - If image is blurry, ALL lines are equally blurry
   - If image has noise, ALL lines have same noise
   - If image is rotated, entire multi-line block rotates together

6. **Curved Text**:
   - Each line should curve independently
   - All lines should use same curve parameters

### JSON Truth Data

**Single-Line Mode JSON** (backward compatible):
```json
{
  "text": "Hello World",
  "num_lines": 1,
  "lines": ["Hello World"],
  "bboxes": [
    {
      "char": "H",
      "x0": 10, "y0": 5, "x1": 25, "y1": 35
      // NOTE: No "line_index" field
    },
    ...
  ]
}
```

**Multi-Line Mode JSON** (new feature):
```json
{
  "text": "Hello World Testing",
  "num_lines": 2,
  "lines": ["Hello World", "Testing"],
  "line_spacing": 1.2,
  "line_break_mode": "word",
  "text_alignment": "left",
  "bboxes": [
    {
      "char": "H",
      "x0": 10, "y0": 5, "x1": 25, "y1": 35,
      "line_index": 0  // NEW FIELD
    },
    {
      "char": "T",
      "x0": 10, "y0": 45, "x1": 22, "y1": 75,
      "line_index": 1  // Second line
    },
    ...
  ]
}
```

## Validation Checklist

After running demos, verify:

- [ ] **Backward Compatibility**: Single-line images work correctly
  - No `line_index` in bboxes
  - Images look normal
  - No errors during generation

- [ ] **Multi-Line Word Breaking**: Words stay intact
  - Check images from `*_multiline_word` specs
  - Words should not be split mid-word

- [ ] **Multi-Line Character Breaking**: Even distribution
  - Check images from `*_multiline_char` specs
  - Characters distributed evenly

- [ ] **Text Alignments**: Correct positioning
  - Left-aligned: lines start at same x
  - Center-aligned: lines centered
  - Right-aligned: lines end at same x

- [ ] **Line Spacing**: Proper vertical/horizontal spacing
  - Lines should not overlap
  - Spacing should be consistent within image

- [ ] **Effects Uniformity**: Same effect across all lines
  - Check blur, noise, rotation in stage 2-3
  - All lines should have same effect intensity

- [ ] **Truth Data**: Correct JSON structure
  - Multi-line: has `line_index`, `num_lines`, `lines`
  - Single-line: no `line_index`

- [ ] **All Directions**: LTR, RTL, TTB, BTT all work
  - Each direction produces correct output
  - No crashes or errors

## Demo Configuration Summary

| Config | Images | Text Dir | Features Tested |
|--------|--------|----------|-----------------|
| ltr_stage1 | 150 | LTR | Basic: single-line (compat), word break, char break |
| ltr_stage2 | 150 | LTR | Intermediate: alignments, spacing, effects |
| ltr_stage3 | 150 | LTR | Advanced: curves, full augmentation, variable length |
| rtl_stage1 | 150 | RTL | Basic: single-line, word/char breaking |
| rtl_stage2 | 150 | RTL | Intermediate: alignments, spacing, effects |
| rtl_stage3 | 150 | RTL | Advanced: curves, full augmentation, variable length |
| ttb_stage1 | 150 | TTB | Basic: single-line, word/char breaking |
| ttb_stage2 | 150 | TTB | Intermediate: alignments, spacing, effects |
| ttb_stage3 | 150 | TTB | Advanced: curves, full augmentation, variable length |
| btt_stage1 | 150 | BTT | Basic: single-line, word/char breaking |
| btt_stage2 | 150 | BTT | Intermediate: alignments, spacing, effects |
| btt_stage3 | 150 | BTT | Advanced: curves, full augmentation, variable length |
| **TOTAL** | **1800** | All | Complete feature coverage (12 stages) |

## Troubleshooting

### "No such file or directory: data.nosync/fonts"

You need to set up font, background, and corpus directories. Minimal setup:

```bash
mkdir -p data.nosync/fonts
mkdir -p data.nosync/backgrounds/solid_white
mkdir -p data.nosync/corpus_text/ltr

# Add at least one font (copy a .ttf file)
# Add corpus files with text
# Backgrounds directory can be empty for solid white
```

### "Proportions do not sum to 1.0"

This is a validation error in the YAML config. The demo configs are already correct. If you modified them, ensure proportions sum to exactly 1.0.

### Generation is slow

The demos use small worker counts (2 workers) for compatibility. For faster generation:

```bash
--generation-workers 8 \
--workers 8 \
--chunk-size 200 \
--io-batch-size 50
```

## Next Steps

1. **Run the quick test** (LTR Stage 1)
2. **Inspect 5-10 generated images** visually
3. **Check 2-3 JSON files** for correct structure
4. **Run full demo suite** if quick test passes
5. **Review all outputs** systematically
6. **Try your own configurations** based on demo examples

## Getting Help

- See `configs/README_MULTILINE_DEMOS.md` for detailed configuration documentation
- See `docs/how-to/multiline_text.md` for usage guide
- See `docs/api/generator.md` for API reference
- See `docs/api/text_layout.md` for line breaking details

## Configuration Files Created

**Demo Configs** (18 files in `configs/`):
- `demo_multiline_ltr_stage1.yaml` - LTR basic features
- `demo_multiline_ltr_stage2.yaml` - LTR intermediate features
- `demo_multiline_ltr_stage3.yaml` - LTR advanced features
- `demo_multiline_rtl_stage1.yaml` - RTL basic features
- `demo_multiline_rtl_stage2.yaml` - RTL intermediate features
- `demo_multiline_rtl_stage3.yaml` - RTL advanced features
- `demo_multiline_ttb_stage1.yaml` - TTB basic features
- `demo_multiline_ttb_stage2.yaml` - TTB intermediate features
- `demo_multiline_ttb_stage3.yaml` - TTB advanced features
- `demo_multiline_btt_stage1.yaml` - BTT basic features
- `demo_multiline_btt_stage2.yaml` - BTT intermediate features
- `demo_multiline_btt_stage3.yaml` - BTT advanced features

**Demo Scripts** (5 files in `scripts/`):
- `demo_multiline_ltr.sh` - Run all 3 LTR stages (450 images)
- `demo_multiline_rtl.sh` - Run all 3 RTL stages (450 images)
- `demo_multiline_ttb.sh` - Run all 3 TTB stages (450 images)
- `demo_multiline_btt.sh` - Run all 3 BTT stages (450 images)
- `demo_multiline_all.sh` - Run ALL 12 stages (1800 images)

All scripts are executable and ready to run!
