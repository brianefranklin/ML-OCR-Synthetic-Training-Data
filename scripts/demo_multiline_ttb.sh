#!/bin/bash
# Demo script for TTB (Top-to-Bottom) multi-line text generation
# Tests multi-line features with vertical text

echo "====================================================================="
echo "TTB (Top-to-Bottom) Multi-Line Demo - Stage 1: Basic Features"
echo "====================================================================="
echo "Testing:"
echo "  - Backward compatibility (single-line vertical)"
echo "  - Multi-line vertical with word breaking"
echo "  - Multi-line vertical with character breaking"
echo ""

python3 -m src.main \
--batch-config configs/demo_multiline_ttb_stage1.yaml \
--output-dir output.nosync/demo_multiline/ttb_stage1 \
--font-dir data.nosync/fonts \
--background-dir data.nosync/backgrounds/solid_white \
--corpus-dir data.nosync/corpus_text \
--generation-workers 2 \
--workers 2 \
--chunk-size 50 \
--io-batch-size 25 \
--log-level INFO

echo ""
echo "====================================================================="
echo "TTB Multi-Line Demo - Stage 2: Intermediate Features"
echo "====================================================================="
echo "Testing:"
echo "  - Top, center, and bottom text alignment for vertical text"
echo "  - Variable line spacing"
echo "  - Effects applied uniformly across lines"
echo ""

python3 -m src.main \
--batch-config configs/demo_multiline_ttb_stage2.yaml \
--output-dir output.nosync/demo_multiline/ttb_stage2 \
--font-dir data.nosync/fonts \
--background-dir data.nosync/backgrounds/solid_white \
--corpus-dir data.nosync/corpus_text \
--generation-workers 2 \
--workers 2 \
--chunk-size 50 \
--io-batch-size 25 \
--log-level INFO

echo ""
echo "====================================================================="
echo "TTB Multi-Line Demo - Stage 3: Advanced Features"
echo "====================================================================="
echo "Testing:"
echo "  - Multi-line vertical with curved text (arc curves)"
echo "  - Multi-line vertical with full augmentation pipeline"
echo "  - Variable text length (1 char to paragraph)"
echo ""

python3 -m src.main \
--batch-config configs/demo_multiline_ttb_stage3.yaml \
--output-dir output.nosync/demo_multiline/ttb_stage3 \
--font-dir data.nosync/fonts \
--background-dir data.nosync/backgrounds/solid_white \
--corpus-dir data.nosync/corpus_text \
--generation-workers 2 \
--workers 2 \
--chunk-size 50 \
--io-batch-size 25 \
--log-level INFO

echo ""
echo "====================================================================="
echo "TTB Multi-Line Demo Complete!"
echo "====================================================================="
echo "Output directories:"
echo "  - output.nosync/demo_multiline/ttb_stage1"
echo "  - output.nosync/demo_multiline/ttb_stage2"
echo "  - output.nosync/demo_multiline/ttb_stage3"
echo ""
echo "Review the generated images to verify:"
echo "  1. Single-line vertical text works correctly (backward compatibility)"
echo "  2. Multi-line vertical with lines arranged horizontally"
echo "  3. Word and character breaking work for vertical text"
echo "  4. Vertical alignments (top, center, bottom) are correct"
echo "  5. Effects are applied uniformly across all lines"
echo "  6. Curved text works with multi-line vertical"
echo "====================================================================="
