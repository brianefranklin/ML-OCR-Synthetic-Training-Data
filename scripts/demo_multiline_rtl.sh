#!/bin/bash
# Demo script for RTL multi-line text generation
# Tests multi-line features with right-to-left text

echo "====================================================================="
echo "RTL Multi-Line Demo - Stage 1: Basic Features"
echo "====================================================================="
echo "Testing:"
echo "  - Backward compatibility (single-line RTL)"
echo "  - Multi-line RTL with word breaking"
echo "  - Multi-line RTL with character breaking"
echo ""

python3 -m src.main \
--batch-config configs/demo_multiline_rtl_stage1.yaml \
--output-dir output.nosync/demo_multiline/rtl_stage1 \
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
echo "RTL Multi-Line Demo - Stage 2: Intermediate Features"
echo "====================================================================="
echo "Testing:"
echo "  - Right, center, and left text alignment for RTL"
echo "  - Variable line spacing"
echo "  - Effects applied uniformly across lines"
echo ""

python3 -m src.main \
--batch-config configs/demo_multiline_rtl_stage2.yaml \
--output-dir output.nosync/demo_multiline/rtl_stage2 \
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
echo "RTL Multi-Line Demo - Stage 3: Advanced Features"
echo "====================================================================="
echo "Testing:"
echo "  - Multi-line RTL with curved text (arc curves)"
echo "  - Multi-line RTL with full augmentation pipeline"
echo "  - Variable text length (1 char to paragraph)"
echo ""

python3 -m src.main \
--batch-config configs/demo_multiline_rtl_stage3.yaml \
--output-dir output.nosync/demo_multiline/rtl_stage3 \
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
echo "RTL Multi-Line Demo Complete!"
echo "====================================================================="
echo "Output directories:"
echo "  - output.nosync/demo_multiline/rtl_stage1"
echo "  - output.nosync/demo_multiline/rtl_stage2"
echo "  - output.nosync/demo_multiline/rtl_stage3"
echo ""
echo "Review the generated images to verify:"
echo "  1. Single-line RTL images work correctly (backward compatibility)"
echo "  2. Multi-line RTL with word breaking"
echo "  3. Multi-line RTL with character breaking"
echo "  4. Text alignments (right, center, left) are correct for RTL"
echo "  5. Effects are applied uniformly across all lines"
echo "  6. Curved text works with multi-line RTL"
echo "====================================================================="
