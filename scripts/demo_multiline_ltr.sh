#!/bin/bash
# Demo script for LTR multi-line text generation
# Tests all new multi-line features with left-to-right text

echo "====================================================================="
echo "LTR Multi-Line Demo - Stage 1: Basic Features"
echo "====================================================================="
echo "Testing:"
echo "  - Backward compatibility (single-line mode)"
echo "  - Multi-line with word breaking"
echo "  - Multi-line with character breaking"
echo ""

python3 -m src.main \
--batch-config configs/demo_multiline_ltr_stage1.yaml \
--output-dir output.nosync/demo_multiline/ltr_stage1 \
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
echo "LTR Multi-Line Demo - Stage 2: Intermediate Features"
echo "====================================================================="
echo "Testing:"
echo "  - Left, center, and right text alignment"
echo "  - Variable line spacing"
echo "  - Effects applied uniformly across lines"
echo ""

python3 -m src.main \
--batch-config configs/demo_multiline_ltr_stage2.yaml \
--output-dir output.nosync/demo_multiline/ltr_stage2 \
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
echo "LTR Multi-Line Demo - Stage 3: Advanced Features"
echo "====================================================================="
echo "Testing:"
echo "  - Multi-line with curved text (arc curves)"
echo "  - Multi-line with full augmentation pipeline"
echo "  - Variable text length (1 char to paragraph)"
echo ""

python3 -m src.main \
--batch-config configs/demo_multiline_ltr_stage3.yaml \
--output-dir output.nosync/demo_multiline/ltr_stage3 \
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
echo "LTR Multi-Line Demo Complete!"
echo "====================================================================="
echo "Output directories:"
echo "  - output.nosync/demo_multiline/ltr_stage1"
echo "  - output.nosync/demo_multiline/ltr_stage2"
echo "  - output.nosync/demo_multiline/ltr_stage3"
echo ""
echo "Review the generated images to verify:"
echo "  1. Single-line images work correctly (backward compatibility)"
echo "  2. Multi-line word breaking keeps words intact"
echo "  3. Multi-line character breaking splits anywhere"
echo "  4. Text alignments (left, center, right) are correct"
echo "  5. Effects are applied uniformly across all lines"
echo "  6. Curved text works with multi-line"
echo "====================================================================="
