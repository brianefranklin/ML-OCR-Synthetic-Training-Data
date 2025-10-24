#!/bin/bash
# Demo script for BTT (Bottom-to-Top) multi-line text generation
# Tests multi-line features with bottom-to-top vertical text

echo "====================================================================="
echo "BTT (Bottom-to-Top) Multi-Line Demo - Stage 1: Basic Features"
echo "====================================================================="
echo "Testing:"
echo "  - Backward compatibility (single-line BTT)"
echo "  - Multi-line BTT with character breaking"
echo "  - Multi-line BTT with word breaking"
echo ""

python3 -m src.main \
--batch-config configs/demo_multiline_btt_stage1.yaml \
--output-dir output.nosync/demo_multiline/btt_stage1 \
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
echo "BTT Multi-Line Demo - Stage 2: Intermediate Features"
echo "====================================================================="
echo "Testing:"
echo "  - Bottom, center, and top text alignment for BTT"
echo "  - Variable line spacing"
echo "  - Effects applied uniformly across lines"
echo ""

python3 -m src.main \
--batch-config configs/demo_multiline_btt_stage2.yaml \
--output-dir output.nosync/demo_multiline/btt_stage2 \
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
echo "BTT Multi-Line Demo - Stage 3: Advanced Features"
echo "====================================================================="
echo "Testing:"
echo "  - Multi-line BTT with curved text (arc curves)"
echo "  - Multi-line BTT with full augmentation pipeline"
echo "  - Variable text length (1 char to paragraph)"
echo ""

python3 -m src.main \
--batch-config configs/demo_multiline_btt_stage3.yaml \
--output-dir output.nosync/demo_multiline/btt_stage3 \
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
echo "BTT Multi-Line Demo Complete!"
echo "====================================================================="
echo "Output directories:"
echo "  - output.nosync/demo_multiline/btt_stage1"
echo "  - output.nosync/demo_multiline/btt_stage2"
echo "  - output.nosync/demo_multiline/btt_stage3"
echo ""
echo "Review the generated images to verify:"
echo "  1. Single-line BTT text works correctly (backward compatibility)"
echo "  2. Multi-line BTT with lines arranged horizontally"
echo "  3. Word and character breaking work for BTT"
echo "  4. Vertical alignments (bottom, center, top) are correct"
echo "  5. Effects are applied uniformly across all lines"
echo "  6. Curved text works with multi-line BTT"
echo "====================================================================="
