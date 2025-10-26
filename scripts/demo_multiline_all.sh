#!/bin/bash
# Master demo script - runs all multi-line demonstrations
# Tests all text directions and all new features

echo "###################################################################"
echo "#                                                                 #"
echo "#  Multi-Line Text Generation - Comprehensive Demo Suite         #"
echo "#                                                                 #"
echo "###################################################################"
echo ""
echo "This script will generate demo images for all text directions:"
echo "  - Left-to-Right (LTR) - 3 stages (450 images)"
echo "  - Right-to-Left (RTL) - 3 stages (450 images)"
echo "  - Top-to-Bottom (TTB) - 3 stages (450 images)"
echo "  - Bottom-to-Top (BTT) - 3 stages (450 images)"
echo ""
echo "Total images to generate: 1800"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."
echo ""

# LTR Demos (3 stages)
./scripts/demo_multiline_ltr.sh

echo ""
echo "Waiting 2 seconds before next demo..."
sleep 2
echo ""

# RTL Demo
./scripts/demo_multiline_rtl.sh

echo ""
echo "Waiting 2 seconds before next demo..."
sleep 2
echo ""

# TTB Demo
./scripts/demo_multiline_ttb.sh

echo ""
echo "Waiting 2 seconds before next demo..."
sleep 2
echo ""

# BTT Demo
./scripts/demo_multiline_btt.sh

echo ""
echo "###################################################################"
echo "#                                                                 #"
echo "#  All Multi-Line Demos Complete!                                #"
echo "#                                                                 #"
echo "###################################################################"
echo ""
echo "Generated 1800 demo images across all text directions (12 stages)."
echo ""
echo "Output structure:"
echo "  output.nosync/demo_multiline/"
echo "    ├── ltr_stage1/  (150 images - basic LTR features)"
echo "    ├── ltr_stage2/  (150 images - intermediate LTR features)"
echo "    ├── ltr_stage3/  (150 images - advanced LTR features)"
echo "    ├── rtl_stage1/  (150 images - basic RTL features)"
echo "    ├── rtl_stage2/  (150 images - intermediate RTL features)"
echo "    ├── rtl_stage3/  (150 images - advanced RTL features)"
echo "    ├── ttb_stage1/  (150 images - basic TTB features)"
echo "    ├── ttb_stage2/  (150 images - intermediate TTB features)"
echo "    ├── ttb_stage3/  (150 images - advanced TTB features)"
echo "    ├── btt_stage1/  (150 images - basic BTT features)"
echo "    ├── btt_stage2/  (150 images - intermediate BTT features)"
echo "    └── btt_stage3/  (150 images - advanced BTT features)"
echo ""
echo "Verification Checklist:"
echo "  ✓ Single-line mode (backward compatibility)"
echo "  ✓ Multi-line with word breaking"
echo "  ✓ Multi-line with character breaking"
echo "  ✓ Text alignments (left, center, right, top, bottom)"
echo "  ✓ Variable line spacing"
echo "  ✓ Effects applied uniformly across lines"
echo "  ✓ Curved text with multi-line"
echo "  ✓ Variable text length (1 char to paragraph)"
echo "  ✓ All text directions (LTR, RTL, TTB, BTT)"
echo ""
echo "Next steps:"
echo "  1. Review generated images visually"
echo "  2. Check JSON label files for 'line_index' field (multi-line)"
echo "  3. Verify no 'line_index' in single-line mode images"
echo "  4. Confirm effects are uniform across all lines"
echo ""
echo "###################################################################"
