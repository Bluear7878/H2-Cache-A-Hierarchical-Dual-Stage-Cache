#!/bin/bash

# eval "$(conda shell.bash hook)"
conda activate nunchaku

MULTI_VALUES=(0.15)
SINGLE_VALUES=(0.22)

BASE_RESULTS_DIR="./all_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_RESULTS_DIR"

for multi in "${MULTI_VALUES[@]}"; do
  for single in "${SINGLE_VALUES[@]}"; do
    
    SAVE_DIR="${BASE_RESULTS_DIR}/multi_${multi}_single_${single}"
    
    echo ""
    echo "############################################################"
    echo "##  Execute: Multi=${multi}, Single=${single}"
    echo "##  SVAE Folder: ${SAVE_DIR}"
    echo "############################################################"
    
    python3 total_test.py \
      --multi_threshold "$multi" \
      --single_threshold "$single" \
      --save_root "$SAVE_DIR"
      
    if [ $? -ne 0 ]; then
        echo "❌ Error: Multi=${multi}, Single=${single} Failed."
    fi
    
  done
done

echo ""
echo "✅ Done."