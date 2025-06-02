#!/bin/bash

# OpenGenome to MDS Conversion Script for Biowulf
# This script runs the conversion with appropriate settings

set -e  # Exit on any error

echo "Starting OpenGenome to MDS conversion on Biowulf..."

# Activate the conda environment
echo "Activating conda environment..."
source activate opengenome_conversion

# Verify environment
echo "Verifying environment..."
python -c "import datasets; import streaming; print('Environment verified!')"

# Set the data path
DATA_PATH="/data/lindseylm/PROPHAGE_IDENTIFICATION_LLM/DATA"

# First, run a dry run to inspect the data
echo "=== DRY RUN: Inspecting OpenGenome dataset ==="
python convert_opengenome_to_mds.py \
    --data_path "$DATA_PATH" \
    --stages stage1 \
    --splits train \
    --dry_run

echo ""
echo "=== CONVERTING STAGE 1 (8k context) ==="
python convert_opengenome_to_mds.py \
    --data_path "$DATA_PATH" \
    --stages stage1 \
    --splits train validation test \
    --shard_size 100000

echo ""
echo "=== CONVERTING STAGE 2 (131k context) ==="
python convert_opengenome_to_mds.py \
    --data_path "$DATA_PATH" \
    --stages stage2 \
    --splits train validation test \
    --shard_size 50000  # Smaller shards for longer sequences

echo ""
echo "=== CONVERSION COMPLETE ==="
echo "Converted datasets available at: $DATA_PATH/opengenome_mds/"

# Display directory structure
echo ""
echo "Directory structure:"
ls -la "$DATA_PATH/opengenome_mds/"

echo ""
echo "Stage 1 structure:"
ls -la "$DATA_PATH/opengenome_mds/stage1/"

echo ""
echo "Stage 2 structure:"
ls -la "$DATA_PATH/opengenome_mds/stage2/"

# Show shard counts
echo ""
echo "Shard counts:"
for stage in stage1 stage2; do
    for split in train validation test; do
        shard_count=$(ls "$DATA_PATH/opengenome_mds/$stage/$split/"*.mds 2>/dev/null | wc -l)
        echo "  $stage/$split: $shard_count shards"
    done
done

echo ""
echo "Done! You can now use these MDS files with ModernBERT."
