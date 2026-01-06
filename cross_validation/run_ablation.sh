#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
echo "Liver Transplantability - Ablation Study"
echo "=========================================="
JSON_FILES="data/example_donor.json"
SCHEMA_PATH="data/schema.json"
ABLATION_DIR="ablation_results"
mkdir -p "$ABLATION_DIR"
DEVICE="mps"
if [[ "$1" == "--cpu" ]]; then
 DEVICE="cpu"
 echo "Using CPU mode"
fi
if [[ "$1" == "--cuda" ]]; then
 DEVICE="cuda"
 echo "Using CUDA mode"
fi
echo "Using device: $DEVICE"
echo ""
echo "Running Ablation Study..."
echo "=========================="
python src/ablation.py \
 --json_files $JSON_FILES \
 --schema_path $SCHEMA_PATH \
 --output_dir "$ABLATION_DIR" \
 --epochs 100 \
 --batch_size 8 \
 --device $DEVICE
echo ""
echo "Ablation study complete!"
echo "Results saved to: $ABLATION_DIR/ablation_results.json"
echo ""
echo "To view results:"
echo "  cat $ABLATION_DIR/ablation_results.json"
