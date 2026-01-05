# LOOCV = gold standard when N < 100, a script that automates the running of 40 experiments.

#!/bin/bash
set -e
JSON_FILES="data/ST-2024-0002.json"
SCHEMA_PATH="data/schema.json"
OUTPUT_BASE="./checkpoints"
RESULTS_DIR="./results"
echo "Starting SSL pretraining..."
python src/train_ssl.py \
 --json_files $JSON_FILES \
 --schema_path $SCHEMA_PATH \
 --output_dir $OUTPUT_BASE/ssl \
 --epochs 100 \
 --batch_size 32 \
 --objective contrastive \
 --device cuda
echo "SSL pretraining complete!"
echo "Starting classifier training..."
python src/train_classifier.py \
 --json_files $JSON_FILES \
 --schema_path $SCHEMA_PATH \
 --pretrained_encoder $OUTPUT_BASE/ssl/best_encoder.pt \
 --output_dir $OUTPUT_BASE/classifier \
 --epochs 100 \
 --batch_size 16 \
 --freeze_encoder True \
 --device cuda
echo "Classifier training complete!"
echo "Running LOOCV evaluation..."
python src/evaluate.py \
 --json_files $JSON_FILES \
 --schema_path $SCHEMA_PATH \
 --classifier_path $OUTPUT_BASE/classifier/best_classifier.pt \
 --output_dir $RESULTS_DIR \
 --device cuda
echo "Pipeline complete! Results saved to $RESULTS_DIR"
