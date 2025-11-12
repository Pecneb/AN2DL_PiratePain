#!/bin/bash

# Complete Pipeline Script for Pirate Pain Classification
# This script runs the entire workflow from hyperparameter tuning to prediction

set -e  # Exit on error

echo "================================"
echo "Pirate Pain Classification Pipeline"
echo "================================"
echo ""

# Configuration
N_TRIALS=50
K_FOLDS=5
MAX_EPOCHS=50
PATIENCE=10
FINAL_EPOCHS=100
FINAL_PATIENCE=15
SCALING="extra"

# Activate virtual environment
source .venv/bin/activate

# Step 1: Hyperparameter Tuning
echo "Step 1/3: Running Optuna hyperparameter optimization..."
echo "  Trials: $N_TRIALS"
echo "  K-Folds: $K_FOLDS"
echo "  Max Epochs: $MAX_EPOCHS"
echo ""

python hyperparameter_tuning.py \
    --method optuna \
    --n_trials $N_TRIALS \
    --k_folds $K_FOLDS \
    --epochs $MAX_EPOCHS \
    --patience $PATIENCE \
    --scaling $SCALING

echo ""
echo "✅ Hyperparameter optimization complete!"
echo ""

# Check if best config exists
if [ ! -f "./lightning_logs_tuning/best_config_optuna.json" ]; then
    echo "❌ Best config not found. Hyperparameter tuning may have failed."
    exit 1
fi

# Display best config
echo "Best Configuration:"
cat ./lightning_logs_tuning/best_config_optuna.json
echo ""

# Step 2: Train Final Model
echo "Step 2/3: Training final model on full dataset..."
echo "  Max Epochs: $FINAL_EPOCHS"
echo "  Patience: $FINAL_PATIENCE"
echo ""

python train_final_model.py \
    --config ./lightning_logs_tuning/best_config_optuna.json \
    --max_epochs $FINAL_EPOCHS \
    --patience $FINAL_PATIENCE \
    --scaling $SCALING

echo ""
echo "✅ Final model training complete!"
echo ""

# Check if model exists
if [ ! -f "./final_model/pirate_pain_best_model.ckpt" ]; then
    echo "❌ Model checkpoint not found. Training may have failed."
    exit 1
fi

# Step 3: Generate Predictions
echo "Step 3/3: Generating predictions on test data..."
echo ""

python predict.py \
    --model ./final_model/pirate_pain_best_model.ckpt \
    --test_data ./data/pirate_pain_test.csv \
    --output ./submission.csv

echo ""
echo "✅ Predictions generated!"
echo ""

# Verify submission
if [ -f "./submission.csv" ]; then
    echo "Submission file created: ./submission.csv"
    echo ""
    echo "First 5 predictions:"
    head -6 ./submission.csv
    echo ""
    echo "Prediction distribution:"
    python -c "import pandas as pd; print(pd.read_csv('submission.csv')['pain_level'].value_counts())"
    echo ""
else
    echo "❌ Submission file not created. Prediction may have failed."
    exit 1
fi

# Summary
echo "================================"
echo "Pipeline Complete! 🚀"
echo "================================"
echo ""
echo "Results:"
echo "  - Best config: ./lightning_logs_tuning/best_config_optuna.json"
echo "  - Trained model: ./final_model/pirate_pain_best_model.ckpt"
echo "  - Submission: ./submission.csv"
echo ""
echo "Next steps:"
echo "  1. Review visualizations: open ./lightning_logs_tuning/*.html"
echo "  2. Check TensorBoard logs: tensorboard --logdir ./final_model"
echo "  3. Submit ./submission.csv to the competition!"
echo ""
