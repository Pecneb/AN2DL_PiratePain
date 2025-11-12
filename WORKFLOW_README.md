# Complete Workflow: From Hyperparameter Tuning to Submission

This guide walks you through the complete process of training and deploying your Pirate Pain Classification model.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Step 1: Hyperparameter Optimization](#step-1-hyperparameter-optimization)
3. [Step 2: Train Final Model](#step-2-train-final-model)
4. [Step 3: Generate Predictions](#step-3-generate-predictions)
5. [Monitoring and Analysis](#monitoring-and-analysis)
6. [Troubleshooting](#troubleshooting)

---

## Overview

**Workflow Steps:**
```
Hyperparameter Tuning → Train Final Model → Generate Predictions → Submit
    (optuna)              (full dataset)      (test data)
```

**Files:**
- `hyperparameter_tuning.py` - Find best hyperparameters using Optuna
- `train_final_model.py` - Train final model on full training set
- `predict.py` - Generate predictions on test data

---

## Step 1: Hyperparameter Optimization

Find the best hyperparameters using Optuna's Bayesian optimization.

### Quick Test Run (5 minutes)
```bash
.venv/bin/python hyperparameter_tuning.py \
    --method optuna \
    --n_trials 10 \
    --k_folds 3 \
    --epochs 20 \
    --patience 5
```

### Full Optimization (2-4 hours)
```bash
nohup .venv/bin/python hyperparameter_tuning.py \
    --method optuna \
    --n_trials 50 \
    --k_folds 5 \
    --epochs 50 \
    --patience 10 \
    > tuning_output.log 2>&1 &
```

### Monitor Progress
```bash
# View live output
tail -f tuning_output.log

# Check number of completed trials
grep "Trial.*Result" tuning_output.log | wc -l

# See current best score
grep "Best is trial" tuning_output.log | tail -1
```

### Check Results
```bash
# Get process ID
jobs -l

# Check if still running
ps aux | grep hyperparameter_tuning

# Kill if needed
kill <PID>
```

### Output Files
After completion, you'll have:
- `./lightning_logs_tuning/best_config_optuna.json` - Best hyperparameters
- `./hyperparameter_search_results.csv` - All trial results
- `./lightning_logs_tuning/*.html` - Interactive visualizations
- `./lightning_logs_tuning/` - TensorBoard logs

### View Best Configuration
```bash
cat ./lightning_logs_tuning/best_config_optuna.json
```

Example output:
```json
{
  "hidden_size": 128,
  "num_layers": 2,
  "rnn_type": "LSTM",
  "bidirectional": true,
  "dropout_rate": 0.23,
  "learning_rate": 0.0023,
  "weight_decay": 4.5e-06,
  "batch_size": 32
}
```

---

## Step 2: Train Final Model

Train a model on the **full training dataset** using the best hyperparameters.

### Command
```bash
.venv/bin/python train_final_model.py \
    --config ./lightning_logs_tuning/best_config_optuna.json \
    --max_epochs 100 \
    --patience 15 \
    --scaling extra
```

### Options
- `--config`: Path to best config JSON (from Step 1)
- `--max_epochs`: Maximum training epochs (default: 100)
- `--patience`: Early stopping patience (default: 15)
- `--scaling`: Scaling method - `inter`, `extra`, or `hybrid` (default: `extra`)
- `--output_dir`: Directory to save model (default: `./final_model`)

### Output Files
After training, you'll have:
- `./final_model/pirate_pain_best_model.ckpt` - Best model checkpoint
- `./final_model/pirate_pain_best_model-last.ckpt` - Last checkpoint
- `./final_model/training_info.json` - Training configuration
- `./final_model/scaler.pkl` - Fitted scaler (if using `extra` or `hybrid` scaling)
- `./final_model/final_model_logs/` - TensorBoard logs

### Monitor Training
```bash
# View TensorBoard logs
tensorboard --logdir ./final_model

# Then open: http://localhost:6006
```

---

## Step 3: Generate Predictions

Generate predictions on test data using your trained model.

### Command
```bash
.venv/bin/python predict.py \
    --model ./final_model/pirate_pain_best_model.ckpt \
    --test_data ./data/pirate_pain_test.csv \
    --output ./submission.csv
```

### Options
- `--model`: Path to trained model checkpoint
- `--test_data`: Path to test data CSV
- `--output`: Path to save submission CSV (default: `./submission.csv`)
- `--batch_size`: Batch size for inference (default: 32)
- `--info`: Path to training_info.json (auto-detected)

### Output Files
- `./submission.csv` - Competition submission file
- `./submission_probabilities.csv` - Prediction probabilities for analysis

### Example Output
```
sample_index  pain_level
0             no_pain
1             low_pain
2             high_pain
3             no_pain
...
```

### Verify Submission
```bash
# Check file format
head submission.csv

# Count predictions by class
.venv/bin/python -c "import pandas as pd; print(pd.read_csv('submission.csv')['pain_level'].value_counts())"
```

---

## Monitoring and Analysis

### View Optuna Visualizations
```bash
# Open in browser
open ./lightning_logs_tuning/optimization_history.html
open ./lightning_logs_tuning/param_importances.html
open ./lightning_logs_tuning/parallel_coordinate.html
```

### View All Trial Results
```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('./hyperparameter_search_results.csv')
print('Top 10 Trials:')
print(df.sort_values('value', ascending=False)[['number', 'value', 'params_hidden_size', 'params_rnn_type', 'params_bidirectional']].head(10))
"
```

### Analyze Predictions
```bash
.venv/bin/python -c "
import pandas as pd
import numpy as np

# Load predictions
pred = pd.read_csv('submission.csv')
prob = pd.read_csv('submission_probabilities.csv')

print('Prediction Distribution:')
print(pred['pain_level'].value_counts())
print()

# Show confidence statistics
print('Prediction Confidence (max probability):')
max_probs = prob[['prob_no_pain', 'prob_low_pain', 'prob_high_pain']].max(axis=1)
print(f'Mean: {max_probs.mean():.3f}')
print(f'Min: {max_probs.min():.3f}')
print(f'Max: {max_probs.max():.3f}')
print()

# Find low-confidence predictions
low_conf = max_probs < 0.5
print(f'Low-confidence predictions (< 0.5): {low_conf.sum()} ({low_conf.sum()/len(pred)*100:.1f}%)')
"
```

---

## Troubleshooting

### Issue: "Config file not found"
**Solution:** Run hyperparameter tuning first:
```bash
.venv/bin/python hyperparameter_tuning.py --method optuna --n_trials 10
```

### Issue: "Model file not found"
**Solution:** Train the final model first:
```bash
.venv/bin/python train_final_model.py
```

### Issue: "Scaler not found" warning during prediction
**Cause:** Using `extra` or `hybrid` scaling but scaler wasn't saved

**Solution:** Retrain the model:
```bash
.venv/bin/python train_final_model.py --scaling extra
```

### Issue: Out of memory during training
**Solution:** Reduce batch size:
```bash
# Edit best_config_optuna.json and change batch_size to 16 or 8
# Then retrain
```

### Issue: Training too slow
**Solution:** Check device:
```bash
.venv/bin/python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('MPS available:', torch.backends.mps.is_available())
"
```

### Issue: Predictions seem wrong
**Checks:**
1. Verify test data preprocessing matches training
2. Check scaling method matches training
3. Review prediction probabilities for confidence
4. Compare distribution with training labels

```bash
# Compare distributions
.venv/bin/python -c "
import pandas as pd
train_labels = pd.read_csv('./data/pirate_pain_train_labels.csv')
predictions = pd.read_csv('./submission.csv')

print('Training Distribution:')
print(train_labels['pain_level'].value_counts(normalize=True))
print('\nPrediction Distribution:')
print(predictions['pain_level'].value_counts(normalize=True))
"
```

---

## Complete Pipeline Example

Here's a complete example running all steps:

```bash
# Step 1: Hyperparameter tuning (run in background)
nohup .venv/bin/python hyperparameter_tuning.py \
    --method optuna \
    --n_trials 50 \
    --k_folds 5 \
    --epochs 50 \
    --patience 10 \
    > tuning_output.log 2>&1 &

# Monitor progress
tail -f tuning_output.log
# Press Ctrl+C to stop viewing (process continues in background)

# Wait for completion (check with: grep "OPTIMIZATION COMPLETE" tuning_output.log)

# Step 2: View best config
cat ./lightning_logs_tuning/best_config_optuna.json

# Step 3: Train final model
.venv/bin/python train_final_model.py \
    --config ./lightning_logs_tuning/best_config_optuna.json \
    --max_epochs 100 \
    --patience 15

# Step 4: Generate predictions
.venv/bin/python predict.py \
    --model ./final_model/pirate_pain_best_model.ckpt \
    --test_data ./data/pirate_pain_test.csv \
    --output ./submission.csv

# Step 5: Verify submission
head submission.csv
wc -l submission.csv

# Step 6: Submit to competition! 🚀
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Start hyperparameter tuning | `python hyperparameter_tuning.py --method optuna --n_trials 50` |
| Monitor tuning progress | `tail -f tuning_output.log` |
| Train final model | `python train_final_model.py` |
| Generate predictions | `python predict.py` |
| View TensorBoard | `tensorboard --logdir ./final_model` |
| Open Optuna visualizations | `open ./lightning_logs_tuning/*.html` |

---

## Tips for Best Results

1. **Hyperparameter Tuning:**
   - Start with 10 trials for testing, then 50-100 for production
   - More trials = better results but longer runtime
   - Use `--k_folds 5` for robust evaluation

2. **Final Training:**
   - Use `--max_epochs 100` or higher
   - Monitor TensorBoard to ensure convergence
   - Save checkpoints in case training interrupts

3. **Prediction:**
   - Always check prediction probabilities
   - Low confidence may indicate need for more training
   - Verify submission format before uploading

4. **Experiment:**
   - Try different scaling methods (`inter`, `extra`, `hybrid`)
   - Compare results from multiple Optuna runs
   - Ensemble predictions from multiple models

---

**Good luck with your submission! 🚀**
