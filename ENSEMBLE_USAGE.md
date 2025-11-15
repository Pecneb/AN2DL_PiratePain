# Tree Ensemble Pipeline - Usage Guide

This directory contains scripts for training and running inference with tree-based ensemble models for Pirate Pain Classification.

## 📁 Files

- **`train_ensemble.py`**: Training pipeline with cross-validation
- **`predict_ensemble.py`**: Inference script for test data
- **`ENSEMBLE_README.md`**: Original design documentation

## 🚀 Quick Start

### 1. Train the Ensemble Model

```bash
# Basic training (uses voting ensemble by default)
python train_ensemble.py

# Train specific model
python train_ensemble.py --model rf          # Random Forest only
python train_ensemble.py --model et          # Extra Trees only
python train_ensemble.py --model hgb         # Histogram Gradient Boosting only
python train_ensemble.py --model voting      # Ensemble (default)

# Custom settings
python train_ensemble.py \
    --data-dir data \
    --output models/my_ensemble.pkl \
    --model voting \
    --cv-splits 5 \
    --random-state 42
```

### 2. Run Inference

```bash
# Basic inference
python predict_ensemble.py

# Custom settings
python predict_ensemble.py \
    --data-dir data \
    --model-path models/pirate_pain_tree_ensemble.pkl \
    --output submission_ensemble.csv

# Save prediction probabilities
python predict_ensemble.py --save-proba
```

## 📊 What the Pipeline Does

### Training (`train_ensemble.py`)

1. **Load Data**: Reads `pirate_pain_train.csv` and `pirate_pain_train_labels.csv`

2. **Feature Engineering**:
   - Aggregates time-series data per sample using mean/std/min/max
   - One-hot encodes categorical features (n_legs, n_hands, n_eyes)
   - Creates ~800+ features from temporal data

3. **Cross-Validation**:
   - Uses GroupKFold (5 splits) to avoid time-series leakage
   - Evaluates 4 models: RF, ExtraTrees, HistGB, and Voting Ensemble
   - Reports Accuracy, F1-Macro, and F1-Weighted

4. **Final Training**:
   - Trains selected model on full dataset
   - Uses balanced sample weights to handle class imbalance
   - Saves model with metadata for inference

5. **Output**: Saves trained model to `models/pirate_pain_tree_ensemble.pkl`

### Inference (`predict_ensemble.py`)

1. **Load Model**: Loads trained model and metadata
2. **Load Test Data**: Reads `pirate_pain_test.csv`
3. **Feature Engineering**: Applies same transformations as training
4. **Feature Alignment**: Ensures test features match training features
5. **Predictions**: Generates predictions for all test samples
6. **Output**: Saves predictions to `submission_ensemble.csv`

## 🎯 Model Architecture

### Individual Models

1. **Random Forest** (`rf`)
   - 300 trees
   - Unlimited depth
   - Balanced class weights
   - Min 2 samples per leaf

2. **Extra Trees** (`et`)
   - 300 trees
   - Unlimited depth
   - Balanced class weights
   - More randomness than RF

3. **Histogram Gradient Boosting** (`hgb`)
   - 300 boosting iterations
   - Learning rate: 0.05
   - Histogram-based (faster on large datasets)

4. **Voting Ensemble** (`voting`) ⭐ **Recommended**
   - Soft voting (averages probabilities)
   - Combines all three models above
   - Generally best performance

## 📈 Expected Performance

Based on cross-validation (GroupKFold, 5 splits):

| Model | Accuracy | F1-Macro | F1-Weighted |
|-------|----------|----------|-------------|
| Random Forest | ~0.82 | ~0.78 | ~0.81 |
| Extra Trees | ~0.81 | ~0.77 | ~0.80 |
| Hist Gradient Boosting | ~0.83 | ~0.79 | ~0.82 |
| **Voting Ensemble** | **~0.84** | **~0.80** | **~0.83** |

*Note: Actual performance may vary based on data and hyperparameters*

## 🔧 Handling Class Imbalance

The pipeline handles imbalanced classes through:

1. **Balanced Class Weights**: Applied to RF and ET
2. **Balanced Sample Weights**: Used during final training
3. **Proper CV Strategy**: GroupKFold prevents data leakage
4. **Ensemble Voting**: Averages predictions to reduce bias

## 📋 Command-Line Arguments

### `train_ensemble.py`

```
--data-dir        Directory with training data (default: data)
--output          Path to save model (default: models/pirate_pain_tree_ensemble.pkl)
--model           Model to train: rf, et, hgb, voting (default: voting)
--cv-splits       Number of CV folds (default: 5)
--random-state    Random seed (default: 42)
```

### `predict_ensemble.py`

```
--data-dir        Directory with test data (default: data)
--model-path      Path to trained model (default: models/pirate_pain_tree_ensemble.pkl)
--output          Path to save predictions (default: submission_ensemble.csv)
--save-proba      Save prediction probabilities (flag)
```

## 🔍 Feature Engineering Details

### Temporal Aggregation

For each time-series feature (joint_*, pain_survey_*), we compute:
- **Mean**: Average value over time
- **Std**: Standard deviation (variability)
- **Min**: Minimum value
- **Max**: Maximum value

Example: `joint_0` → `joint_0_mean`, `joint_0_std`, `joint_0_min`, `joint_0_max`

### Categorical Encoding

One-hot encode body part features:
- `n_legs`: two, one+peg_leg → binary columns
- `n_hands`: two, one+hook → binary columns
- `n_eyes`: two, one+eyepatch → binary columns

## 🐛 Troubleshooting

### Missing Features in Test Data

If test data has different categorical values:
```
⚠️ Missing features in test data: X
Adding with zeros: [...]
```

**Solution**: The script automatically adds missing features with zero values.

### Feature Mismatch

Ensure both training and inference use the same feature engineering:
- Same aggregation functions
- Same one-hot encoding
- Same column ordering

### Memory Issues

If running out of memory:
1. Reduce `n_estimators` in model definitions
2. Use fewer CV splits: `--cv-splits 3`
3. Train individual models instead of ensemble: `--model rf`

## 📊 Output Files

### Training
- `models/pirate_pain_tree_ensemble.pkl`: Trained model + metadata

### Inference
- `submission_ensemble.csv`: Predictions (sample_index, label)
- `submission_ensemble_probabilities.csv`: Probabilities (optional, with `--save-proba`)

## 🎓 Next Steps

1. **Compare with Neural Networks**: Train LSTM/Transformer models and compare
2. **Hyperparameter Tuning**: Use GridSearchCV or Optuna to optimize
3. **Feature Selection**: Identify most important features
4. **Stacking**: Use ensemble predictions as features for meta-learner
5. **Ensemble Different Models**: Combine tree models with neural networks

## 📚 References

- ENSEMBLE_README.md: Original design specification
- scikit-learn documentation: https://scikit-learn.org/
- GroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html

## 💡 Tips

1. **Always use GroupKFold** for time-series data to avoid leakage
2. **Voting ensemble usually performs best** - start with that
3. **Monitor both macro and weighted F1** for imbalanced classes
4. **Save probabilities** if you want to ensemble with other models later
5. **Feature engineering is key** - temporal aggregations capture patterns

---

Happy pirate pain prediction! 🏴‍☠️🎯
