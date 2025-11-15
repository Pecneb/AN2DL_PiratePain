"""
Inference Script for Tree Ensemble Pirate Pain Classifier

This script loads a trained ensemble model and makes predictions on test data.

Usage:
    python predict_ensemble.py
    python predict_ensemble.py --data-dir data --model-path models/pirate_pain_tree_ensemble.pkl
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

import joblib


# ============================================
# 1. LOAD MODEL
# ============================================

def load_model(model_path: str):
    """
    Load trained model package.
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Dictionary with model, label_encoder, and metadata
    """
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    
    model_package = joblib.load(model_path)
    
    print(f"✓ Model loaded from: {model_path}")
    print(f"  Model type: {model_package['model_type']}")
    print(f"  Features: {model_package['n_features']}")
    print(f"  Classes: {model_package['classes']}")
    
    return model_package


# ============================================
# 2. FEATURE ENGINEERING (SAME AS TRAINING)
# ============================================

def build_feature_table(ts_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Aggregates each pirate's time series into a single feature vector.
    MUST match the feature engineering used during training!
    
    Args:
        ts_df: DataFrame with time-series data per sample
        verbose: Print feature engineering details
    
    Returns:
        DataFrame with aggregated features per sample_index
    """
    if verbose:
        print("\n" + "="*70)
        print("FEATURE ENGINEERING")
        print("="*70)
    
    # Identify time-varying numeric columns
    survey_cols = [c for c in ts_df.columns if c.startswith("pain_survey_")]
    joint_cols = [c for c in ts_df.columns if c.startswith("joint_")]
    num_cols = survey_cols + joint_cols
    
    if verbose:
        print(f"Survey columns: {len(survey_cols)}")
        print(f"Joint columns: {len(joint_cols)}")
        print(f"Total numeric features: {len(num_cols)}")
    
    # Aggregate numeric features over time
    agg_funcs = ["mean", "std", "min", "max"]
    feat_num = ts_df.groupby("sample_index")[num_cols].agg(agg_funcs)
    feat_num.columns = [f"{c}_{func}" for c, func in feat_num.columns]
    
    # Static categorical features
    static_cat = ts_df.groupby("sample_index")[["n_legs", "n_hands", "n_eyes"]].first()
    
    # One-hot encode (same as training)
    static_cat_ohe = pd.get_dummies(
        static_cat,
        columns=["n_legs", "n_hands", "n_eyes"],
        drop_first=False,
        dtype=int,
    )
    
    # Combine features
    X = pd.concat([feat_num, static_cat_ohe], axis=1)
    
    if verbose:
        print(f"Total features per sample: {len(X.columns)}")
        print(f"Total samples: {len(X)}")
    
    return X


# ============================================
# 3. ALIGN FEATURES WITH TRAINING
# ============================================

def align_features(X_test: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    """
    Ensure test features match training features exactly.
    
    Args:
        X_test: Test feature matrix
        expected_features: List of feature names from training
    
    Returns:
        Aligned feature matrix
    """
    print("\n" + "="*70)
    print("ALIGNING FEATURES")
    print("="*70)
    
    print(f"Expected features: {len(expected_features)}")
    print(f"Test features: {len(X_test.columns)}")
    
    # Find missing and extra features
    expected_set = set(expected_features)
    test_set = set(X_test.columns)
    
    missing_features = expected_set - test_set
    extra_features = test_set - expected_set
    
    if missing_features:
        print(f"\n⚠️  Missing features in test data: {len(missing_features)}")
        print(f"   Adding with zeros: {list(missing_features)[:5]}...")
        for feat in missing_features:
            X_test[feat] = 0
    
    if extra_features:
        print(f"\n⚠️  Extra features in test data: {len(extra_features)}")
        print(f"   Dropping: {list(extra_features)[:5]}...")
    
    # Reorder to match training
    X_aligned = X_test[expected_features]
    
    print(f"\n✓ Features aligned: {X_aligned.shape}")
    
    return X_aligned


# ============================================
# 4. MAKE PREDICTIONS
# ============================================

def predict(model, X: pd.DataFrame, label_encoder, return_proba: bool = False):
    """
    Make predictions on test data.
    
    Args:
        model: Trained model
        X: Feature matrix
        label_encoder: Label encoder for class names
        return_proba: Whether to return probabilities
    
    Returns:
        Predictions (and optionally probabilities)
    """
    print("\n" + "="*70)
    print("MAKING PREDICTIONS")
    print("="*70)
    
    print(f"Predicting for {len(X)} samples...")
    
    # Get predictions
    y_pred_encoded = model.predict(X)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    print("✓ Predictions complete!")
    
    # Class distribution
    unique, counts = np.unique(y_pred, return_counts=True)
    print("\nPredicted class distribution:")
    for class_name, count in zip(unique, counts):
        pct = count / len(y_pred) * 100
        print(f"  {class_name:12} : {count:4} ({pct:5.2f}%)")
    
    if return_proba:
        y_proba = model.predict_proba(X)
        print("\n✓ Probabilities computed")
        return y_pred, y_proba
    
    return y_pred


# ============================================
# 5. SAVE PREDICTIONS
# ============================================

def save_predictions(sample_indices: np.ndarray, predictions: np.ndarray,
                    output_path: str = "submission_ensemble.csv") -> None:
    """
    Save predictions to CSV file.
    
    Args:
        sample_indices: Sample indices
        predictions: Predicted labels
        output_path: Path to save predictions
    """
    print("\n" + "="*70)
    print("SAVING PREDICTIONS")
    print("="*70)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'sample_index': sample_indices,
        'label': predictions
    })
    
    # Save to CSV
    submission.to_csv(output_path, index=False)
    
    print(f"✓ Predictions saved to: {output_path}")
    print(f"  Rows: {len(submission)}")
    print(f"\nFirst 10 predictions:")
    print(submission.head(10))


# ============================================
# 6. MAIN INFERENCE PIPELINE
# ============================================

def main(data_dir: str = "data", model_path: str = "models/pirate_pain_tree_ensemble.pkl",
         output_path: str = "submission_ensemble.csv", save_proba: bool = False):
    """
    Main inference pipeline.
    
    Args:
        data_dir: Directory containing test data
        model_path: Path to trained model
        output_path: Path to save predictions
        save_proba: Whether to save probabilities
    """
    print("\n" + "="*70)
    print("🏴‍☠️ PIRATE PAIN CLASSIFICATION - ENSEMBLE INFERENCE 🏴‍☠️")
    print("="*70)
    
    # --- Load model ---
    model_package = load_model(model_path)
    model = model_package['model']
    label_encoder = model_package['label_encoder']
    expected_features = model_package['feature_columns']
    
    # --- Load test data ---
    print("\n📂 Loading test data...")
    data_dir = Path(data_dir)
    test_df = pd.read_csv(data_dir / "pirate_pain_test.csv")
    
    print(f"✓ Loaded {len(test_df)} rows")
    print(f"✓ Unique samples: {test_df['sample_index'].nunique()}")
    
    # --- Feature engineering ---
    X_test = build_feature_table(test_df, verbose=True)
    
    # --- Align features ---
    X_test_aligned = align_features(X_test, expected_features)
    
    # --- Make predictions ---
    if save_proba:
        predictions, probabilities = predict(
            model, X_test_aligned, label_encoder, return_proba=True
        )
    else:
        predictions = predict(model, X_test_aligned, label_encoder)
    
    # --- Save predictions ---
    sample_indices = X_test.index.values
    save_predictions(sample_indices, predictions, output_path)
    
    # --- Save probabilities if requested ---
    if save_proba:
        proba_path = output_path.replace('.csv', '_probabilities.csv')
        proba_df = pd.DataFrame(
            probabilities,
            columns=[f'prob_{cls}' for cls in label_encoder.classes_]
        )
        proba_df.insert(0, 'sample_index', sample_indices)
        proba_df.to_csv(proba_path, index=False)
        print(f"\n✓ Probabilities saved to: {proba_path}")
    
    print("\n" + "="*70)
    print("✅ INFERENCE COMPLETE!")
    print("="*70)
    print(f"\nSubmission file: {output_path}")
    print("Ready to submit! 🎯")


# ============================================
# 7. COMMAND-LINE INTERFACE
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with trained tree ensemble model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing test data (default: data)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/pirate_pain_tree_ensemble.pkl",
        help="Path to trained model (default: models/pirate_pain_tree_ensemble.pkl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission_ensemble.csv",
        help="Path to save predictions (default: submission_ensemble.csv)"
    )
    parser.add_argument(
        "--save-proba",
        action="store_true",
        help="Save prediction probabilities to separate file"
    )
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_path=args.output,
        save_proba=args.save_proba
    )
