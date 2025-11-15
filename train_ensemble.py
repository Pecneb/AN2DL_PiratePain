"""
Tree Ensemble Pipeline for Pirate Pain Classification

This script implements a complete tree-based ensemble pipeline that:
1. Loads and aggregates time-series data per sample
2. Engineers features from temporal data
3. Trains multiple tree-based models (RF, ExtraTrees, HistGB)
4. Combines them into a soft-voting ensemble
5. Uses GroupKFold for proper time-series cross-validation
6. Handles class imbalance with balanced weights
7. Saves the trained model for inference

Author: Based on ENSEMBLE_README.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    f1_score,
    accuracy_score
)
from sklearn.utils.class_weight import compute_sample_weight
import joblib


# ============================================
# 1. FEATURE ENGINEERING
# ============================================

def build_feature_table(ts_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Aggregates each pirate's time series (one sample_index) into a single
    feature vector.

    Args:
        ts_df: DataFrame with time-series data per sample
        verbose: Print feature engineering details

    Returns:
        DataFrame with aggregated features per sample_index
    
    Features created:
    - Aggregates pain_survey_* and joint_* over time with mean/std/min/max
    - Keeps n_legs, n_hands, n_eyes as one-hot encoded static features
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
        print(f"Total numeric time-varying features: {len(num_cols)}")
    
    # Aggregate numeric features over time
    agg_funcs = ["mean", "std", "min", "max"]
    feat_num = ts_df.groupby("sample_index")[num_cols].agg(agg_funcs)
    
    # Flatten MultiIndex columns: (col, func) -> "col_func"
    feat_num.columns = [f"{c}_{func}" for c, func in feat_num.columns]
    
    if verbose:
        print(f"Aggregated features per sample: {len(feat_num.columns)}")
    
    # Static categorical features (constant per sample)
    static_cat = ts_df.groupby("sample_index")[["n_legs", "n_hands", "n_eyes"]].first()
    
    # One-hot encode categorical features
    static_cat_ohe = pd.get_dummies(
        static_cat,
        columns=["n_legs", "n_hands", "n_eyes"],
        drop_first=False,
        dtype=int,
    )
    
    if verbose:
        print(f"One-hot encoded categorical features: {len(static_cat_ohe.columns)}")
    
    # Combine numeric + categorical features
    X = pd.concat([feat_num, static_cat_ohe], axis=1)
    
    if verbose:
        print(f"Total features per sample: {len(X.columns)}")
        print(f"Total samples: {len(X)}")
        print(f"Feature table shape: {X.shape}")
    
    return X


# ============================================
# 2. MODEL DEFINITIONS
# ============================================

def make_models(random_state: int = 42) -> dict:
    """
    Returns a dictionary of tree-based models and a soft-voting ensemble.
    
    Args:
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with model name -> model object
    """
    print("\n" + "="*70)
    print("MODEL DEFINITIONS")
    print("="*70)
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
        verbose=0
    )
    print("✓ Random Forest: 300 trees, balanced weights")
    
    # Extra Trees
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
        verbose=0
    )
    print("✓ Extra Trees: 300 trees, balanced weights")
    
    # Histogram-based Gradient Boosting
    hgb = HistGradientBoostingClassifier(
        max_depth=None,
        learning_rate=0.05,
        max_iter=300,
        random_state=random_state,
        verbose=0
    )
    print("✓ Histogram Gradient Boosting: lr=0.05, 300 iterations")
    
    # Soft Voting Ensemble
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("et", et), ("hgb", hgb)],
        voting="soft",
        n_jobs=-1,
        verbose=False
    )
    print("✓ Voting Ensemble: soft voting over RF + ET + HGB")
    
    return {
        "rf": rf,
        "et": et,
        "hgb": hgb,
        "voting": ensemble,
    }


# ============================================
# 3. CROSS-VALIDATION
# ============================================

def evaluate_models_cv(models: dict, X: pd.DataFrame, y: np.ndarray, 
                       groups: np.ndarray, n_splits: int = 5) -> dict:
    """
    Evaluate models using GroupKFold cross-validation.
    
    Args:
        models: Dictionary of models to evaluate
        X: Feature matrix
        y: Encoded labels
        groups: Group indices (sample_index) for GroupKFold
        n_splits: Number of CV folds
    
    Returns:
        Dictionary with CV results per model
    """
    print("\n" + "="*70)
    print(f"CROSS-VALIDATION (GroupKFold, {n_splits} splits)")
    print("="*70)
    
    gkf = GroupKFold(n_splits=n_splits)
    cv_scores = {}
    
    for name, model in models.items():
        print(f"\n📊 Model: {name.upper()}")
        print("-" * 70)
        
        cv = cross_validate(
            model,
            X,
            y,
            cv=gkf,
            groups=groups,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
            return_train_score=False,
            n_jobs=-1,
            verbose=0
        )
        cv_scores[name] = cv
        
        print(f"  Accuracy:      {cv['test_accuracy'].mean():.4f} "
              f"± {cv['test_accuracy'].std():.4f}")
        print(f"  F1-Macro:      {cv['test_f1_macro'].mean():.4f} "
              f"± {cv['test_f1_macro'].std():.4f}")
        print(f"  F1-Weighted:   {cv['test_f1_weighted'].mean():.4f} "
              f"± {cv['test_f1_weighted'].std():.4f}")
        
        # Show per-fold scores
        print(f"\n  Per-fold F1-Macro: ", end="")
        for i, score in enumerate(cv['test_f1_macro'], 1):
            print(f"Fold{i}={score:.4f}", end="  ")
        print()
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY: Model Comparison")
    print("="*70)
    print(f"{'Model':<20} {'Accuracy':<15} {'F1-Macro':<15} {'F1-Weighted':<15}")
    print("-" * 70)
    
    for name, cv in cv_scores.items():
        acc = cv['test_accuracy'].mean()
        f1_macro = cv['test_f1_macro'].mean()
        f1_weighted = cv['test_f1_weighted'].mean()
        print(f"{name:<20} {acc:<15.4f} {f1_macro:<15.4f} {f1_weighted:<15.4f}")
    
    return cv_scores


# ============================================
# 4. FINAL MODEL TRAINING
# ============================================

def train_final_model(model, X: pd.DataFrame, y: np.ndarray, 
                     label_encoder: LabelEncoder) -> None:
    """
    Train final model on full dataset with balanced sample weights.
    
    Args:
        model: Model to train
        X: Feature matrix
        y: Encoded labels
        label_encoder: Label encoder for class names
    """
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL ON FULL DATASET")
    print("="*70)
    
    # Compute balanced sample weights
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    
    print(f"Training samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(label_encoder.classes_)}")
    print(f"Sample weights computed (balanced)")
    
    # Fit model
    print("\n🔄 Training in progress...")
    model.fit(X, y, sample_weight=sample_weight)
    print("✅ Training complete!")
    
    # In-sample evaluation (sanity check)
    print("\n" + "="*70)
    print("IN-SAMPLE EVALUATION (Sanity Check)")
    print("="*70)
    
    y_pred = model.predict(X)
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=label_encoder.classes_))
    
    print("\nConfusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    
    # Pretty print confusion matrix
    print("\nConfusion Matrix (labeled):")
    print(f"{'True \\ Pred':<15}", end='')
    for class_name in label_encoder.classes_:
        print(f"{class_name:<15}", end='')
    print()
    print("-" * (15 * (len(label_encoder.classes_) + 1)))
    
    for i, true_class in enumerate(label_encoder.classes_):
        print(f"{true_class:<15}", end='')
        for j in range(len(label_encoder.classes_)):
            print(f"{cm[i, j]:<15}", end='')
        print()
    
    # Overall metrics
    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_weighted = f1_score(y, y_pred, average='weighted')
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1-Macro:    {f1_macro:.4f}")
    print(f"  F1-Weighted: {f1_weighted:.4f}")


# ============================================
# 5. SAVE MODEL
# ============================================

def save_model(model, label_encoder: LabelEncoder, feature_columns: list,
               output_path: str = "models/pirate_pain_tree_ensemble.pkl") -> None:
    """
    Save trained model with metadata.
    
    Args:
        model: Trained model
        label_encoder: Label encoder
        feature_columns: List of feature column names
        output_path: Path to save model
    """
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Package model with metadata
    model_package = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "model_type": type(model).__name__,
        "n_features": len(feature_columns),
        "classes": label_encoder.classes_.tolist(),
    }
    
    joblib.dump(model_package, output_path)
    
    print(f"✅ Model saved to: {output_path}")
    print(f"   Model type: {model_package['model_type']}")
    print(f"   Features: {model_package['n_features']}")
    print(f"   Classes: {model_package['classes']}")


# ============================================
# 6. MAIN PIPELINE
# ============================================

def main(data_dir: str = "data", output_path: str = "models/pirate_pain_tree_ensemble.pkl",
         model_name: str = "voting", n_splits: int = 5, random_state: int = 42):
    """
    Main training pipeline.
    
    Args:
        data_dir: Directory containing training data
        output_path: Path to save trained model
        model_name: Which model to train ('rf', 'et', 'hgb', or 'voting')
        n_splits: Number of CV folds
        random_state: Random seed
    """
    print("\n" + "="*70)
    print("🏴‍☠️ PIRATE PAIN CLASSIFICATION - TREE ENSEMBLE PIPELINE 🏴‍☠️")
    print("="*70)
    
    # --- Load data ---
    print("\n📂 Loading data...")
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "pirate_pain_train.csv")
    labels_df = pd.read_csv(data_dir / "pirate_pain_train_labels.csv")
    
    print(f"✓ Loaded {len(train_df)} rows from training data")
    print(f"✓ Loaded {len(labels_df)} labels")
    print(f"✓ Unique samples: {train_df['sample_index'].nunique()}")
    
    # Check class distribution
    print("\n📊 Class Distribution:")
    print(labels_df['label'].value_counts().sort_index())
    
    # --- Feature engineering ---
    X = build_feature_table(train_df, verbose=True)
    
    # Align labels with feature matrix
    y = labels_df.set_index("sample_index").loc[X.index, "label"]
    
    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    print(f"\n✓ Labels encoded: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    # Groups for GroupKFold
    groups = X.index.values
    
    # --- Define models ---
    models = make_models(random_state=random_state)
    
    # --- Cross-validation ---
    cv_scores = evaluate_models_cv(models, X, y_enc, groups, n_splits=n_splits)
    
    # --- Train final model ---
    if model_name not in models:
        print(f"\n❌ Error: Model '{model_name}' not found!")
        print(f"   Available models: {list(models.keys())}")
        return
    
    print(f"\n🎯 Selected model for final training: {model_name.upper()}")
    final_model = models[model_name]
    
    train_final_model(final_model, X, y_enc, le)
    
    # --- Save model ---
    save_model(final_model, le, X.columns.tolist(), output_path)
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Use the saved model at: {output_path}")
    print(f"2. Run inference with: python predict_ensemble.py")
    print(f"3. Compare with neural network models")


# ============================================
# 7. COMMAND-LINE INTERFACE
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train tree ensemble models for Pirate Pain Classification"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing training data (default: data)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/pirate_pain_tree_ensemble.pkl",
        help="Path to save trained model (default: models/pirate_pain_tree_ensemble.pkl)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="voting",
        choices=["rf", "et", "hgb", "voting"],
        help="Model to train: rf=RandomForest, et=ExtraTrees, hgb=HistGradientBoosting, voting=Ensemble (default: voting)"
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        output_path=args.output,
        model_name=args.model,
        n_splits=args.cv_splits,
        random_state=args.random_state
    )
