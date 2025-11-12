"""
Generate Predictions using Trained Model
Loads the best trained model and generates predictions on test data.

Usage:
    python predict.py --model ./final_model/pirate_pain_best_model.ckpt --test_data ./data/pirate_pain_test.csv
"""

import os
import json
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import pytorch_lightning as pl

# Import the model from hyperparameter_tuning script
from hyperparameter_tuning import RecurrentClassifierPL, parse_body_parts, SEED

# Set seeds
pl.seed_everything(SEED, workers=True)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_and_preprocess_test_data(
    test_path: str,
    scaler_path: str = None,
    scaling: str = 'extra'
):
    """
    Load and preprocess test data.
    
    Args:
        test_path: Path to test data CSV
        scaler_path: Path to saved scaler (for 'extra' and 'hybrid' scaling)
        scaling: Scaling method used during training
    
    Returns:
        X: Sequences array (n_samples, seq_len, n_features)
        sample_indices: Original sample indices for submission
    """
    print("\n" + "="*80)
    print("LOADING AND PREPROCESSING TEST DATA")
    print("="*80)
    
    # Load data
    test_df = pd.read_csv(test_path)
    print(f"✓ Loaded test data: {test_path}")
    
    # Parse body parts
    # for col in ['n_legs', 'n_eyes', 'n_hands']:
    #     if col in test_df.columns:
    #         test_df[col] = test_df[col].apply(parse_body_parts)
    test_df = parse_body_parts(test_df)
    
    # Get unique samples
    sample_indices = test_df['sample_index'].unique()
    print(f"✓ Found {len(sample_indices)} test samples")
    
    # Features to use
    joint_cols = [col for col in test_df.columns if col.startswith('joint_')]
    bodypart_cols = ['n_legs', 'n_eyes', 'n_hands']
    feature_cols = joint_cols + bodypart_cols
    
    print(f"✓ Using {len(feature_cols)} features")
    
    # Build sequences
    sequences = []
    indices = []
    
    for sample_idx in sample_indices:
        sample_data = test_df[test_df['sample_index'] == sample_idx]
        
        # Extract sequence
        seq = sample_data[feature_cols].values
        
        sequences.append(seq)
        indices.append(sample_idx)
    
    X = np.array(sequences, dtype=np.float32)
    
    print(f"✓ Test sequences shape: {X.shape}")
    
    # Scale data
    print(f"\nScaling method: {scaling}")
    
    if scaling == 'inter':
        # Scale each sample independently (no saved scaler needed)
        for i in range(len(X)):
            scaler = StandardScaler()
            X[i] = scaler.fit_transform(X[i])
        print("✓ Applied inter-sample scaling")
        
    elif scaling == 'extra':
        # Load and apply global scaler
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✓ Loaded scaler from: {scaler_path}")
            
            n_samples, seq_len, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
            X_flat = scaler.transform(X_flat)
            X = X_flat.reshape(n_samples, seq_len, n_features)
            print("✓ Applied global scaling")
        else:
            print(f"⚠️  Warning: Scaler not found at {scaler_path}")
            print("   Fitting new scaler on test data (not recommended)")
            n_samples, seq_len, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
            scaler = StandardScaler()
            X_flat = scaler.fit_transform(X_flat)
            X = X_flat.reshape(n_samples, seq_len, n_features)
        
    elif scaling == 'hybrid':
        # Load global scaler for joints, per-sample for body parts
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✓ Loaded scaler from: {scaler_path}")
            
            n_samples, seq_len, n_features = X.shape
            n_joint_features = len(joint_cols)
            
            # Scale joints with loaded scaler
            X_joints = X[:, :, :n_joint_features].reshape(-1, n_joint_features)
            X_joints = scaler.transform(X_joints)
            X[:, :, :n_joint_features] = X_joints.reshape(n_samples, seq_len, n_joint_features)
            
            # Scale body parts per-sample
            for i in range(n_samples):
                body_scaler = StandardScaler()
                X[i, :, n_joint_features:] = body_scaler.fit_transform(X[i, :, n_joint_features:])
            
            print("✓ Applied hybrid scaling")
        else:
            print(f"⚠️  Warning: Scaler not found at {scaler_path}")
            print("   Proceeding without proper scaling")
    
    print("="*80 + "\n")
    
    return X, np.array(indices)


def predict(
    model_path: str,
    test_path: str,
    output_path: str = './submission.csv',
    batch_size: int = 32,
    info_path: str = None
):
    """
    Generate predictions on test data.
    
    Args:
        model_path: Path to trained model checkpoint
        test_path: Path to test data CSV
        output_path: Path to save submission CSV
        batch_size: Batch size for inference
        info_path: Path to training_info.json (for scaling method)
    """
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    
    # Load training info if available
    scaling = 'extra'  # Default
    scaler_path = None
    
    if info_path and os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        scaling = info.get('scaling', 'extra')
        scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')
        print(f"✓ Loaded training info from: {info_path}")
        print(f"  Scaling method: {scaling}")
    else:
        print(f"⚠️  Training info not found, using default scaling: {scaling}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = RecurrentClassifierPL.load_from_checkpoint(model_path)
    model.eval()
    
    # Move model to device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)
    print(f"✓ Model loaded on device: {device}")
    
    # Load and preprocess test data
    X_test, sample_indices = load_and_preprocess_test_data(
        test_path=test_path,
        scaler_path=scaler_path,
        scaling=scaling
    )
    
    # Create dataloader
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Making predictions on {len(X_test)} samples...")
    
    # Generate predictions
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            X_batch = batch[0].to(device)
            logits = model(X_batch)
            
            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities)
    
    print(f"✓ Generated {len(predictions)} predictions")
    
    # Map predictions to labels
    label_map = {0: 'no_pain', 1: 'low_pain', 2: 'high_pain'}
    pain_levels = [label_map[p] for p in predictions]
    
    # Show distribution
    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPrediction distribution:")
    for idx, count in zip(unique, counts):
        print(f"  {label_map[idx]}: {count} ({count/len(predictions)*100:.1f}%)")
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'sample_index': sample_indices,
        'pain_level': pain_levels
    })
    
    # Save submission
    submission.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Submission saved to: {output_path}")
    print(f"{'='*80}\n")
    
    # Also save probabilities (optional, for analysis)
    prob_path = output_path.replace('.csv', '_probabilities.csv')
    prob_df = pd.DataFrame(
        probabilities,
        columns=['prob_no_pain', 'prob_low_pain', 'prob_high_pain']
    )
    prob_df.insert(0, 'sample_index', sample_indices)
    prob_df.to_csv(prob_path, index=False)
    print(f"✅ Prediction probabilities saved to: {prob_path}")
    
    # Show first few predictions
    print("\nFirst 10 predictions:")
    print(submission.head(10).to_string(index=False))
    
    return submission


def main():
    parser = argparse.ArgumentParser(description='Generate predictions on test data')
    parser.add_argument('--model', type=str,
                       default='./final_model/pirate_pain_best_model.ckpt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_data', type=str,
                       default='./data/pirate_pain_test.csv',
                       help='Path to test data CSV')
    parser.add_argument('--output', type=str,
                       default='./submission.csv',
                       help='Path to save submission CSV')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--info', type=str,
                       default='./final_model/training_info.json',
                       help='Path to training info JSON')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("\nPlease train the final model first:")
        print("  python train_final_model.py")
        return
    
    # Check if test data exists
    if not os.path.exists(args.test_data):
        print(f"❌ Test data not found: {args.test_data}")
        return
    
    # Generate predictions
    submission = predict(
        model_path=args.model,
        test_path=args.test_data,
        output_path=args.output,
        batch_size=args.batch_size,
        info_path=args.info
    )
    
    print("\n✅ Prediction complete!")
    print(f"Submit {args.output} to the competition 🚀")


if __name__ == '__main__':
    main()
