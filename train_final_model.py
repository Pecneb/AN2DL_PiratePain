"""
Train Final Model using Best Hyperparameters from Optuna
Trains on full training dataset and saves the best model for inference.

Usage:
    python train_final_model.py --config ./lightning_logs_tuning/best_config_optuna.json
"""

import os
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Import the model from hyperparameter_tuning script
from hyperparameter_tuning import RecurrentClassifierPL, parse_body_parts, identify_pirate, SEED

# Set seeds
pl.seed_everything(SEED, workers=True)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_and_preprocess_data(
    train_path: str = './data/pirate_pain_train.csv',
    labels_path: str = './data/pirate_pain_train_labels.csv',
    scaling: str = 'extra'
):
    """
    Load and preprocess the full training dataset.
    
    Args:
        train_path: Path to training data CSV
        labels_path: Path to labels CSV
        scaling: Scaling method ('inter', 'extra', or 'hybrid')
    
    Returns:
        X: Sequences array (n_samples, seq_len, n_features)
        y: Labels array (n_samples,)
        scaler: Fitted scaler object
    """
    print("\n" + "="*80)
    print("LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    # Load data
    train_df = pd.read_csv(train_path)
    labels_df = pd.read_csv(labels_path)
    
    print(f"✓ Loaded {len(labels_df)} samples")
    
    # Parse body parts
    # for col in ['n_legs', 'n_eyes', 'n_hands']:
    #     if col in train_df.columns:
    #         train_df[col] = train_df[col].apply(parse_body_parts)
    train_df = parse_body_parts(train_df)
    
    # Merge labels
    train_df = train_df.merge(labels_df, on='sample_index', how='left')
    
    # Encode labels
    label_map = {'no_pain': 0, 'low_pain': 1, 'high_pain': 2}
    train_df['label'] = train_df['label'].map(label_map)
    
    # Get unique samples
    sample_indices = train_df['sample_index'].unique()
    
    # Features to use
    joint_cols = [col for col in train_df.columns if col.startswith('joint_')]
    bodypart_cols = ['n_legs', 'n_eyes', 'n_hands']
    feature_cols = joint_cols + bodypart_cols
    
    print(f"✓ Using {len(feature_cols)} features ({len(joint_cols)} joints + {len(bodypart_cols)} body parts)")
    
    # Build sequences
    sequences = []
    labels = []
    
    for sample_idx in sample_indices:
        sample_data = train_df[train_df['sample_index'] == sample_idx]
        
        # Extract sequence (drop sample_index, time, pain_level, label)
        seq = sample_data[feature_cols].values
        label = sample_data['label'].iloc[0]
        
        sequences.append(seq)
        labels.append(label)
    
    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    
    print(f"✓ Sequences shape: {X.shape}")
    print(f"✓ Labels shape: {y.shape}")
    
    # Scale data
    print(f"\nScaling method: {scaling}")
    
    if scaling == 'inter':
        # Scale each sample independently
        for i in range(len(X)):
            scaler = StandardScaler()
            X[i] = scaler.fit_transform(X[i])
        scaler = None  # No global scaler
        
    elif scaling == 'extra':
        # Global scaling across all samples
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        scaler = StandardScaler()
        X_flat = scaler.fit_transform(X_flat)
        X = X_flat.reshape(n_samples, seq_len, n_features)
        
    elif scaling == 'hybrid':
        # Scale joints globally, body parts per-sample
        n_samples, seq_len, n_features = X.shape
        n_joint_features = len(joint_cols)
        
        # Scale joints globally
        X_joints = X[:, :, :n_joint_features].reshape(-1, n_joint_features)
        scaler = StandardScaler()
        X_joints = scaler.fit_transform(X_joints)
        X[:, :, :n_joint_features] = X_joints.reshape(n_samples, seq_len, n_joint_features)
        
        # Scale body parts per-sample
        for i in range(n_samples):
            body_scaler = StandardScaler()
            X[i, :, n_joint_features:] = body_scaler.fit_transform(X[i, :, n_joint_features:])
    else:
        raise ValueError(f"Unknown scaling method: {scaling}")
    
    print(f"✓ Data scaled")
    print("="*80 + "\n")
    
    return X, y, scaler


def train_final_model(
    config_path: str,
    max_epochs: int = 100,
    patience: int = 15,
    min_delta: int = 0,
    scaling: str = 'extra',
    output_dir: str = './final_model'
):
    """
    Train final model with best hyperparameters on full training set.
    
    Args:
        config_path: Path to best config JSON
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        scaling: Scaling method
        output_dir: Directory to save final model
    """
    # Load best config
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL")
    print("="*80)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("\nBest Hyperparameters:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # Load and preprocess data
    X, y, scaler = load_and_preprocess_data(scaling=scaling)
    
    # Get batch size from config (if available)
    batch_size = config.pop('batch_size', 32)
    
    # Create dataset and dataloader
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    print(f"✓ Created DataLoader with batch_size={batch_size}")
    
    # Create model
    input_size = X.shape[2]
    num_classes = len(np.unique(y))
    
    model = RecurrentClassifierPL(
        input_size=input_size,
        num_classes=num_classes,
        **config
    )
    
    print(f"✓ Created model: {config['rnn_type']} (hidden_size={config['hidden_size']}, num_layers={config['num_layers']})")
    print(f"  Bidirectional: {config['bidirectional']}")
    print(f"  Dropout: {config['dropout_rate']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Weight decay: {config['weight_decay']}")
    
    # Setup callbacks
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='pirate_pain_best_model',
        monitor='train_loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        patience=patience,
        mode='min',
        verbose=True,
        min_delta=min_delta
    )
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name='final_model_logs'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=10
    )
    
    print(f"\n{'='*80}")
    print(f"Starting training for up to {max_epochs} epochs (patience={patience})")
    print(f"{'='*80}\n")
    
    # Train model
    trainer.fit(model, train_loader)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Best model saved to: {checkpoint_callback.best_model_path}")
    print(f"✅ Last model saved to: {checkpoint_callback.last_model_path}")
    print(f"✅ TensorBoard logs: {logger.log_dir}")
    
    # Save scaler if it exists
    if scaler is not None:
        import pickle
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ Scaler saved to: {scaler_path}")
    
    # Save training info
    info = {
        'config': config,
        'batch_size': batch_size,
        'scaling': scaling,
        'input_size': input_size,
        'num_classes': num_classes,
        'num_samples': len(X),
        'best_model_path': checkpoint_callback.best_model_path,
        'last_model_path': checkpoint_callback.last_model_path
    }
    
    info_path = os.path.join(output_dir, 'training_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"✅ Training info saved to: {info_path}")
    
    print(f"\n{'='*80}")
    print("View training progress with:")
    print(f"  tensorboard --logdir {output_dir}")
    print(f"{'='*80}\n")
    
    return checkpoint_callback.best_model_path


def main():
    parser = argparse.ArgumentParser(description='Train final model with best hyperparameters')
    parser.add_argument('--config', type=str, 
                       default='./lightning_logs_tuning/best_config_optuna.json',
                       help='Path to best config JSON file')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0,
                       help='Early stopping delta loss parameter.')
    parser.add_argument('--scaling', type=str, default='extra',
                       choices=['inter', 'extra', 'hybrid'],
                       help='Scaling method')
    parser.add_argument('--output_dir', type=str, default='./final_model',
                       help='Output directory for final model')
    
    args = parser.parse_args()
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        print("\nPlease run hyperparameter tuning first:")
        print("  python hyperparameter_tuning.py --method optuna --n_trials 50")
        return
    
    # Train model
    best_model_path = train_final_model(
        config_path=args.config,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        scaling=args.scaling,
        output_dir=args.output_dir
    )
    
    print("\n✅ Final model training complete!")
    print(f"Next step: Run inference with predict.py")


if __name__ == '__main__':
    main()
