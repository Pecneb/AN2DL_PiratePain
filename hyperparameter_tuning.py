"""
Hyperparameter Tuning Script for Pirate Pain Time Series Classification
Uses PyTorch Lightning with K-Fold Cross-Validation and Grid Search

Usage:
    python hyperparameter_tuning.py --method grid --k_folds 5 --epochs 50
    python hyperparameter_tuning.py --method optuna --k_folds 3 --n_trials 50
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, List
from itertools import product
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import joblib
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Install with: pip install optuna")

# Set seeds for reproducibility
SEED = 42
pl.seed_everything(SEED, workers=True)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================================
# Custom Loss Functions
# ============================================================================


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """

    def __init__(self, alpha=1.0, gamma=2.0, num_classes=3, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - raw logits
            targets: (batch_size,) - class indices
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # Probability of true class
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# Data Processing Functions
# ============================================================================


def parse_body_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Parse body part columns from text to numeric."""
    _df = df.copy()
    body_part_map = {"two": 2, "one": 1}

    def body_part_parser_helper(x: str):
        if isinstance(x, str):
            parts = x.lower().split("+")
            num = parts[0]
            return body_part_map.get(num, 1)
        return x

    for col in _df.columns:
        if col.startswith("n_"):
            _df[col] = _df[col].apply(body_part_parser_helper)

    return _df


def identify_pirate(df: pd.DataFrame) -> pd.DataFrame:
    """Identify if the row contains pirate or regular person data."""
    _df = df.copy()
    body_part_map = {"two": 2, "one": 1}

    def body_part_parser_helper(x: str):
        if isinstance(x, str):
            parts = x.lower().split("+")
            num = parts[0]
            return body_part_map.get(num, 1)
        return x

    def pirate_identify_helper(row):
        n_bodypart = (
            body_part_parser_helper(row["n_legs"])
            + body_part_parser_helper(row["n_eyes"])
            + body_part_parser_helper(row["n_hands"])
        )
        if n_bodypart < 6:
            return 1
        return 0

    _df["is_pirate"] = _df.apply(pirate_identify_helper, axis=1)

    _df = _df.drop(columns=["n_legs", "n_eyes", "n_hands"])

    return _df


def one_hot_encode_column(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    One-hot encode given column(s) in-place (returns a new DataFrame).
    Keeps original column order and inserts the new dummy columns where
    the original column was located.
    """
    _df = df.copy()
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        if col not in _df.columns:
            continue

        # Create dummies from the column (cast to str to handle numeric categories)
        dummies = pd.get_dummies(_df[col].astype(str), prefix=col)

        # Find position of original column to preserve ordering
        orig_cols = list(_df.columns)
        idx = orig_cols.index(col)

        # Split dataframe into left (before col) and right (after col)
        left_df = _df.iloc[:, :idx]
        right_df = _df.iloc[:, idx + 1 :]

        # Concatenate left, dummies, right
        _df = pd.concat(
            [
                left_df.reset_index(drop=True),
                dummies.reset_index(drop=True),
                right_df.reset_index(drop=True),
            ],
            axis=1,
        )

    return _df


def scale_dataset(
    df: pd.DataFrame, method: str = "extra"
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale the dataset using different methods.

    Args:
        df: DataFrame to scale
        method: 'inter' (per-sample), 'extra' (global), or 'hybrid'
    """
    _df = df.copy()
    metadata_cols = ["sample_index", "time"]
    feature_cols = [
        col
        for col in _df.columns
        if col not in metadata_cols and col.startswith("joint_")
    ]

    if method == "inter":
        for sample_idx in _df["sample_index"].unique():
            mask = _df["sample_index"] == sample_idx
            scaler = StandardScaler()
            _df.loc[mask, feature_cols] = scaler.fit_transform(
                _df.loc[mask, feature_cols]
            )

    elif method == "extra":
        scaler = StandardScaler()
        _df[feature_cols] = scaler.fit_transform(_df[feature_cols])

    elif method == "hybrid":
        # First inter-sample
        for sample_idx in _df["sample_index"].unique():
            mask = _df["sample_index"] == sample_idx
            scaler = StandardScaler()
            _df.loc[mask, feature_cols] = scaler.fit_transform(
                _df.loc[mask, feature_cols]
            )
        # Then extra-sample
        scaler = StandardScaler()
        _df[feature_cols] = scaler.fit_transform(_df[feature_cols])

    return _df, scaler


def encode_labels(label_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Encode string labels to integers and return mapping."""
    _label_df = label_df.copy()
    labels = sorted(label_df["label"].unique())
    label_encoding = {label: i for i, label in enumerate(labels)}
    _label_df["label"] = _label_df["label"].map(label_encoding)
    return _label_df, label_encoding


def build_sequences(
    df: pd.DataFrame, labels_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sequences from dataframe, dropping metadata columns."""
    dataset = []
    labels = []

    for idx in df["sample_index"].unique():
        person_ds = (
            df[df["sample_index"] == idx]
            .drop(columns=["sample_index", "time"])
            .to_numpy()
        )
        person_label = labels_df[labels_df["sample_index"] == idx]["label"].values[0]
        dataset.append(person_ds)
        labels.append(person_label)

    return np.stack(dataset), np.array(labels)


# ============================================================================
# PyTorch Lightning Module
# ============================================================================


class RecurrentClassifierPL(pl.LightningModule):
    """PyTorch Lightning module for RNN-based time series classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        rnn_type: str = "GRU",
        bidirectional: bool = False,
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer_name: str = "Adam",
        criterion_name: str = "CrossEntropy",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.criterion_name = criterion_name
        self.num_classes = num_classes

        # Select RNN module
        rnn_map = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}
        rnn_module = rnn_map[rnn_type]

        # Dropout only applied between layers if num_layers > 1
        dropout_val = dropout_rate if num_layers > 1 else 0

        # Create recurrent layer
        self.rnn = rnn_module(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_val,
        )

        # Classifier input size
        classifier_input_size = hidden_size * 2 if bidirectional else hidden_size

        # Final classification layer
        self.classifier = nn.Linear(classifier_input_size, num_classes)

        # Loss function
        self.criterion = self._get_criterion(criterion_name, num_classes)

        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def _get_criterion(self, criterion_name: str, num_classes: int):
        """Get loss function by name."""
        if criterion_name == "CrossEntropy":
            return nn.CrossEntropyLoss()
        elif criterion_name == "CrossEntropyWeighted":
            # Automatically compute class weights (inverse frequency)
            # This is a placeholder - actual weights computed during training
            return nn.CrossEntropyLoss()
        elif criterion_name == "FocalLoss":
            # Focal Loss for imbalanced classes
            return FocalLoss(alpha=1.0, gamma=2.0, num_classes=num_classes)
        elif criterion_name == "LabelSmoothing":
            # CrossEntropy with label smoothing
            return nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            raise ValueError(f"Unknown criterion: {criterion_name}")

    def forward(self, x):
        """Forward pass through the network."""
        # x shape: (batch, seq_len, features)
        rnn_out, hidden = self.rnn(x)

        # Extract final hidden state
        if self.rnn_type == "LSTM":
            hidden = hidden[0]  # (h_n, c_n) -> h_n

        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        if self.bidirectional:
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
            hidden_to_classify = torch.cat(
                [hidden[-1, 0, :, :], hidden[-1, 1, :, :]], dim=1
            )
        else:
            hidden_to_classify = hidden[-1]

        # Classification
        logits = self.classifier(hidden_to_classify)
        return logits

    def training_step(self, batch, batch_idx):
        """Training step."""
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        self.training_step_outputs.append(
            {
                "loss": loss.detach(),
                "preds": preds.detach(),
                "targets": targets.detach(),
            }
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.append(
            {
                "loss": loss.detach(),
                "preds": preds.detach(),
                "targets": targets.detach(),
            }
        )

        return loss

    def on_train_epoch_end(self):
        """Compute epoch-level metrics for training."""
        if len(self.training_step_outputs) > 0:
            all_preds = (
                torch.cat([x["preds"] for x in self.training_step_outputs])
                .cpu()
                .numpy()
            )
            all_targets = (
                torch.cat([x["targets"] for x in self.training_step_outputs])
                .cpu()
                .numpy()
            )

            f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
            self.log("train_f1", f1, prog_bar=True)

            self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        """Compute epoch-level metrics for validation."""
        if len(self.validation_step_outputs) > 0:
            all_preds = (
                torch.cat([x["preds"] for x in self.validation_step_outputs])
                .cpu()
                .numpy()
            )
            all_targets = (
                torch.cat([x["targets"] for x in self.validation_step_outputs])
                .cpu()
                .numpy()
            )

            f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
            self.log("val_f1", f1, prog_bar=True)

            self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer."""
        if self.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                nesterov=True,
            )
        elif self.optimizer_name == "RMSprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        elif self.optimizer_name == "AdaGrad":
            optimizer = torch.optim.Adagrad(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        return optimizer


# ============================================================================
# K-Fold Cross-Validation Function
# ============================================================================


def run_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    config: Dict,
    k_folds: int = 5,
    epochs: int = 50,
    patience: int = 10,
    batch_size: int = 32,
    experiment_name: str = "default",
    save_dir: str = "./lightning_logs",
    verbose: bool = True,
) -> Dict:
    """
    Run K-fold cross-validation for a given hyperparameter configuration.

    Args:
        X: Input sequences (num_samples, seq_len, features)
        y: Labels (num_samples,)
        config: Hyperparameter configuration dict
        k_folds: Number of folds
        epochs: Max epochs per fold
        patience: Early stopping patience
        batch_size: Batch size
        experiment_name: Name for logging
        save_dir: Directory for logs and models
        verbose: Print progress

    Returns:
        Dictionary with results including mean and std of validation metrics
    """
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Fold {fold + 1}/{k_folds}")
            print(f"{'='*60}")

        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_fold), torch.LongTensor(y_train_fold)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_fold), torch.LongTensor(y_val_fold)
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        # Create model
        model = RecurrentClassifierPL(
            input_size=X.shape[2], num_classes=len(np.unique(y)), **config
        )

        # Callbacks
        early_stop = EarlyStopping(
            monitor="val_f1", patience=patience, mode="max", verbose=verbose
        )

        checkpoint = ModelCheckpoint(
            monitor="val_f1",
            mode="max",
            save_top_k=1,
            dirpath=os.path.join(save_dir, "checkpoints", experiment_name),
            filename=f"fold_{fold}_{{epoch}}_{{val_f1:.3f}}",
        )

        # Logger
        logger = TensorBoardLogger(
            save_dir=save_dir, name=experiment_name, version=f"fold_{fold}"
        )

        # Trainer
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=[early_stop, checkpoint],
            logger=logger,
            deterministic=True,
            accelerator="auto",
            devices=1,
            enable_progress_bar=verbose,
            log_every_n_steps=10,
        )

        # Train
        trainer.fit(model, train_loader, val_loader)

        # Validate
        val_results = trainer.validate(model, val_loader, verbose=False)

        fold_result = {
            "fold": fold,
            "val_loss": val_results[0]["val_loss"],
            "val_acc": val_results[0]["val_acc"],
            "val_f1": val_results[0]["val_f1"],
            "best_model_path": checkpoint.best_model_path,
        }

        fold_results.append(fold_result)

        if verbose:
            print(f"Fold {fold + 1} Results:")
            print(f"  Val Loss: {fold_result['val_loss']:.4f}")
            print(f"  Val Acc:  {fold_result['val_acc']:.4f}")
            print(f"  Val F1:   {fold_result['val_f1']:.4f}")

    # Aggregate results
    val_f1_scores = [r["val_f1"] for r in fold_results]
    val_acc_scores = [r["val_acc"] for r in fold_results]

    results = {
        "config": config,
        "fold_results": fold_results,
        "mean_val_f1": np.mean(val_f1_scores),
        "std_val_f1": np.std(val_f1_scores),
        "mean_val_acc": np.mean(val_acc_scores),
        "std_val_acc": np.std(val_acc_scores),
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"K-Fold CV Results for {experiment_name}")
        print(f"{'='*60}")
        print(
            f"Mean Val F1:  {results['mean_val_f1']:.4f} ± {results['std_val_f1']:.4f}"
        )
        print(
            f"Mean Val Acc: {results['mean_val_acc']:.4f} ± {results['std_val_acc']:.4f}"
        )
        print(f"{'='*60}\n")

    return results


# ============================================================================
# Grid Search Function
# ============================================================================


def grid_search(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict,
    k_folds: int = 5,
    epochs: int = 50,
    patience: int = 10,
    save_dir: str = "./lightning_logs",
    results_file: str = "./grid_search_results.csv",
) -> Tuple[Dict, List[Dict]]:
    """
    Perform grid search with k-fold cross-validation.

    Args:
        X: Input sequences
        y: Labels
        param_grid: Dictionary of parameter lists to search
        k_folds: Number of folds
        epochs: Max epochs
        patience: Early stopping patience
        save_dir: Save directory
        results_file: CSV file to save results

    Returns:
        Tuple of (best_config, all_results)
    """
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    print(f"\n{'#'*80}")
    print(f"GRID SEARCH")
    print(f"{'#'*80}")
    print(f"Total configurations: {len(combinations)}")
    print(f"K-folds per config: {k_folds}")
    print(f"Total training runs: {len(combinations) * k_folds}")
    print(f"{'#'*80}\n")

    all_results = []
    best_score = -np.inf
    best_config = None

    for idx, combo in enumerate(combinations, 1):
        config = dict(zip(param_names, combo))
        config_str = "_".join([f"{k}_{v}" for k, v in config.items()])

        print(f"\n{'='*80}")
        print(f"Configuration {idx}/{len(combinations)}: {config_str}")
        print(f"{'='*80}")
        for k, v in config.items():
            print(f"  {k}: {v}")

        # Extract batch_size if it's in the grid
        batch_size = config.pop("batch_size", 32)

        # Run k-fold CV
        results = run_kfold_cv(
            X=X,
            y=y,
            config=config,
            k_folds=k_folds,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            experiment_name=config_str,
            save_dir=save_dir,
            verbose=True,
        )

        # Add batch_size back for saving
        results["config"]["batch_size"] = batch_size
        all_results.append(results)

        # Track best
        if results["mean_val_f1"] > best_score:
            best_score = results["mean_val_f1"]
            best_config = results["config"].copy()
            print(f"\n⭐ NEW BEST CONFIGURATION! ⭐")
            print(f"Mean Val F1: {best_score:.4f}")

    # Save results to CSV
    results_df = pd.DataFrame(
        [
            {
                **r["config"],
                "mean_val_f1": r["mean_val_f1"],
                "std_val_f1": r["std_val_f1"],
                "mean_val_acc": r["mean_val_acc"],
                "std_val_acc": r["std_val_acc"],
            }
            for r in all_results
        ]
    )
    results_df = results_df.sort_values("mean_val_f1", ascending=False)
    results_df.to_csv(results_file, index=False)
    print(f"\n✅ Results saved to {results_file}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Best Configuration:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    print(f"Best Mean Val F1: {best_score:.4f}")
    print(f"{'='*80}\n")

    return best_config, all_results


# ============================================================================
# Optuna Optimization Function
# ============================================================================


def optuna_optimization(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    k_folds: int = 5,
    epochs: int = 50,
    patience: int = 10,
    save_dir: str = "./lightning_logs",
    results_file: str = "./optuna_results.csv",
    study_name: str = "pirate_pain_optuna",
) -> Tuple[Dict, optuna.Study]:
    """
    Perform Optuna-based hyperparameter optimization with k-fold CV.

    Args:
        X: Input sequences
        y: Labels
        n_trials: Number of Optuna trials
        k_folds: Number of folds
        epochs: Max epochs
        patience: Early stopping patience
        save_dir: Save directory
        results_file: CSV file to save results
        study_name: Name for Optuna study

    Returns:
        Tuple of (best_config, study)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not installed. Install with: pip install optuna")

    print(f"\n{'#'*80}")
    print(f"OPTUNA HYPERPARAMETER OPTIMIZATION")
    print(f"{'#'*80}")
    print(f"Number of trials: {n_trials}")
    print(f"K-folds per trial: {k_folds}")
    print(f"Total training runs: {n_trials * k_folds}")
    print(f"{'#'*80}\n")

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""

        # Suggest hyperparameters
        config = {
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "rnn_type": trial.suggest_categorical("rnn_type", ["LSTM", "GRU"]),
            "bidirectional": trial.suggest_categorical("bidirectional", [False, True]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-3),
            "optimizer_name": trial.suggest_categorical(
                "optimizer_name", ["Adam", "AdamW", "SGD", "RMSprop"]
            ),
            "criterion_name": trial.suggest_categorical(
                "criterion_name", ["CrossEntropy", "FocalLoss", "LabelSmoothing"]
            ),
        }

        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        # Create trial name
        trial_name = f"trial_{trial.number}"

        print(f"\n{'='*80}")
        print(f"Trial {trial.number + 1}/{n_trials}")
        print(f"{'='*80}")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print(f"  batch_size: {batch_size}")

        # Run k-fold CV
        try:
            results = run_kfold_cv(
                X=X,
                y=y,
                config=config,
                k_folds=k_folds,
                epochs=epochs,
                patience=patience,
                batch_size=batch_size,
                experiment_name=trial_name,
                save_dir=save_dir,
                verbose=False,  # Less verbose for Optuna
            )

            mean_val_f1 = results["mean_val_f1"]

            print(f"Trial {trial.number + 1} Result: Val F1 = {mean_val_f1:.4f}")

            return mean_val_f1

        except Exception as e:
            print(f"Trial {trial.number + 1} failed with error: {e}")
            return 0.0  # Return poor score for failed trials

    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    # Run optimization
    study.optimize(
        objective, n_trials=n_trials, show_progress_bar=True, catch=(Exception,)
    )

    # Get best parameters
    best_config = study.best_params
    best_score = study.best_value

    print(f"\n{'='*80}")
    print(f"OPTUNA OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Val F1: {best_score:.4f}")
    print(f"\nBest Hyperparameters:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    print(f"{'='*80}\n")

    # Save results
    trials_df = study.trials_dataframe()
    trials_df.to_csv(results_file, index=False)
    print(f"✅ Optuna trials saved to {results_file}")

    # Save best config
    best_config_file = os.path.join(save_dir, "best_config_optuna.json")
    os.makedirs(save_dir, exist_ok=True)
    with open(best_config_file, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"✅ Best config saved to {best_config_file}")

    # Plot optimization history
    try:
        import plotly

        # Optimization history
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_html(os.path.join(save_dir, "optimization_history.html"))

        # Parameter importances
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_html(os.path.join(save_dir, "param_importances.html"))

        # Parallel coordinate plot
        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig3.write_html(os.path.join(save_dir, "parallel_coordinate.html"))

        print(f"✅ Optuna visualizations saved to {save_dir}/")
    except ImportError:
        print("⚠️  Install plotly for visualizations: pip install plotly")

    return best_config, study


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning for Pirate Pain Classification"
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument(
        "--method",
        type=str,
        default="grid",
        choices=["grid", "optuna"],
        help="Search method",
    )
    parser.add_argument("--k_folds", type=int, default=5, help="Number of k-folds")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs per fold")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--scaling",
        type=str,
        default="extra",
        choices=["inter", "extra", "hybrid"],
        help="Scaling method",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./lightning_logs_tuning", help="Save directory"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="./hyperparameter_search_results.csv",
        help="Results CSV file",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna trials (only for --method optuna)",
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING - Pirate Pain Classification")
    print(f"{'='*80}")
    print(f"Method: {args.method}")
    print(f"K-Folds: {args.k_folds}")
    print(f"Max Epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
    print(f"Scaling: {args.scaling}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading data...")
    train_path = os.path.join(args.data_dir, "pirate_pain_train.csv")
    labels_path = os.path.join(args.data_dir, "pirate_pain_train_labels.csv")

    df = pd.read_csv(train_path)
    labels_df = pd.read_csv(labels_path)
    print(f"✓ Loaded {len(df['sample_index'].unique())} samples")

    # Preprocess
    print("Preprocessing...")
    # df = parse_body_parts(df)
    df = identify_pirate(df)
    df, scaler = scale_dataset(df, method=args.scaling)
    labels_df_encoded, label_encoding = encode_labels(labels_df)

    # Save scaler to output directory
    

    # Build sequences
    print("Building sequences...")
    X, y = build_sequences(df, labels_df_encoded)
    print(f"✓ Sequences shape: {X.shape}")
    print(f"✓ Labels shape: {y.shape}")
    print(f"✓ Number of classes: {len(np.unique(y))}")

    # Define parameter grid (customize this!)
    param_grid = {
        "hidden_size": [64, 128],
        "num_layers": [1, 2],
        "rnn_type": ["LSTM", "GRU"],
        "bidirectional": [False, True],
        "dropout_rate": [0.2, 0.3],
        "learning_rate": [1e-3, 5e-4],
        "weight_decay": [0.0, 1e-5],
        "batch_size": [16, 32],
        "optimizer_name": ["Adam", "AdamW"],
        "criterion_name": ["CrossEntropy", "FocalLoss"],
    }

    if args.method == "grid":
        best_config, all_results = grid_search(
            X=X,
            y=y,
            param_grid=param_grid,
            k_folds=args.k_folds,
            epochs=args.epochs,
            patience=args.patience,
            save_dir=args.save_dir,
            results_file=args.results_file,
        )

        # Save best config
        best_config_file = os.path.join(args.save_dir, "best_config.json")
        with open(best_config_file, "w") as f:
            json.dump(best_config, f, indent=2)
        print(f"✅ Best config saved to {best_config_file}")

    elif args.method == "optuna":
        if not OPTUNA_AVAILABLE:
            print("❌ Optuna is not installed.")
            print("Install with: pip install optuna")
            sys.exit(1)

        best_config, study = optuna_optimization(
            X=X,
            y=y,
            n_trials=args.n_trials,
            k_folds=args.k_folds,
            epochs=args.epochs,
            patience=args.patience,
            save_dir=args.save_dir,
            results_file=args.results_file,
            study_name=f"pirate_pain_{args.scaling}",
        )

        # Print additional Optuna stats
        print(f"\n📊 Optuna Study Statistics:")
        print(f"  Total trials: {len(study.trials)}")
        print(
            f"  Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
        )
        print(
            f"  Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
        )
        print(
            f"  Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}"
        )

    print("\n✅ Hyperparameter tuning complete!")
    print(f"📊 View results in TensorBoard: tensorboard --logdir {args.save_dir}")


if __name__ == "__main__":
    main()
