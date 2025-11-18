# AN2DL_PiratePain

Time Series classification challenge

## 🏴‍☠️ The Pirate Pain Dataset

Ahoy, matey! This dataset contains multivariate time series data, captured from both ordinary folk and pirates over repeated observations in time. Each sample collects temporal dynamics of body joints and pain perception, with the goal of predicting the subject’s true pain status:

    no_pain
    low_pain
    high_pain

## ⚓ Files

    pirate_pain_train.csv — training set
    pirate_pain_train_labels.csv — labels for the training set
    pirate_pain_test.csv — test set (with no labels)
    sample_submission.csv — an example of random submission

## 🧭 Data Overview

Each record represents a time step within a subject’s recording, identified by sample_index and time. The dataset includes several groups of features:

    pain_survey_1–pain_survey_4 — simple rule-based sensor aggregations estimating perceived pain.
    n_legs, n_hands, n_eyes — subject characteristics.
    joint_00–joint_30 — continuous measurements of body joint angles (neck, elbow, knee, etc.) across time.

## 🏴‍☠️ Task

Predict the real pain level of each subject based on their time-series motion data.

---

## 🏆 Best Model: CNN-LSTM with Multi-Head Attention

### Architecture Overview

Our best performing model is a **CNN-LSTM Hybrid with Multi-Head Attention**, which combines multiple deep learning techniques to capture complex temporal patterns in the pirate pain data.

```
Input Features → Input Embedding → Positional Encoding → 
CNN Blocks (Local Feature Extraction) → 
Bidirectional LSTM (Temporal Modeling) → 
Multi-Head Attention (Focus Mechanism) → 
Residual Connection + Layer Normalization → 
Fully Connected Layers → Pain Classification
```

### Key Components

#### 1. **Input Embedding Layer**
- Projects raw features (joint angles + additional features) into a higher-dimensional embedding space
- Transforms input from 30+ features to embedding dimension (typically 64 or 128)
- Creates richer representations before CNN processing

#### 2. **Learnable Positional Embeddings**
- Adds temporal positional information to each timestep
- Unlike fixed sinusoidal encodings, these are learnable parameters
- Helps the model understand the sequence order and temporal relationships
- Shape: `(1, 160, embedding_dim)` for 160 timesteps

#### 3. **CNN Front-End (Local Feature Extraction)**
- 2-layer 1D Convolutional blocks for capturing local temporal patterns
- Each block contains:
  - Conv1D layer (kernel size 5-7)
  - Batch Normalization
  - ReLU activation
  - Dropout (0.4-0.5)
- Filters double with each layer (e.g., 64 → 128)
- Efficiently captures short-term motion patterns in joint movements

#### 4. **Bidirectional LSTM (Temporal Modeling)**
- 2-layer Bidirectional LSTM for modeling long-range temporal dependencies
- Forward pass captures past → present patterns
- Backward pass captures future → present patterns
- Hidden size: 64-128 units per direction
- Total output: `2 × hidden_size` (bidirectional concatenation)

#### 5. **Multi-Head Attention (Focus Mechanism)**
- **4-8 attention heads** to capture different temporal patterns simultaneously
- Self-attention mechanism over LSTM output sequence
- Allows model to focus on most relevant timesteps for pain classification
- Attention formula: `Attention(Q, K, V) = softmax(QK^T / √d_k)V`
- Can be toggled on/off via `use_attention` parameter

#### 6. **Residual Connection + Layer Normalization**
- Residual skip connection around attention layer
- Layer normalization for stable training
- Improves gradient flow and training stability

#### 7. **Classification Head**
- Fully connected layers for final classification
- Uses last timestep output from attention mechanism
- Architecture: `Linear(lstm_hidden × 2, lstm_hidden) → ReLU → Dropout → Linear(lstm_hidden, n_classes)`

### Feature Engineering

#### Features Used:
1. **Joint Features** (30 features): `joint_0` to `joint_29`
   - Continuous measurements of body joint angles
   - **Normalized** using StandardScaler
   - `joint_30` dropped (constant feature)

2. **Additional Features**:
   - `time`: Temporal timestep information
   - `is_pirate`: Binary derived feature (0 = not pirate, 1 = pirate)
     - Derived from: `is_pirate = 1` if `n_legs ≠ 2` OR `n_hands ≠ 2` OR `n_eyes ≠ 2`
     - Otherwise: `is_pirate = 0` (normal human)
   - `pain_survey_1` to `pain_survey_4`: Rule-based pain estimates
   - **NOT normalized** (kept in original scale for semantic meaning)

#### Normalization Strategy:
- **Selective normalization**: Only joint features are normalized
- Additional features (binary, categorical, time) remain unnormalized
- Preserves meaningful scales and relationships

### Hyperparameters

**Recommended configuration** (from grid search with 5-fold CV):

```python
{
    'cnn_filters': 64,           # Number of CNN filters in first layer
    'cnn_kernel': 5,             # CNN kernel size
    'cnn_layers': 2,             # Number of CNN blocks
    'lstm_hidden': 64-128,       # LSTM hidden size (per direction)
    'lstm_layers': 2,            # Number of LSTM layers
    'dropout': 0.4-0.5,          # Dropout rate
    'n_heads': 4,                # Number of attention heads
    'use_attention': True,       # Enable multi-head attention
    'lr': 1e-3 or 1e-4,         # Learning rate
    'batch_size': 16-32          # Batch size
}
```

### Training Details

- **Optimizer**: Adam with weight decay
- **Loss Function**: CrossEntropyLoss
- **Training Strategy**: 5-fold Stratified Cross-Validation
- **Early Stopping**: Patience of 20 epochs on validation F1 score
- **Grid Search**: 32 hyperparameter combinations tested
- **Final Training**: 150 epochs on full training dataset
- **Device**: Automatic detection (CUDA > MPS > CPU)

### Model Statistics

- **Total Parameters**: ~500K-1M (depending on configuration)
- **Input Shape**: `(batch_size, 160, n_features)`
- **Output Shape**: `(batch_size, 3)` - probabilities for [no_pain, low_pain, high_pain]
- **Sequence Length**: 160 timesteps
- **Feature Count**: 30 joint features + additional features

### Performance

The model benefits from:
- ✅ **Local pattern recognition** via CNN
- ✅ **Long-range temporal modeling** via Bidirectional LSTM
- ✅ **Adaptive focus** via Multi-Head Attention
- ✅ **Rich representations** via learned embeddings
- ✅ **Stable training** via residual connections and layer normalization
- ✅ **Selective normalization** preserving feature semantics

### Implementation

The model is implemented in `notebooks/CNN_LSTM.ipynb` and includes:
1. Autocorrelation analysis to determine optimal window sizes
2. Feature correlation analysis
3. Complete training pipeline with grid search
4. 5-fold cross-validation
5. Final model training on full dataset
6. Test prediction generation

### Usage

To train and use the model:

```python
# 1. Run autocorrelation analysis
# Execute cells in Section 3 of CNN_LSTM.ipynb

# 2. Run grid search (optional, uses quick test config by default)
# Execute cells in Section 5.6-5.7

# 3. Train final model
# Execute cells in Section 5.8

# 4. Generate predictions
# Execute cells in Section 5.10

# 5. Create submission
# Execute cell in Section 5.11
```

### Why This Model Works

1. **Hierarchical Feature Learning**: CNN captures local motion patterns → LSTM models temporal evolution → Attention focuses on critical moments

2. **Bidirectional Context**: Bidirectional LSTM sees both past and future context, crucial for understanding pain sequences

3. **Attention Mechanism**: Multi-head attention identifies which timesteps matter most for pain classification (e.g., specific motion moments indicating discomfort)

4. **Positional Awareness**: Learnable positional embeddings help model understand temporal relationships and sequence ordering

5. **Feature Engineering**: Binary `is_pirate` feature provides clear semantic signal, while selective normalization preserves meaningful scales

---