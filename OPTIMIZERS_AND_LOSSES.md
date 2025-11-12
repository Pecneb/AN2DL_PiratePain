# Optimizers and Loss Functions Guide

This document explains all the optimizers and loss functions now available in the hyperparameter tuning script.

---

## 🎯 Loss Functions (Criterion)

### 1. **CrossEntropy** (Default)
Standard cross-entropy loss for multi-class classification.

**Formula:**
$$L = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

**Best for:**
- ✅ Balanced datasets
- ✅ Standard classification tasks
- ✅ Good baseline choice

**Pros:**
- Simple and well-understood
- Fast computation
- Works well in most cases

**Cons:**
- Doesn't handle class imbalance well
- Treats all misclassifications equally

---

### 2. **FocalLoss**
Addresses class imbalance by down-weighting easy examples and focusing on hard examples.

**Formula:**
$$FL(p_t) = -\alpha (1 - p_t)^\gamma \log(p_t)$$

where:
- $p_t$ = probability of the true class
- $\gamma$ = focusing parameter (default: 2.0)
- $\alpha$ = balancing factor (default: 1.0)

**Best for:**
- ✅ **Imbalanced datasets** (e.g., 10:1:1 class ratio)
- ✅ Hard-to-classify samples
- ✅ When some classes are underrepresented

**Pros:**
- Automatically handles class imbalance
- Focuses on hard examples
- No need to manually compute class weights

**Cons:**
- More hyperparameters ($\gamma$, $\alpha$)
- Slightly slower than CrossEntropy
- May underfit if $\gamma$ too high

**When to use:** If your training data has class imbalance (check with `pd.value_counts()`)

---

### 3. **LabelSmoothing**
Cross-entropy with label smoothing to prevent overconfident predictions.

**Formula:**
$$L = (1 - \epsilon) \cdot L_{CE} + \epsilon \cdot L_{uniform}$$

where $\epsilon = 0.1$ (smoothing factor)

**Best for:**
- ✅ Preventing overfitting
- ✅ Improving model calibration
- ✅ Better generalization

**Pros:**
- Reduces overconfidence
- Better calibrated probabilities
- Slight regularization effect

**Cons:**
- May reduce training accuracy slightly
- Fixed smoothing factor (0.1)

**When to use:** When model is too confident (high training accuracy but poor validation)

---

### 4. **CrossEntropyWeighted** (Placeholder)
Currently same as CrossEntropy, but can be extended to use class weights.

**Future implementation:**
```python
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

---

## 🚀 Optimizers

### 1. **Adam** (Default)
Adaptive Moment Estimation - combines momentum and adaptive learning rates.

**Update rule:**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\theta_t = \theta_{t-1} - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}$$

**Best for:**
- ✅ **Most deep learning tasks** (default choice)
- ✅ Non-convex optimization
- ✅ Sparse gradients

**Pros:**
- Fast convergence
- Works well out-of-the-box
- Adapts learning rate per parameter
- Handles sparse gradients well

**Cons:**
- May overfit on small datasets
- Can get stuck in local minima
- L2 regularization not as effective

**Hyperparameters:**
- Learning rate: 1e-5 to 1e-2
- Weight decay: 1e-6 to 1e-3

---

### 2. **AdamW**
Adam with decoupled weight decay (better regularization).

**Key difference from Adam:**
Weight decay is applied **directly to weights**, not through gradients.

$$\theta_t = (1 - \eta \lambda) \theta_{t-1} - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}$$

**Best for:**
- ✅ **Recommended over Adam** for most tasks
- ✅ Better generalization
- ✅ When using weight decay

**Pros:**
- Improved weight decay implementation
- Better generalization than Adam
- More stable training
- Less overfitting

**Cons:**
- Slightly slower than Adam
- May need tuning weight decay

**When to use:** Prefer AdamW over Adam when using weight decay > 0

---

### 3. **SGD** (Stochastic Gradient Descent)
Classic optimizer with momentum and Nesterov acceleration.

**Update rule (with Nesterov momentum):**
$$v_t = \mu v_{t-1} + g_t$$
$$\theta_t = \theta_{t-1} - \eta v_t$$

**Best for:**
- ✅ Convex optimization
- ✅ Large datasets
- ✅ When Adam overfits

**Pros:**
- Better generalization than Adam
- More stable final solution
- Less memory usage
- Good with learning rate scheduling

**Cons:**
- **Slower convergence** than Adam/AdamW
- Requires careful learning rate tuning
- May need learning rate scheduling
- Sensitive to initialization

**Hyperparameters:**
- Learning rate: **Higher than Adam** (e.g., 0.01 - 0.1)
- Momentum: 0.9 (fixed)
- Nesterov: True (fixed)

**When to use:** When Adam overfits or for final tuning after Adam

---

### 4. **RMSprop**
Root Mean Square Propagation - adaptive learning rate method.

**Update rule:**
$$v_t = \beta v_{t-1} + (1-\beta) g_t^2$$
$$\theta_t = \theta_{t-1} - \eta \frac{g_t}{\sqrt{v_t} + \epsilon}$$

**Best for:**
- ✅ **Recurrent Neural Networks (RNNs)**
- ✅ Non-stationary objectives
- ✅ Online learning

**Pros:**
- Good for RNNs (originally designed for)
- Adapts learning rate
- Works well with mini-batches
- Less memory than Adam

**Cons:**
- Can be unstable
- Less popular than Adam
- May need learning rate decay

**When to use:** Good alternative for RNN/LSTM/GRU models (your use case!)

---

### 5. **AdaGrad**
Adaptive Gradient - adapts learning rate based on historical gradients.

**Update rule:**
$$G_t = G_{t-1} + g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t$$

**Best for:**
- ✅ Sparse features
- ✅ NLP tasks
- ✅ Large learning rates

**Pros:**
- No manual learning rate tuning
- Good for sparse data
- Works well with sparse gradients

**Cons:**
- **Learning rate decays too aggressively**
- May stop learning too early
- Not recommended for deep networks

**When to use:** Rarely - mostly for sparse features or NLP

---

## 🔍 Comparison Table

| Optimizer | Speed | Generalization | Best Use Case | Learning Rate Range |
|-----------|-------|----------------|---------------|---------------------|
| **Adam** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | General purpose | 1e-5 to 1e-2 |
| **AdamW** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Most tasks** | 1e-5 to 1e-2 |
| **SGD** | ⭐⭐ | ⭐⭐⭐⭐⭐ | Final tuning | 1e-3 to 1e-1 |
| **RMSprop** | ⭐⭐⭐⭐ | ⭐⭐⭐ | **RNNs/LSTMs** | 1e-4 to 1e-2 |
| **AdaGrad** | ⭐⭐⭐ | ⭐⭐ | Sparse features | 1e-2 to 1e-1 |

---

## 💡 Recommendations for Your Use Case

### For Pirate Pain Time Series Classification:

#### **Recommended Combinations:**

1. **Best Overall (Start Here):**
   ```python
   optimizer: AdamW
   criterion: CrossEntropy
   learning_rate: 1e-3
   weight_decay: 1e-4
   ```

2. **If Classes Are Imbalanced:**
   ```python
   optimizer: AdamW
   criterion: FocalLoss
   learning_rate: 1e-3
   weight_decay: 1e-4
   ```

3. **For RNN/LSTM Models (Your Case):**
   ```python
   optimizer: RMSprop or AdamW
   criterion: CrossEntropy
   learning_rate: 5e-4
   weight_decay: 1e-5
   ```

4. **If Overfitting:**
   ```python
   optimizer: AdamW
   criterion: LabelSmoothing
   learning_rate: 1e-3
   weight_decay: 1e-3  # Higher!
   ```

5. **For Best Generalization:**
   ```python
   optimizer: SGD
   criterion: CrossEntropy
   learning_rate: 1e-2  # Much higher!
   weight_decay: 1e-4
   ```

---

## 🔬 How to Choose

### Step 1: Check Class Balance
```python
import pandas as pd
labels_df = pd.read_csv('./data/pirate_pain_train_labels.csv')
print(labels_df['pain_level'].value_counts())
```

**If balanced (similar counts):** Use `CrossEntropy`  
**If imbalanced (>2:1 ratio):** Use `FocalLoss`

### Step 2: Start with AdamW + CrossEntropy
Run a baseline experiment to see if model learns.

### Step 3: If Overfitting
- Try `LabelSmoothing`
- Increase `weight_decay`
- Increase `dropout_rate`

### Step 4: If Underfitting
- Try `Adam` (faster convergence)
- Increase model capacity (`hidden_size`)
- Decrease `weight_decay`

### Step 5: Let Optuna Decide
Let the hyperparameter search explore all combinations!

---

## 📊 Monitoring Tips

### Check if optimizer is working:
```bash
# View TensorBoard
tensorboard --logdir ./lightning_logs_tuning
```

**Good signs:**
- ✅ Training loss decreasing smoothly
- ✅ Validation loss following training loss
- ✅ Learning rate not too high (no divergence)

**Bad signs:**
- ❌ Loss exploding (learning rate too high)
- ❌ Loss not decreasing (learning rate too low)
- ❌ Training loss << validation loss (overfitting)

### Compare optimizers:
```python
import pandas as pd
results = pd.read_csv('./hyperparameter_search_results.csv')

# Group by optimizer
by_optimizer = results.groupby('params_optimizer_name')['value'].agg(['mean', 'std', 'max'])
print(by_optimizer.sort_values('mean', ascending=False))

# Group by criterion
by_criterion = results.groupby('params_criterion_name')['value'].agg(['mean', 'std', 'max'])
print(by_criterion.sort_values('mean', ascending=False))
```

---

## 🎯 Quick Reference

### Loss Functions:
- **CrossEntropy**: Balanced data, baseline ✅
- **FocalLoss**: Imbalanced classes 🎯
- **LabelSmoothing**: Prevent overconfidence 🛡️

### Optimizers:
- **AdamW**: Best overall choice ⭐
- **Adam**: Fast convergence 🚀
- **SGD**: Best generalization 🎓
- **RMSprop**: Good for RNNs 🔄
- **AdaGrad**: Sparse features 📊

### Learning Rates:
- Adam/AdamW: **1e-4 to 1e-3**
- RMSprop: **5e-4 to 5e-3**
- SGD: **1e-2 to 1e-1** (10-100x higher!)

---

**Let Optuna find the best combination for your specific dataset!** 🔍
