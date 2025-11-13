Nice, let’s turn the pirate pain dataset into a **tree-ensemble pipeline** in sklearn first. 🏴‍☠️🌲

Below is a **complete, ready-to-run pipeline** that:

1. Loads `pirate_pain_train.csv` + `pirate_pain_train_labels.csv`
2. Aggregates each time series (per `sample_index`) into a feature vector
3. Encodes the categorical columns (`n_legs`, `n_hands`, `n_eyes`)
4. Trains **tree-based ensembles** (RF, ExtraTrees, HistGB)
5. Builds a **soft-voting ensemble**
6. Uses **GroupKFold** over `sample_index` to avoid leakage
7. Fits the final ensemble with **balanced sample weights**
8. Saves the trained model

You can put this into a single `.py` script or into a notebook cell.

---

## Full sklearn tree-ensemble pipeline

```python
import pandas as pd
import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import joblib


# ---------- 1. Feature extraction: time series -> tabular ----------

def build_feature_table(ts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates each pirate's time series (one sample_index) into a single
    feature vector.

    - Aggregates pain_survey_* and joint_* over time with mean/std/min/max
    - Keeps n_legs, n_hands, n_eyes as one-hot encoded static features
    """
    # time-varying numeric columns
    survey_cols = [c for c in ts_df.columns if c.startswith("pain_survey_")]
    joint_cols = [c for c in ts_df.columns if c.startswith("joint_")]
    num_cols = survey_cols + joint_cols

    # aggregate numeric over time
    agg_funcs = ["mean", "std", "min", "max"]
    feat_num = ts_df.groupby("sample_index")[num_cols].agg(agg_funcs)
    # flatten MultiIndex columns: (col, func) -> "col_func"
    feat_num.columns = [f"{c}_{func}" for c, func in feat_num.columns]

    # static categoricals: they don't change over time for a given sample
    static_cat = (
        ts_df.groupby("sample_index")[["n_legs", "n_hands", "n_eyes"]]
        .first()
    )

    # one-hot encode n_legs, n_hands, n_eyes
    static_cat_ohe = pd.get_dummies(
        static_cat,
        columns=["n_legs", "n_hands", "n_eyes"],
        drop_first=False,
        dtype=int,
    )

    # combine numeric + one-hot categoricals
    X = pd.concat([feat_num, static_cat_ohe], axis=1)

    return X


# ---------- 2. Define our tree models ----------

def make_models(random_state: int = 0):
    """
    Returns a dict of tree-based models and a soft-voting ensemble.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )

    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )

    # Note: older sklearn versions don't support class_weight for HistGB
    hgb = HistGradientBoostingClassifier(
        max_depth=None,
        learning_rate=0.05,
        max_iter=300,
        random_state=random_state,
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("et", et), ("hgb", hgb)],
        voting="soft",
        n_jobs=-1,
    )

    return {
        "rf": rf,
        "et": et,
        "hgb": hgb,
        "voting": ensemble,
    }


# ---------- 3. Main training / evaluation pipeline ----------

def main():
    # --- 3.1 Load data ---
    train_df = pd.read_csv("pirate_pain_train.csv")
    labels_df = pd.read_csv("pirate_pain_train_labels.csv")

    # --- 3.2 Build features per sample_index ---
    X = build_feature_table(train_df)

    # align labels with X.index (sample_index)
    y = labels_df.set_index("sample_index").loc[X.index, "label"]

    # encode labels to integers for sklearn
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # groups for GroupKFold: one group per sample_index
    groups = X.index.values

    # --- 3.3 Define models ---
    models = make_models(random_state=42)

    # --- 3.4 Cross-validation with GroupKFold ---
    gkf = GroupKFold(n_splits=5)

    print("=== Cross-validation results (GroupKFold over sample_index) ===")
    cv_scores = {}

    for name, model in models.items():
        print(f"\nModel: {name}")
        cv = cross_validate(
            model,
            X,
            y_enc,
            cv=gkf,
            groups=groups,
            scoring=["accuracy", "f1_macro"],
            return_train_score=False,
            n_jobs=-1,
        )
        cv_scores[name] = cv

        print(f"  accuracy:  {cv['test_accuracy'].mean():.4f} "
              f"+/- {cv['test_accuracy'].std():.4f}")
        print(f"  f1_macro:  {cv['test_f1_macro'].mean():.4f} "
              f"+/- {cv['test_f1_macro'].std():.4f}")

    # --- 3.5 Choose ensemble as final model (you can change this) ---
    best_name = "voting"
    best_model = models[best_name]

    # --- 3.6 Fit final ensemble on full dataset with balanced weights ---
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_enc)
    best_model.fit(X, y_enc, sample_weight=sample_weight)

    print(f"\nFitted final model: {best_name} on full training set.")

    # --- 3.7 In-sample evaluation (sanity check) ---
    y_pred = best_model.predict(X)
    print("\nIn-sample classification report (sanity check, not CV):")
    print(classification_report(y_enc, y_pred, target_names=le.classes_))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_enc, y_pred))

    # --- 3.8 Save model + label encoder + feature columns ---
    out_obj = {
        "model": best_model,
        "label_encoder": le,
        "feature_columns": X.columns.tolist(),
    }
    joblib.dump(out_obj, "pirate_pain_tree_ensemble.pkl")
    print("\nSaved model to pirate_pain_tree_ensemble.pkl")


if __name__ == "__main__":
    main()
```

---

## How this maps to your problem

* **Time series → single row per pirate**
  `build_feature_table()` does the heavy lifting:

  * Aggregates `pain_survey_*` and `joint_00`–`joint_30` over time
  * Preserves `n_legs`, `n_hands`, `n_eyes` via one-hot (only categories that actually appear are used, so we don’t invent new values)

* **Proper time-series CV**
  `GroupKFold` with `groups = sample_index` ensures we never mix time steps of the same pirate across folds.

* **Tree ensembles**

  * RandomForest + ExtraTrees with `class_weight="balanced"`
  * HistGradientBoosting as a strong gradient-boosted tree baseline
  * Combined with a **soft-voting ensemble** that averages predicted probabilities.

* **Imbalance handling**

  * `class_weight="balanced"` in RF / ExtraTrees
  * `sample_weight` passed to the final `VotingClassifier.fit` so all three models see balanced weights when fitting on the full dataset.

---

If you’re happy with this structure, next step we can do is:

* Build a **PyTorch (or sklearn) deep model** taking the raw `(seq_len, n_features)` time series and
* Either compare it to this tree ensemble, or **stack** them (e.g., meta-learner on top of tree + NN outputs).
