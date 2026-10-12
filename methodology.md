# Methodology: Titanic Survival Prediction

## Refined Research Question

**Can we predict whether a Titanic passenger survived, and which passenger characteristics (class, sex, age, family size, fare, and derived features) are most predictive of survival?**

More specifically: does a gradient-boosted ensemble outperform a logistic regression baseline, and does feature engineering (title extraction, family-size binning, fare bucketing) meaningfully improve either model's AUC and F1-score?

---

## Dataset Description

**Source:** The Titanic dataset is a canonical public ML dataset originally compiled from British Board of Trade inquiry records and maintained on Kaggle (`train.csv`, 891 rows) and via seaborn's built-in loader.

### Features

| Feature | Type | Description |
|---|---|---|
| `Pclass` | Ordinal int (1/2/3) | Passenger ticket class; proxy for socioeconomic status |
| `Sex` | Binary categorical | Male or female |
| `Age` | Continuous float | Age in years; ~20% missing |
| `SibSp` | Discrete int | # siblings/spouses aboard |
| `Parch` | Discrete int | # parents/children aboard |
| `Fare` | Continuous float | Ticket price in British pounds; right-skewed |
| `Embarked` | Nominal categorical | Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton); 2 missing |
| `Name` | String | Passenger name; used only for title extraction |

**Excluded:** `PassengerId` (arbitrary index), `Ticket` (high cardinality, no ordinal meaning), `Cabin` (>77% missing; presence/absence encoded separately).

### Target

`Survived` — binary (0 = did not survive, 1 = survived). Class distribution: ~61.6% did not survive / ~38.4% survived. Mild imbalance; handled via stratified splits and AUC metric rather than resampling.

### Size

891 samples × 8 used features → 891 × ~12 after engineering.

### Limitations

- Small dataset; results may not generalize beyond historical Titanic-like settings.
- Age missingness (~20%) requires imputation; any imputation introduces uncertainty.
- Cabin missingness makes direct socioeconomic inference unreliable.
- Fare is per-ticket, not per-person — family tickets may inflate individual fare representation.

---

## Data Cleaning Plan

1. **Drop irrelevant columns:** `PassengerId`, `Ticket`, `Cabin` (keep a binary `HasCabin` indicator before dropping).
2. **Impute `Age`:** median imputation grouped by `Pclass` and `Sex` to preserve demographic structure.
3. **Impute `Embarked`:** mode imputation (S is overwhelming mode; 2 missing values).
4. **Impute `Fare`:** median imputation (1 missing in test set; 0 in train).
5. **Validate dtypes and null counts** after each step; assert zero remaining nulls before modelling.

---

## Feature Engineering Plan

| New Feature | Source | Rationale |
|---|---|---|
| `Title` | Extracted from `Name` (regex) | Social status proxy; groups rare titles (Rev, Dr, Lady…) into `Rare` |
| `FamilySize` | `SibSp + Parch + 1` | Total people in family group; non-linear survival effect (solo and very large families fared worst) |
| `IsAlone` | `FamilySize == 1` | Binary flag capturing the solo-traveller effect |
| `FareBin` | `pd.qcut(Fare, 4)` | Ordinal quartile bucket to reduce fare's right-skew influence |
| `AgeBin` | `pd.cut(Age, bins=[0,12,18,35,60,100])` | Captures non-linear age effects (children prioritised, elderly disadvantaged) |
| `HasCabin` | `Cabin.notna()` → bool | Cabin record correlates with wealth; mostly 1st-class passengers |

All categorical features (including engineered ones) are one-hot encoded; `Sex` is label-encoded (binary). Numerical features are standard-scaled for the logistic regression model (not strictly needed for tree methods but kept in a unified pipeline).

---

## Models

### Model 1: Logistic Regression (Baseline)

**Why:** Simple, interpretable, fast, and establishes a linear baseline. Regularised with L2 (`C=1.0`) to prevent overfitting on the small dataset. Requires scaled numerical features (included in its Pipeline).

**Implementation:** `sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)`

### Model 2: Random Forest Classifier

**Why:** Captures non-linear interactions (e.g., `Sex × Pclass` interactions that are well-documented in the Titanic data) without requiring feature scaling. Naturally robust to irrelevant features via random subspace selection. Provides feature importances.

**Implementation:** `sklearn.ensemble.RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)`

### Model 3: XGBoost Classifier

**Why:** State-of-the-art on tabular classification; handles the mild class imbalance via `scale_pos_weight`. Expected to produce the best AUC given the literature evidence (Chen & Guestrin, 2016). Provides SHAP-compatible feature importance.

**Implementation:** `xgboost.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss', random_state=42)`

---

## Evaluation Metrics

| Metric | Why chosen |
|---|---|
| **ROC-AUC** | Primary metric. Threshold-independent; appropriate for binary classification with mild imbalance. Measures discriminatory power across all possible thresholds. |
| **F1-Score (macro)** | Secondary metric. Balances precision and recall; penalises models that classify every passenger the same way. Macro-averaging treats both classes equally. |
| **Accuracy** | Reported for context/comparison only; less meaningful under class imbalance but familiar to stakeholders. |
| **Confusion Matrix** | Visualised for each model; reveals false-negative and false-positive patterns, which have different real-world costs in the survival context. |

**Evaluation protocol:** Stratified 5-fold cross-validation on the full training set (891 samples). Final numbers reported as mean ± standard deviation across folds.
