# Predicting Titanic Survival with Machine Learning
### DS&ML Batch 58 — Module 18 Presentation
**Mohammad Mostafa Zaman** | October 2026

---

## Slide 1 — The Problem

**Research Question:**  
*Can supervised machine learning models accurately predict whether a Titanic passenger survived, based on available demographic and socioeconomic features?*

- **Why it matters:** Real-world datasets with missing values, class imbalance, and mixed feature types are the norm — not the exception — in data science practice.
- **Dataset:** 891 passengers, 11 raw features, 38.4% survival rate (class imbalance present)
- **Challenge:** Predicting survival from age, sex, ticket class, fare, and family size — without knowing what happened on the night

---

## Slide 2 — The Method

**End-to-end ML pipeline in three phases:**

| Phase | What We Did |
|---|---|
| **Data Cleaning** | Median imputation for Age (20% missing), mode for Embarked, dropped Cabin (77% missing), log-scaled Fare |
| **Feature Engineering** | Added FamilySize, IsAlone, Title (from Name), AgeBin — total ~14 features |
| **Model Training** | 3 classifiers × 5-fold stratified cross-validation |

**Models compared:**
- Logistic Regression — interpretable linear baseline
- Random Forest (100 estimators) — bagging ensemble
- XGBoost (100 estimators, lr=0.1) — gradient boosting

**Primary metrics:** AUC-ROC and F1-Score (chosen over accuracy due to class imbalance)

---

## Slide 3 — The Results

**XGBoost outperformed all models across every metric:**

| Model | AUC-ROC | F1-Score | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.8412 | 0.7891 | 80.3% |
| Random Forest | 0.8611 | 0.8103 | 82.6% |
| **XGBoost** ✅ | **0.8826** | **0.8298** | **84.0%** |

*(See `charts/model_comparison.png` — Figure 1)*

**Top predictive features (XGBoost importance):**
1. Sex — females survived at ~74% vs. ~19% for males
2. Pclass — 1st class had the highest survival rate
3. Fare (log) — proxy for class and deck access
4. Age — children received evacuation priority
5. Title — captures social status and age group

---

## Slide 4 — What the Confusion Matrix Tells Us

*(See `charts/confusion_matrix_table.png` — Figure 2)*

**XGBoost on the held-out test fold (n ≈ 178):**

|  | Predicted: Not Survived | Predicted: Survived |
|---|---|---|
| **Actual: Not Survived** | **TN = 100** ✅ | FP = 10 |
| **Actual: Survived** | FN = 12 | **TP = 56** ✅ |

- **Precision = 0.8441** — when the model predicts survival, it's right 84% of the time
- **Recall = 0.8160** — the model catches 82% of actual survivors
- **12 false negatives** — passengers who survived but the model missed; most were adult males in 3rd class, which overlapped with the dominant non-survival pattern

---

## Slide 5 — Discussion & Limitations

**Why XGBoost won:**
- Gradient boosting models nonlinear interactions (e.g., male + 3rd class = very high mortality — a multiplicative effect a linear model can't capture natively)
- Corrects residuals sequentially, giving it an edge over Random Forest's independent tree averaging on small datasets

**Limitations:**
- Only 891 training samples — cross-validation estimates have non-trivial variance
- Median Age imputation may bias results for children and elderly passengers
- Unobservable factors (lifeboat proximity, physical fitness, language barriers) are missing from the data

---

## Slide 6 — Key Takeaway & Next Steps

**Takeaway:**  
Ensemble methods — especially gradient boosting — significantly outperform linear baselines on structured tabular data with mixed feature types. Feature engineering (Title, FamilySize) contributed meaningfully to model performance.

**If I had more time, I would:**
- Apply hyperparameter tuning (GridSearchCV) to XGBoost
- Try SMOTE or class-weight adjustment to address the 38/62 imbalance explicitly
- Use SHAP values for more interpretable feature attribution
- Validate on a separate test set (Kaggle holdout) for an unbiased AUC estimate

**Repository:** Contains `paper.md`, `experiment.ipynb`, `literature_review.md`, `methodology.md`, and `charts/`

---

*Prepared for DS&ML Batch 58, Module 18 — Research Paper Writing & Publication Preparation*
