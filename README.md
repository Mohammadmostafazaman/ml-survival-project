# Titanic Survival Prediction — ML Classification Project

**DS&ML Batch 58 | Module 17–18 Assignment**  
**Author:** Mohammad Mostafa Zaman  
**Date:** October 2026

---

## Project Overview

This project applies supervised machine learning to predict passenger survival on the RMS Titanic. Three classifiers — Logistic Regression, Random Forest, and XGBoost — were trained and compared using 5-fold stratified cross-validation. XGBoost achieved the best performance (AUC = 0.8826, F1 = 0.8298).

---

## Repository Structure

```
ml-survival-project/
├── README.md                  ← This file
├── paper.md                   ← Full research paper (IMRaD format)
├── slides.md                  ← 6-slide presentation deck
├── literature_review.md       ← 5 summarized sources
├── methodology.md             ← Research question, dataset, cleaning, models, metrics
├── experiment.ipynb           ← Jupyter notebook — full ML pipeline with outputs
└── charts/
    ├── model_comparison.png   ← Figure 1: AUC-ROC & F1-Score comparison
    └── confusion_matrix_table.png  ← Figure 2: XGBoost confusion matrix
```

---

## Key Deliverables

| File | Description |
|---|---|
| [📄 paper.md](./paper.md) | Research paper — Abstract, Introduction, Methodology, Results, Discussion, Conclusion, References |
| [📓 experiment.ipynb](./experiment.ipynb) | Full ML pipeline: EDA, data cleaning, feature engineering, model training, evaluation, charts |
| [📊 slides.md](./slides.md) | 6-slide presentation: Problem → Method → Results → Confusion Matrix → Discussion → Takeaway |
| [🔬 literature_review.md](./literature_review.md) | 5 sources summarized: Breiman (2001), Chen & Guestrin (2016), Géron (2019), Varian (2014), Kuhn & Johnson (2013) |
| [⚙️ methodology.md](./methodology.md) | Research question, dataset description, cleaning plan, feature engineering, model rationale, evaluation metrics |

---

## Results Summary

| Model | AUC-ROC | F1-Score | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.8412 | 0.7891 | 80.3% |
| Random Forest | 0.8611 | 0.8103 | 82.6% |
| **XGBoost** ✅ | **0.8826** | **0.8298** | **84.0%** |

**Top features:** Sex, Pclass, Fare (log), Age, Title

---

## How to Run

1. Clone this repository
2. Install dependencies: `pip install pandas numpy scikit-learn xgboost matplotlib seaborn`
3. Open `experiment.ipynb` in Jupyter or Google Colab
4. Run all cells — outputs and charts are already rendered in the notebook

---

## References

1. Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016.*
3. Géron, A. (2019). *Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow* (2nd ed.). O'Reilly.
4. Varian, H. R. (2014). Big data: New tricks for econometrics. *Journal of Economic Perspectives, 28*(2).
5. Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling.* Springer.
