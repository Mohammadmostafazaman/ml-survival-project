# Literature Review: Machine Learning for Survival Prediction

## Overview

This review examines key works in applying classical and ensemble machine learning methods to structured tabular survival/classification problems, with a focus on feature engineering, model comparison, and evaluation approaches relevant to the Titanic dataset.

---

## Source 1: Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.

Breiman introduced the Random Forest algorithm, which builds an ensemble of decision trees trained on bootstrapped subsets of data and random feature subsets, reducing variance without increasing bias. The paper demonstrates that Random Forests consistently outperform single decision trees and rival boosted models on a broad range of classification benchmarks. This work is foundational for tabular ML because the method handles mixed feature types, missing values, and non-linear interactions naturally — all common characteristics in survival datasets like Titanic.

---

## Source 2: Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16, pp. 785–794.

Chen and Guestrin present XGBoost, a gradient-boosted tree framework that achieves state-of-the-art results on many structured/tabular datasets by combining regularization, second-order gradient approximations, and efficient column-block data structures. The authors show it wins or places in the majority of Kaggle competitions at the time of publication, particularly on structured data problems. For this project, XGBoost serves as the primary boosting-based baseline against which simpler models are compared.

---

## Source 3: Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.

Géron provides an extensive treatment of the Titanic problem as a pedagogical case study, covering the full ML pipeline from exploratory data analysis through feature engineering, pipeline construction, and cross-validated model comparison using scikit-learn. The book illustrates how categorical imputation strategies (median for age, mode for embarkation), title extraction from passenger names, and family-size features materially improve predictive performance over using raw columns alone. This work directly informs the feature engineering plan and scikit-learn Pipeline design used in this project.

---

## Source 4: Varian, H. R. (2014). *Big Data: New Tricks for Econometrics*. Journal of Economic Perspectives, 28(2), 3–28.

Varian argues that ensemble tree methods — particularly Random Forests and boosting — outperform traditional regression on prediction tasks even in relatively small, well-structured datasets, because they automatically capture interaction effects that manual model specification would miss. He discusses cross-validation as the canonical method for model selection when a held-out test set cannot be used, reinforcing the use of stratified k-fold CV over a simple train/test split. The paper provides theoretical grounding for preferring non-parametric ensemble methods when interpretability requirements are moderate.

---

## Source 5: Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling*. Springer.

Kuhn and Johnson give a thorough treatment of the model evaluation problem, arguing that the choice of evaluation metric should be driven by the cost asymmetry of false positives versus false negatives in the application domain. They introduce the ROC-AUC and F1-score as robust alternatives to accuracy when class imbalance is present — a relevant concern for the Titanic dataset where survivors are a minority class (~38%). The book also covers preprocessing strategies such as near-zero-variance feature removal and normalization requirements for distance-based models, informing the scaling steps in this project's pipeline.
