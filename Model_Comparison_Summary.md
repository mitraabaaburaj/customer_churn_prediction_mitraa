#  Model Comparison Summary


## Dataset & Problem Setup

**Dataset:** Kaggle Bank Customer Churn — 10,000 customers, 20.4% churn rate
**Problem type:** Binary classification (Churn = 1 / Stay = 0)
**Core challenge:** Class imbalance — a naive model that always predicts "stays" hits ~80% accuracy while catching zero actual churners.

---

## Why Accuracy Alone Is Misleading

| Naive "always stays" model | |
|---|---|
| Accuracy | 79.6% — looks fine |
| Recall for churners | 0% — completely useless |
| F1-Score | 0% — completely useless |

This is why **ROC-AUC, Recall, and F1** are the primary metrics throughout this project.

---

## Class Imbalance Handling

Random oversampling of the minority class (churners) applied to the training set only.

| | Before | After |
|---|---|---|
| Training: Retained | 6,370 | 6,370 |
| Training: Churned | 1,630 | 6,370 |
| Churn rate (train) | 20.4% | 50.0% |
| Test set | Unchanged | 80:20 real distribution |

The test set was never touched evaluation reflects real-world class distribution.

---

## Decision Tree (Pre-check)

Before training the final three models,a plain Decision Tree with no depth limit was tested:

| Metric | Train | Test | Gap |
|---|---|---|---|
| Accuracy | 1.0000 | 0.7990 | 20.1 pp |

This confirmed severe overfitting the tree memorised the training data.This result justified moving to ensemble methods.

---

## Model Results

| Metric | Logistic Regression | Random Forest | XGBoost | Rule-Based |
|---|---|---|---|---|
| Accuracy | 0.7825 | 0.8330 | 0.8160 | 0.8040 |
| Precision | 0.4772 | 0.5737 | 0.5363 | — |
| Recall | 0.7199 | 0.6978 | 0.7076 | 0.1327 |
| F1-Score | 0.5739 | 0.6297 | 0.6102 | 0.2160 |
| ROC-AUC | 0.8403 | 0.8649 | 0.8592 | N/A |
| CV AUC (5-fold) | 0.8376 | 0.9201 | **0.9242** | N/A |
| CV Std | 0.0087 | 0.0068 | 0.0068 | — |



## Model-by-Model Analysis

### Logistic Regression — Baseline

Logistic Regression performed the weakest on accuracy and F1,but it had the **highest Recall** at 0.7199 slightly above XGBoost.This is expected: because LR applies `class_weight` adjustments more aggressively with balanced training data,it tends to predict churn more liberally,catching more true churners at the cost of more false alarms.

**Why it underperforms overall:** The Products vs Churn relationship is non-linear 2 products = 7.6% churn,3 products = 82.7%.A straight decision boundary cannot represent this spike regardless of regularisation. Logistic Regression's precision was also the lowest (0.4772), meaning half of its churn predictions were wrong.

**Best use:** Explainability baseline.Coefficient signs confirm feature directions from EDA.

---

### Random Forest — Reliable Ensemble

Random Forest improved accuracy (+5.05 pp over LR) by averaging 300 trees each trained on random data subsets.This naturally handles the non-linear Products pattern without any explicit modelling.

**Hyperparameter tuning result:**
- Before: Accuracy 0.8330, AUC 0.8649, F1 0.6297
- After GridSearch: Accuracy 0.8545, AUC 0.8594, F1 0.6321
- Change: +2.15 pp accuracy, marginal F1 improvement

The small improvement confirmed the model was already reasonably configured.

**Feature importance (top 5):** Age (0.255), Purchases (0.177), SeniorInactive (0.118), Balance (0.088), OverBanked (0.066).Three of the top five are from feature engineering — validating that combining signals was useful.

**Best use:** Production model where interpretability via feature importance is needed. Solid generalisation confirmed by CV AUC = 0.9201.

---

### XGBoost — Best Model 

XGBoost builds 300 trees sequentially,each one specifically correcting the errors of the previous round.By round 300,the model has focused on the genuinely difficult customers near the decision boundary customers who look like they might stay but actually churn,and vice versa.

**Why CV AUC wins the decision:** XGBoost achieves 0.9242 vs Random Forest's 0.9201 on 5-fold CV.Both have the same CV Std (0.0068),meaning neither is more volatile.The gap is consistent XGBoost generalises slightly better across all five folds.

**Feature importance (top 5):** SeniorInactive (0.322), Purchases (0.160), Age (0.099), Geography_Germany (0.087), IsActiveMember (0.062). SeniorInactive the engineered feature combining age and inactivity ranked first.This is the strongest validation of the feature engineering step.

**Best use:** Primary production scoring model.Monthly batch scoring, P(churn) thresholds for outreach tiers.

---

### Rule-Based System — Explainability Layer

The rule system scores each customer using 7 weighted rules derived from EDA observations. High scores (≥50 pts) flag a customer as churn-risk.

| Rule | Points | EDA Justification |
|---|---|---|
| Low purchases + Basic + low income | +40 | Core low-engagement profile |
| Age > 55 | +30 | 55.2% churn rate observed |
| Zero balance + Basic | +20 | Idle account pattern |
| Germany geography | +15 | 32.4% vs 16% baseline |
| Tenure ≤ 1 year | +10 | No loyalty established |
| Age 46–55 | +15 | 34% churn rate observed |
| High idle balance + Basic | +20 | Preparing-to-switch signal |

**Test set performance:**
- Accuracy: 0.8040 — deceptively high (80% baseline effect)
- Recall: 0.1327 — catches only 13% of actual churners
- F1: 0.2160 — reflects poor recall

The ML models outperform the rule system on every meaningful metric.However,the rule system serves a different purpose it is the explanation layer.When a customer asks why they received a retention offer,the rules provide an auditable,human-readable answer.

---

## Confusion Matrix Summary (Test Set — 2,000 customers)

| Model | True Positive (Caught Churners) | False Negative (Missed Churners) | False Positive (False Alarms) |
|---|---|---|---|
| Logistic Regression | 293 / 407 | 114 | 321 |
| Random Forest | 284 / 407 | 123 | 211 |
| XGBoost | 288 / 407 | 119 | 249 |

Random Forest has the fewest false alarms (211) fewer unnecessary retention offers.XGBoost catches more churners (288) with a moderate false alarm count.

---

## Final Recommendation

| Aspect | Recommendation |
|---|---|
| Primary scorer | XGBoost — highest CV AUC, highest recall |
| Explanation layer | Rule-based system — auditable, no ML required |
| Monitoring metric | CV AUC (tracks generalisation, not just test set luck) |
| Retrain schedule | Quarterly, or after major product changes |

**Scoring tiers:**

| P(churn) | Action |
|---|---|
| > 0.65 | Personal outreach within 7 days |
| 0.40–0.65 | Watch list + light engagement nudge |
| < 0.40 | Standard relationship management |

---

## Key Insight: Why Models Beat Rules

The non-linear Products pattern (2 products = safe, 3–4 = danger) and the compound interaction between Age,Balance,and Inactivity cannot be expressed cleanly in simple threshold rules. The ML models particularly XGBoost,learn these interactions automatically. Rules provide coverage for the obvious cases,the models handle the ambiguous middle.


