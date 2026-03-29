# Customer Churn Prediction


## Project Overview

Banks lose money every time a customer leaves not just the account balance,but the lifetime revenue they would have generated.This project builds a machine learning pipeline to predict which customers are likely to churn,using the Kaggle Bank Customer Churn dataset.

Three models are trained and compared: Logistic Regression, Random Forest, and XGBoost.A rule-based churn scoring system is also implemented to complement the ML approach with human-readable logic.

---

## Project Structure

```
├── Customer_Churn_Prediction.ipynb      ← Main Jupyter Notebook (all code)
├── Churn_Modelling.csv                  ← Dataset (Kaggle)
├── README.md                            ← This file
├── Model_Comparison_Summary.md          ← Model comparison writeup
├── Customer_Churn_Project_Document.docx ← Full project report (Word)
│
├── plot_01_churn_distribution.png       ← Class distribution: 20.4% churn rate
├── plot_02_age_churn.png                ← Age distribution & churn by age group
├── plot_03_purchases_churn.png          ← Non-linear products vs churn U-curve
├── plot_04_membership_churn.png         ← Basic 26.9% vs Premium 14.3% churn
├── plot_05_income_churn.png             ← Income: near-flat (weak predictor)
├── plot_06_geography_churn.png          ← Germany 32.4% vs France/Spain ~16%
├── plot_07_balance_gender.png           ← Balance distribution & gender churn
├── plot_08_model_comparison.png         ← All-metrics bar chart comparison
├── plot_09_roc_curves.png               ← ROC curves (all 3 models)
├── plot_10_confusion_matrices.png       ← Confusion matrices (test set)
├── plot_11_feature_importance.png       ← Random Forest feature importance
├── plot_12_gb_feature_importance.png    ← XGBoost feature importance
└── plot_13_tuning.png                   ← RF: before vs after hyperparameter tuning
```

---

## Dataset

- **Source:** [Kaggle — Bank Customer Churn Modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
- **Size:** 10,000 customers × 14 columns
- **Target:** `Exited` — 1 = churned, 0 = stayed
- **Churn Rate:** 20.4% — class imbalance handled with oversampling

### Field Mapping

The assignment specifies: **Age, Income, Purchases, Membership, Churn**. The Kaggle dataset uses different column names — here is the mapping applied in code:

| Assignment Field | CSV Column        | Notes                                        |
|---               |---                |---                                           |
| Age              | `Age`             | Direct match                                 |
| Income           | `EstimatedSalary` | Income proxy — no actual income column       |
| Purchases        | `NumOfProducts`   | Products held (1–4) used as engagement proxy |
| Membership       | `IsActiveMember`  | 0 = Basic (inactive), 1 = Premium (active)   |
| Churn            | `Exited`          | Target — 1 = left, 0 = stayed                |

---

## Setup & Installation

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter

jupyter notebook Customer_Churn_Prediction.ipynb
```



##  Methodology

### 1 — Load & Inspect
Loaded 10,000-row CSV.Verified column types.Confirmed zero nulls and zero duplicates. Applied field mapping to rename columns to match assignment requirements.

### 2 — Exploratory Data Analysis (EDA)
13 visualisations produced across churn by age group,products,membership,income,geography, balance,and gender.Key EDA surprises drove all subsequent modelling decisions income turned out to be flat,income barely matters,while products had a non-linear spike at 3–4 that ruled out Logistic Regression as a viable primary model.

### 3 — Data Cleaning
Dropped identifier columns (RowNumber, CustomerId, Surname).No missing value imputation needed.All values confirmed within sensible ranges.

### 4 — Encoding
- `Gender` → LabelEncoder (binary)
- `Geography` → One-Hot Encoding (3 categories — avoids ordinal assumption)

### 5 — Feature Engineering
Four new features, each grounded in a specific EDA observation:

| Feature | Logic | EDA Justification |
|---|---|---|
| `SeniorInactive` | Age > 50 AND inactive | Combines the two strongest individual signals |
| `OverBanked` | NumOfProducts >= 3 | Over-selling danger zone (83–100% churn) |
| `IdleAccount` | Balance > 50k AND inactive | High balance + inactivity = preparing to switch |
| `HighIdleRisk` | Balance > 100k AND Age > 50 AND inactive | Compound risk signal |

All four appeared in the **top 5 features** for both ensemble models confirming they gave the model something it could not reconstruct from individual columns.

### 6 — Class Imbalance (SMOTE-style Oversampling)
Without handling the 80/20 split models default to predicting "stays" for almost everyone. Random oversampling applied to training set only.

- Before: 6,370 retained / 1,630 churned
- After oversampling: 6,370 / 6,370 balanced
- Test set: kept at real-world 80:20 split for honest evaluation

### 7 — Model Training
A plain Decision Tree was tested first it overfit severely (100% train accuracy, 79.9% test).This confirmed the move to ensemble methods.

Three final models trained with 5-fold stratified cross-validation:

| Model | Key Params |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Random Forest | `n_estimators=300`, `max_depth=10` |
| XGBoost | `n_estimators=300`, `learning_rate=0.08`, `max_depth=5` |

### 8 — Hyperparameter Tuning
GridSearchCV (3-fold) on Random Forest.Best params: `max_depth=None, min_samples_leaf=3, n_estimators=200`.Accuracy improved by +2.15 percentage points.Small gain confirms the model was already well-configured XGBoost's advantage is structural,not parameter-based.

### 9 — Rule-Based Churn Scoring
7 weighted rules produce a risk score per customer.Every rule maps directly to a pattern observed in EDA.Customers scoring above the threshold are flagged as churn-risk.

---

## Results

| Model | Accuracy | Recall | F1-Score | ROC-AUC | CV AUC (5-fold) |
|---|---|---|---|---|---|
| Logistic Regression | 0.7825 | 0.7199 | 0.5739 | 0.8403 | 0.8376 |
| Random Forest | 0.8330 | 0.6978 | 0.6297 | 0.8649 | 0.9201 |
| **XGBoost** | 0.8160 | **0.7076** | 0.6102 | 0.8592 | **0.9242** |
| Rule-Based System | 0.8040 | 0.1327 | 0.2160 | N/A | N/A |

### Best Model: XGBoost (CV AUC = 0.9242)

XGBoost achieves the highest cross-validated AUC across all 5 folds the most reliable measure of generalisation.It also achieves the highest recall (0.7076),meaning it catches the most actual churners the error that matters most in a retention context.

---

## Key EDA Findings

| Finding | Data |
|---|---|
| Age is the strongest predictor | 51–60 group churns at 55.2% |
| Products has non-linear U-curve | 2 products = 7.6% churn; 3–4 = 83–100% |
| Inactivity precedes churn | Basic: 26.9% vs Premium: 14.3% |
| Germany is a structural problem | 32.4% vs France/Spain ~16% |
| Income barely matters | Near-flat churn rate across all income bands |
| High balance + inactive = warning | Churned customers hold higher balances |

---

## Business Recommendations

1. **Act at Premium → Basic transition** — inactivity is a leading indicator. Trigger retention outreach when a customer drops to Basic, not when they submit the closure request.
2. **Investigate Germany separately** — a 2× churn rate cannot be fixed by the model. Exit interviews and root cause analysis needed.
3. **Cap cross-selling at 2 products** — 3+ products spike to 83–100% churn. Overselling creates frustrated customers.
4. **Run monthly XGBoost scoring** — P(churn) > 0.65 → personal outreach within 7 days. P(churn) 0.40–0.65 → watch list with light engagement.
5. **Senior loyalty programme** — the 45–60 age group has savings and financial knowledge to switch. Better rates or a loyalty tier addresses the core driver.

---


**MITRAA B**
**MSc AIML - 2303717674422024**