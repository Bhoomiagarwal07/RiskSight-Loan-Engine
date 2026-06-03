# 🏦 RiskSight Loan Engine

> End-to-end supervised ML pipeline predicting loan approvals using binary classification,
> feature engineering, and three model comparison.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

---

## 📌 Problem Statement

**SecureTrust Bank** processes hundreds of loan applications daily through manual review —
a process that is slow, biased, and inconsistent. Two costly errors happen every day:

| Error Type | What happens | Business cost |
|---|---|---|
| ❌ False Negative | Good customer **rejected** | Bank loses legitimate revenue |
| ⚠️ False Positive | Risky customer **approved** | Bank faces loan default losses |

**Solution:** Build an ML system that minimises both errors using
**Precision**, **Recall**, and **F1 Score** — not Accuracy.

> Why not Accuracy? The dataset has 65% rejected vs 35% approved.
> A model rejecting everyone scores 65% accuracy — and is completely useless.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Rows | 1,000 loan applicants |
| Raw features | 19 (income, credit score, DTI ratio, employment, etc.) |
| Engineered features | 9 new features added |
| Target column | `Loan_Approved` — Yes (1) or No (0) |
| Class split | 65% Rejected · 35% Approved |

---

## ⚙️ Feature Engineering

9 features created from business logic — not random transformations:

| Feature | Formula | Business reason |
|---|---|---|
| `Total_Income` | Applicant + Coapplicant Income | Household earning power |
| `Income_per_Dependent` | Total_Income / (Dependents + 1) | Disposable money per person |
| `Has_Coapplicant` | 1 if Coapplicant > 0 | Second earner = lower default risk |
| `Loan_to_Income` | Loan_Amount / Total_Income | Core affordability ratio ⭐ |
| `Savings_to_Loan` | Savings / Loan_Amount | Financial safety net |
| `EMI_estimate` | Loan_Amount / Loan_Term | Monthly repayment burden |
| `Collateral_to_Loan` | Collateral_Value / Loan_Amount | Bank security cover |
| `Credit_x_DTI` | Credit_Score × (1 − DTI_Ratio) | Joint creditworthiness ⭐ |
| `Credit_Category` | Bins: Poor / Fair / Good | Industry threshold effect |

⭐ = highest impact features

---

## 🤖 Model Results

### Before Feature Engineering

| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| Naive Bayes | 0.780 | 0.820 | 0.800 | 87.0% |
| Logistic Regression | 0.750 | 0.790 | 0.769 | 85.5% |
| KNN | 0.700 | 0.720 | 0.710 | 82.5% |

### After Feature Engineering

| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| **Naive Bayes** | **0.705** | **0.917** | **0.797** ✅ | **86.0%** |
| Logistic Regression | 0.684 | 0.900 | 0.777 | 84.5% |
| KNN | 0.680 | 0.850 | 0.756 | 83.5% |

### 🏆 Final Model — Naive Bayes

```
Precision  :  0.705   (70.5% of approved predictions are correct)
Recall     :  0.917   (catches 55 of 60 qualified applicants)
F1 Score   :  0.797
Accuracy   :  86.0%

Confusion Matrix:
              Predicted Rejected    Predicted Approved
Actual Rejected      117 (TN)            23 (FP)
Actual Approved        5 (FN)            55 (TP)
```

**117** risky applicants correctly rejected → defaults avoided  
**55** good applicants correctly approved → revenue captured  
Only **5** good applicants wrongly rejected → minimal customer loss

---

## 🔍 Key Insights from EDA

| # | Finding |
|---|---|
| 1 | Credit Score is the strongest predictor — approved avg **726.9** vs rejected **651.6** |
| 2 | DTI Ratio is the strongest risk signal — approved avg **0.25** vs rejected **0.39** |
| 3 | Loan Amount alone is weak — the *ratio* to income is what matters |
| 4 | 65% class imbalance — Accuracy is misleading, F1 Score is the right metric |
| 5 | Savings and Collateral are weak individually — their ratio to loan amount is powerful |

---

## 💡 Key Learnings

**1. Feature engineering matters more than algorithm choice**
The same 3 models — different data representation — different results.
Ratio features gave every model meaningful signal raw numbers could not.

**2. Business logic drives the best features**
`Loan_to_Income`, `EMI_estimate`, `Collateral_to_Loan` — real loan officers
check these manually. Domain knowledge creates better features than random math.

**3. Naive Bayes has a structural advantage here**
With many binary OHE columns, Naive Bayes handles independence well.
After engineering, its Recall of 91.7% minimises missed good customers.

**4. KNN suffers from the Curse of Dimensionality**
One-hot encoding created many sparse binary columns.
KNN uses distance — in high-dimensional sparse space, every point
looks equally distant. Ratio features helped but could not fully fix this.

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR-USERNAME/RiskSight-Loan-Engine.git
cd RiskSight-Loan-Engine
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Open the notebook**
```bash
jupyter notebook creditwise_loan_engine.ipynb
```

Run all cells from top to bottom — the notebook is fully self-contained.

---

## 📁 Repository Structure

```
RiskSight-Loan-Engine/
│
├── creditwise_loan_engine.ipynb   ← complete analysis notebook
├── loan_approval_data.csv         ← dataset (1,000 applicants)
├── requirements.txt               ← Python dependencies
└── README.md                      ← this file
```

---

## 🛠️ Tech Stack

- **Python 3.10**
- **Pandas** — data manipulation and EDA
- **NumPy** — numerical operations
- **Matplotlib + Seaborn** — visualisation
- **Scikit-learn** — imputation, encoding, scaling, models, metrics

---

## 📈 Next Steps

- [ ] Try Random Forest and XGBoost for potential improvement
- [ ] Apply SMOTE to handle class imbalance
- [ ] Deploy as Streamlit dashboard — live loan approval predictor
- [ ] Add cross-validation (cv=5) for more robust evaluation

---

## 👤 Author

**Bhoomi Agarwal** — Aspiring Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/bhoomi-agarwal-522277330/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/Bhoomiagarwal07)

---

*Built as part of a self-taught data science learning journey · June 2026*
