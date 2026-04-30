# 🧠 Employee Churn Prediction using ANN

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  A deep learning project that uses an <strong>Artificial Neural Network (ANN)</strong> to predict employee churn (attrition) based on HR data.<br/>
  Built as part of the Neural Networks course — <strong>Faculty of Artificial Intelligence, Kafrelsheikh University</strong>.
</p>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Installation & Usage](#-installation--usage)
- [Expected Results](#-expected-results)
- [Key Design Decisions](#-key-design-decisions)
- [Author](#-author)

---

## 📌 Project Overview

Employee churn is one of the costliest challenges in human resource management. This project builds a complete end-to-end machine learning pipeline to **predict which employees are likely to leave the company**, enabling HR departments to take proactive retention decisions.

### Pipeline Summary

| Step | Description |
|---|---|
| 🔗 **Data Merging** | Load and join 3 CSV files on `EmployeeID` |
| 🔍 **EDA** | Visualize distributions, correlations, and class imbalance |
| ⚙️ **Preprocessing** | Encode, scale, handle missing values & class imbalance |
| 🏗️ **Model** | Deep ANN with Dropout regularization |
| 📊 **Evaluation** | Accuracy, Loss curves, Confusion Matrix, ROC-AUC |

---

## 📂 Dataset Description

**Source:** [HR Analytics Case Study — Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/hr-analytics-case-study)

The dataset contains **4,410 employee records** distributed across multiple CSV files, all linked by the `EmployeeID` column.

| File | Columns | Key Features |
|---|---|---|
| `general_data.csv` | 24 | Age, Department, MonthlyIncome, JobRole, YearsAtCompany, **Attrition** (target) |
| `employee_survey_data.csv` | 4 | EnvironmentSatisfaction, JobSatisfaction, WorkLifeBalance |
| `manager_survey_data.csv` | 3 | JobInvolvement, PerformanceRating |

### ⚠️ Known Data Issues (handled in code)

| Column | Issue | Fix Applied |
|---|---|---|
| `NumCompaniesWorked` | 19 missing values | Filled with **median** |
| `TotalWorkingYears` | 9 missing values | Filled with **median** |
| `EnvironmentSatisfaction` | 25 missing values | Filled with **median** |
| `JobSatisfaction` | 20 missing values | Filled with **median** |
| `WorkLifeBalance` | 38 missing values | Filled with **median** |

### 🎯 Target Variable — Attrition

```
No  (stayed) : 3,699 employees — 83.9%
Yes (churned):   711 employees — 16.1%
Imbalance ratio: ~5.2×  →  handled via class_weight='balanced'
```

---

## 📁 Project Structure

```
employee-churn-ann/
│
├── 📄 employee_churn_ANN.py          # Main pipeline script (Colab-ready)
├── 📄 README.md                      # This file
│
├── 📂 data/                          # Place your CSV files here
│   ├── general_data.csv
│   ├── employee_survey_data.csv
│   └── manager_survey_data.csv
│
└── 📂 outputs/                       # Auto-generated after running
    ├── employee_churn_ann_model.keras
    ├── attrition_distribution.png
    ├── correlation_heatmap.png
    ├── attrition_by_categories.png
    ├── training_history.png
    ├── confusion_matrix.png
    └── roc_curve.png
```

---

## 🏗️ Model Architecture

The ANN is built using the **TensorFlow / Keras Sequential API**.

```
┌─────────────────────────────────────────────────────┐
│  Input Layer       →  shape: (n_features,)          │
├─────────────────────────────────────────────────────┤
│  Hidden Layer 1    →  Dense(128, ReLU, he_normal)   │
│  Dropout 1         →  rate = 0.3                    │
├─────────────────────────────────────────────────────┤
│  Hidden Layer 2    →  Dense(64,  ReLU, he_normal)   │
│  Dropout 2         →  rate = 0.3                    │
├─────────────────────────────────────────────────────┤
│  Hidden Layer 3    →  Dense(32,  ReLU, he_normal)   │
├─────────────────────────────────────────────────────┤
│  Output Layer      →  Dense(1, Sigmoid)             │
│                       P(Attrition=Yes) ∈ [0, 1]    │
└─────────────────────────────────────────────────────┘
```

### Compilation Settings

| Parameter | Value | Reason |
|---|---|---|
| Optimizer | `Adam (lr = 0.001)` | Adaptive, fast convergence |
| Loss | `Binary Crossentropy` | Standard for binary classification |
| Metric | `Accuracy` | Easy to interpret |
| Class Weight | `balanced` | Corrects the 5.2× class imbalance |
| Early Stopping | `patience = 15` on `val_loss` | Prevents overfitting |

---

## 🚀 Installation & Usage

### ▶️ Option 1 — Google Colab (Recommended)

> ✅ No installation needed. TensorFlow, pandas, and scikit-learn are pre-installed in Colab.

**Step 1** — Open [Google Colab](https://colab.research.google.com) and create a new notebook.

**Step 2** — Upload the script by copying the contents of `employee_churn_ANN.py` into a code cell,
or upload it directly via `File → Upload notebook`.

**Step 3** — Upload your dataset files by running this cell before anything else:

```python
from google.colab import files
uploaded = files.upload()  # Select all 3 CSV files at once
```

**Step 4** — Run all cells via `Runtime → Run all` or press `Ctrl + F9`.

**Step 5** — All output plots and the saved model will appear in the left file browser.
Right-click any file → **Download** to save it locally.

---

### 💻 Option 2 — Run Locally

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/employee-churn-ann.git
cd employee-churn-ann

# 2. Install dependencies
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn

# 3. Add your CSV files to the data/ folder
cp /path/to/general_data.csv         data/
cp /path/to/employee_survey_data.csv data/
cp /path/to/manager_survey_data.csv  data/

# 4. Run the script
python employee_churn_ANN.py
```

---

## 📊 Expected Results

> Results may vary slightly due to random seed and hardware differences.

### Model Performance

| Metric | Expected Value |
|---|---|
| **Test Accuracy** | ~ 85 – 88% |
| **Test Loss** | ~ 0.32 – 0.40 |
| **ROC-AUC Score** | ~ 0.78 – 0.85 |

### Confusion Matrix (on ~882 test samples)

```
                    Predicted: No    Predicted: Yes
  Actual: No  (0)       720               20
  Actual: Yes (1)        90               52
```

| Cell | Name | Meaning |
|---|---|---|
| ✅ Top-left | True Negative (TN) | Correctly predicted employee stays |
| ✅ Bottom-right | True Positive (TP) | Churner correctly caught |
| ⚠️ Bottom-left | False Negative (FN) | Missed churner — most costly HR error |
| ℹ️ Top-right | False Positive (FP) | False alarm — flagged but stayed |

### Classification Report (approximate)

```
              precision    recall    f1-score    support
No Attrition    0.89        0.97       0.93        740
   Attrition    0.72        0.37       0.49        142

    accuracy                           0.87        882
   macro avg    0.80        0.67       0.71        882
weighted avg    0.86        0.87       0.86        882
```

> 💡 **Key metric to watch:** The **Recall** score for `Attrition = Yes`.
> A higher recall means fewer missed churners — the primary business objective.
> Consider lowering the decision threshold from `0.5 → 0.35` to improve recall if needed.

### Training Curves

The model typically converges within **30 – 50 epochs** before early stopping triggers. A healthy training run shows:
- Training and validation accuracy curves that stay close together (no overfitting)
- Validation loss that decreases steadily and then plateaus

---

## 🔑 Key Design Decisions

| Decision | Why |
|---|---|
| **Median imputation** | All missing columns are ordinal numeric (satisfaction scores, career counts). Median is robust to skew — unlike mean. |
| **`stratify=y` in split** | Preserves the 83% / 16% class ratio in both train and test sets for a fair and representative evaluation. |
| **Scaler fitted on train only** | Fitting `StandardScaler` on test data causes data leakage — test statistics would influence the training process. |
| **`drop_first=True` in OHE** | Removes one dummy column per categorical feature to prevent the dummy variable trap (multicollinearity). |
| **`he_normal` initializer** | The recommended weight initialization for ReLU layers — prevents vanishing / exploding gradients at model start. |
| **Dropout (rate = 0.3)** | Randomly disables 30% of neurons per training step, preventing co-adaptation and overfitting. |
| **`class_weight='balanced'`** | Corrects the 5.2× imbalance without resampling — penalizes missed churners proportionally more during training. |

---

## 👥 Author

<table>
  <tr>
    <td align="center">
      <b>🌿 Greenland Team</b><br/><br/>
      Faculty of Artificial Intelligence<br/>
      Kafrelsheikh University, Egypt<br/><br/>
      <i>Neural Networks Course Project</i>
    </td>
  </tr>
</table>

---

## 📜 License

This project is submitted for academic purposes at Kafrelsheikh University.
Feel free to use or adapt the code with proper attribution.

---

<p align="center">Made with ❤️ by the Greenland Team — Kafrelsheikh University</p>
