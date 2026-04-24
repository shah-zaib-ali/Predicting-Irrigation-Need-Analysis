<div align="center">

# 🌾 IRRIGATION NEED PREDICTOR

**`Machine Learning · Gradient Boosting · Ensemble Pipeline`**

A fully automated end-to-end ML classification pipeline that predicts agricultural
irrigation needs using a powerful soft-vote ensemble of three state-of-the-art
gradient boosting frameworks — determining whether a field needs **High**, **Medium**,
or **Low** irrigation with **98.50% accuracy**.

![Python](https://img.shields.io/badge/Python-3.x-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![LightGBM](https://img.shields.io/badge/LightGBM-Ensemble-9B59B6?style=for-the-badge&logo=leaflet&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-FF6F00?style=for-the-badge&logo=xgboost&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-Ensemble-FFD700?style=for-the-badge&logo=yandex&logoColor=black)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

</div>

---

## 🔥 Key Features

#### 🤝 Triple Gradient Boosting Ensemble

![LGB](https://img.shields.io/badge/Model%201-LightGBM-9B59B6?style=for-the-badge&logo=leaflet&logoColor=white)
![XGB](https://img.shields.io/badge/Model%202-XGBoost-FF6F00?style=for-the-badge&logo=xgboost&logoColor=white)
![CAT](https://img.shields.io/badge/Model%203-CatBoost-FFD700?style=for-the-badge&logo=yandex&logoColor=black)

- 🧬 Combines **LightGBM**, **XGBoost**, and **CatBoost** via **soft-vote probability averaging**
- 📊 Trained across **5-Fold Stratified Cross-Validation** with out-of-fold (OOF) predictions
- 🎯 Final OOF Accuracy: **0.984995** on 630,000 training rows

#### 🤖 Fully Automated Pipeline

![AutoDetect](https://img.shields.io/badge/Auto-Target%20Detection-27AE60?style=for-the-badge&logo=python&logoColor=white)
![NoCode](https://img.shields.io/badge/Config-Zero%20Manual%20Setup-3670A0?style=for-the-badge&logo=python&logoColor=white)

- 🔍 **Auto-detects** target and ID columns from any Kaggle-style CSV — no hardcoding needed
- 🔄 Drop in any dataset, run the script, get a submission file

#### 🛡️ Comprehensive Baseline Evaluation

![Baselines](https://img.shields.io/badge/Baselines-6%20Models-E74C3C?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LOOCV](https://img.shields.io/badge/Evaluation-LOOCV%20%2B%205--Fold-2ECC71?style=for-the-badge&logo=scikit-learn&logoColor=white)

- 📐 Evaluates **Logistic Regression**, **Decision Tree** (depth search), **Naive Bayes**, **K-Means augmentation**, and **Random Forest** before ensembling
- 🔁 **Leave-One-Out CV** applied on a stratified 200-sample subset for unbiased error estimation

#### 📊 Auto-Generated Diagnostic Plots

![CM](https://img.shields.io/badge/Plot-Confusion%20Matrices-1572B6?style=for-the-badge&logo=matplotlib&logoColor=white)
![FI](https://img.shields.io/badge/Plot-Feature%20Importance-E67E22?style=for-the-badge&logo=matplotlib&logoColor=white)
![MC](https://img.shields.io/badge/Plot-Model%20Comparison-8E44AD?style=for-the-badge&logo=matplotlib&logoColor=white)

- 🖼️ **Confusion matrices** for all 5 models in a single 2×3 subplot grid
- 📈 **LightGBM feature importance** chart (top 30 features)
- 📉 **Model comparison** bar chart — ranked by CV accuracy

---

## 🛠️ Tech Stack

### 🖥️ Language
![Python](https://img.shields.io/badge/Python-3.x-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

### 🤖 Ensemble Models
![LightGBM](https://img.shields.io/badge/LightGBM-9B59B6?style=for-the-badge&logo=leaflet&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6F00?style=for-the-badge&logo=xgboost&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFD700?style=for-the-badge&logo=yandex&logoColor=black)

### 📐 Baseline Models
![sklearn](https://img.shields.io/badge/Scikit--Learn-Baselines-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![KMeans](https://img.shields.io/badge/KMeans-Feature%20Augmentation-2ECC71?style=for-the-badge&logo=scikit-learn&logoColor=white)

### 🔢 Preprocessing
![LabelEncoder](https://img.shields.io/badge/Encoding-LabelEncoder-3670A0?style=for-the-badge&logo=python&logoColor=white)
![Scaler](https://img.shields.io/badge/Scaling-StandardScaler-013243?style=for-the-badge&logo=numpy&logoColor=white)

### 📊 Data & Visualization
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-E34F26?style=for-the-badge&logo=matplotlib&logoColor=white)

---

## 📊 Results

### Model Comparison

| # | Model | CV Accuracy |
|---|-------|-------------|
| 🥇 | LGB + XGB + CAT Ensemble | **0.9850** |
| 🥈 | Random Forest (300 trees) | 0.9852 |
| 🥉 | Decision Tree (depth=8) | 0.9851 |
| 4 | Decision Tree + KMeans | 0.9850 |
| 5 | Naive Bayes | 0.8271 |
| 6 | Logistic Regression | 0.7488 |

### Classification Report (OOF Ensemble — 630,000 rows)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| High | 0.9698 | 0.9156 | **0.9419** | 21,009 |
| Low | 0.9857 | 0.9953 | **0.9905** | 369,917 |
| Medium | 0.9851 | 0.9752 | **0.9801** | 239,074 |
| **Weighted Avg** | **0.9850** | **0.9850** | **0.9849** | **630,000** |

### Per-Fold Ensemble Accuracy

| Fold | LightGBM | XGBoost | CatBoost |
|------|----------|---------|----------|
| 1 | 0.9843 | 0.9849 | 0.9852 |
| 2 | 0.9844 | 0.9847 | 0.9850 |
| 3 | 0.9845 | 0.9843 | 0.9837 |
| 4 | 0.9845 | 0.9850 | 0.9845 |
| 5 | 0.9848 | 0.9848 | 0.9846 |
| **OOF Mean** | **0.9845** | **0.9847** | **0.9846** |

---

## 🚀 Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/k240529shahzaibali/irrigation-need-predictor.git
cd irrigation-need-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Place your data files**
irrigation-need-predictor/
├── train.csv
├── test.csv
├── sample_submission.csv
└── solution.py
**4. Run the pipeline**
```bash
python kagglecode.py
```

<div align="center">

![Step 1](https://img.shields.io/badge/Step%201-Clone%20Repo-3670A0?style=for-the-badge&logo=git&logoColor=white)
![Step 2](https://img.shields.io/badge/Step%202-Install%20Deps-FF6F00?style=for-the-badge&logo=pypi&logoColor=white)
![Step 3](https://img.shields.io/badge/Step%203-Add%20Data-27AE60?style=for-the-badge&logo=files&logoColor=white)
![Step 4](https://img.shields.io/badge/Step%204-Run%20Pipeline-9B59B6?style=for-the-badge&logo=python&logoColor=white)

</div>

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `submission.csv` | Final Kaggle predictions (270,000 rows) |
| `plots/confusion_matrices.png` | 2×3 grid — 4 baselines + OOF ensemble |
| `plots/feature_importance.png` | Top-30 LightGBM feature importances |
| `plots/model_comparison.png` | All models ranked by CV accuracy |

---

## ⚙️ Requirements

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn
```

---

## 🌾 About the Dataset

![Train](https://img.shields.io/badge/Train-630%2C000%20rows-27AE60?style=for-the-badge)
![Test](https://img.shields.io/badge/Test-270%2C000%20rows-3670A0?style=for-the-badge)
![Features](https://img.shields.io/badge/Features-19-E67E22?style=for-the-badge)
![Classes](https://img.shields.io/badge/Classes-3-9B59B6?style=for-the-badge)

A large-scale agricultural dataset containing soil properties, environmental
measurements, crop information and geographic region data. The pipeline
auto-detects the target column `Irrigation_Need` and classifies each field
record into one of three categories:

<div align="center">

![High](https://img.shields.io/badge/Class-HIGH%20Irrigation-E74C3C?style=for-the-badge)
![Medium](https://img.shields.io/badge/Class-MEDIUM%20Irrigation-F39C12?style=for-the-badge)
![Low](https://img.shields.io/badge/Class-LOW%20Irrigation-27AE60?style=for-the-badge)

</div>

---

## 📄 License

![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)

Distributed under the **Apache 2.0 License**. See `LICENSE` for more information.

