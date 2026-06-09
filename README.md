# Fraud Detection for E-commerce and Bank Transactions

This project focuses on building a robust fraud detection system for fintech applications using real-world e-commerce and credit card transaction data. The goal is to analyze transactional behavior, engineer meaningful features, and prepare a clean, model-ready dataset that can later be used for detecting fraudulent activity using machine learning techniques.

---

## Project Structure

```bash
fraud-detection/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/                       # Add to .gitignore
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned and feature-engineered data
├── notebooks/
│   ├── __init__.py
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   ├── shap-explainability.ipynb
│   └── README.md
├── src/
│   └── __init__.py
├── tests/
│   └── __init__.py
├── models/                     # Saved model artifacts
├── scripts/
│   ├── __init__.py
│   └── README.md
├── requirements.txt
├── README.md
└── .gitignore

```


---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Megdelawit365/fraud-detection
cd fraud-detection
```

### 2. Create a virtual environment  

```bash
python -m venv venv
```

### 2. Activate virtual environment  

Windows:  

```bash
venv\Scripts\activate
```

MAC/Linux: 

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```


---

## Project Overview

### Task 1: Data Analysis and Preprocessing

- Cleaned and preprocessed e-commerce and credit card transaction datasets  
- Engineered time-based, behavioral, and frequency features for fraud detection  
- Mapped IP addresses to countries using range-based geolocation lookup  
- Performed exploratory data analysis and identified key fraud patterns  
- Handled severe class imbalance and prepared data for modeling  
  
---

## Technologies Used

- Python – core programming language
- Pandas, NumPy – data manipulation
- Matplotlib, Seaborn – visualization
- Scikit-learn – preprocessing and modeling utilities
- imbalanced-learn – SMOTE and resampling techniques
- XGBoost – gradient boosting models
- SHAP – model explainability
- Git & GitHub – version control
