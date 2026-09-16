# Detection of Fraud Cases for E-commerce and Bank Transactions

## Repository Structure

```bash
fraud-detection/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/                       
│   ├── raw/                   
│   └── processed/             
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
├── models/                     
├── scripts/
│   ├── __init__.py
│   └── README.md
├── requirements.txt
├── README.md
└── .gitignore
```

## Data Analysis and Preprocessing

This phase cleans the data, finds key patterns, and prepares both datasets (`Fraud_Data.csv` and `creditcard.csv`) for model training.

### Key Insights & Findings

* **Severe Class Imbalance:** Fraud is extremely rare in both datasets (less than 1% in Credit Card data and about 9% in E-Commerce data). 
* **Device vs. User Velocity:** Individual user accounts in `Fraud_Data.csv` only buy once. Looking at activity per **device** instead shows clear patterns of scammers using automated bots across multiple accounts on one machine.
* **Time Delta Signal:** Instant purchases (`time_delta == 0` right after signing up) are a massive indicator of fraud.

### Data Preparation Steps

1. **Feature Engineering:**
   * Created `time_delta` (time between account creation and purchase) and extracted hour/day features.
   * Calculated 1-hour transaction speeds per device (`tx_count_1h_device`) to spot quick automated purchase bursts.
2. **Data Transformation:**
   * **Scaling:** Scaled continuous numeric features using `StandardScaler`. Applied `RobustScaler` to transaction amounts in `creditcard.csv` to reduce the impact of severe outliers.
   * **Categorical Encoding:** Turned text categories (`source`, `browser`, `sex`) into 0s and 1s using One-Hot Encoding (`pd.get_dummies`).
3. **Handling Imbalance (SMOTE):**
   * Split the data into 80% training and 20% testing before balancing to keep the test set real and untouched.
   * Applied **SMOTE** only to the training set to make fake examples of fraud cases so the model learns both classes equally without deleting normal data.

All preprocessed training and testing files are saved in `data/processed/` for model building.