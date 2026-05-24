# National Support Program (NSP): Appointment Cancellation Prediction and Therapist Demand Forecasting

## Project Overview

This project simulates and analyzes operational healthcare data for a hypothetical National Support Program (NSP). The objective is to demonstrate how healthcare organizations can use structured program data to support appointment management, patient access planning, and therapist workforce forecasting.

The project contains two applied analytics problems:

1. **Appointment cancellation prediction**
   - Predict whether a patient appointment will be cancelled using patient-level and appointment-level variables.

2. **Therapist demand forecasting**
   - Forecast next-month therapist session demand across provinces and urban/rural regions using historical utilization, intake trends, relapse burden, overdose alerts, and seasonality.

The project was designed as a portfolio project for healthcare analytics, biostatistics, epidemiology, CRO, pharma, biotech, hospital analytics, and research data roles.

---

# Technologies Used

- Python
- SQL (PostgreSQL)
- pandas
- numpy
- scikit-learn
- xgboost
- shap
- matplotlib
- sqlalchemy
- statsmodels
- python-dotenv
- VS Code
- pgAdmin 4

---

# Project Structure

```text
NSP Project/
│
├── python/
│   ├── _01_sim_out1.py
│   ├── _02_sim_out2.py
│   ├── _03_analysis_descrip.sql
│   ├── _04_sql_import.py
│   ├── _05_analysis_out1.py
│   └── _06_analysis_out2.py
│
├── data/
│   ├── df_outcome2.csv
│   └── df_outcome2.csv
│
├── outputs/
│   ├── models/
│   └── plots/
│
├── docs/
│   └── data_dictionary.csv
│
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

---

# Outcome 1: Appointment Cancellation Prediction

## Objective

Predict whether a patient appointment will be cancelled using simulated patient and appointment characteristics.

## Models
- L2-Regularized Logistic Regression
- XGBoost Classifier
- Calibrated XGBoost (Platt Scaling)

## Methods Demonstrated
- healthcare data simulation
- preprocessing pipelines
- one-hot encoding
- regularization
- classification modeling
- probability calibration
- SHAP explainability
- model evaluation

## Outputs
- model comparison table
- prediction probability dataset
- calibration diagnostics
- feature importance plots
- SHAP explainability plots

---

# Outcome 2: Therapist Demand Forecasting

## Objective

Forecast next-month therapist session demand across regional healthcare service areas.

## Models
- Seasonal Naive Forecast
- SARIMAX
- XGBoost Regressor

## Methods Demonstrated
- time-series forecasting
- lag feature engineering
- seasonal modeling
- healthcare operations forecasting
- machine learning regression
- forecasting evaluation

## Outputs
- model comparison table
- forecasting prediction dataset
- regional forecast plots
- feature importance plots

---

# Database Design

The project includes a relational PostgreSQL schema with:
- primary keys
- foreign keys
- validation checks
- duplicate checks
- missingness checks

---

# Reproducibility

Database credentials are stored locally using a `.env` file and excluded from GitHub using `.gitignore`.

Example `.env.example`:

```text
DB_USER=username
DB_PASSWORD=password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=database
```

---

# Notes

- All data are simulated for educational and portfolio purposes.
- Simulated variables were intentionally designed to introduce realistic operational noise and mild multicollinearity.
- `prob_cancel` was retained only during simulation validation and excluded from predictive modeling to avoid target leakage.
- The project emphasizes predictive modeling and operational analytics rather than causal inference.