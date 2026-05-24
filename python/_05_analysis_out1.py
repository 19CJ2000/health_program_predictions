#####################################################
# OUTCOME 1 ANALYSIS: APPOINTMENT CANCELLATION PREDICTION
#####################################################
# Goal:
#   Predicting patient appointment cancellation using:
#   1. Ridge Regression (L2) 
#   2. XGBoost classifier
#
# Key design choice:
#   Data simulation intentionally introduces mild multicollinearity meant to mimic reality. 
#     Should favor ML prediction models over inferential regression models 
#####################################################

# ---------------------------
# Libraries
# ---------------------------
from _04_sql_import import load_data

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt 
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


# ---------------------------
# Step 1) # Load Data (from data_import.py script)
# ---------------------------
outcome1_df, outcome2_df, patients, appointments, demand = load_data()

# Quality Check
print(outcome1_df.head(20))


# ---------------------------
# Step 2) Preprocessing
# ---------------------------
# Target
y = outcome1_df["cancelled"]

# Features (drop target + IDs + prob_cancel)
X = outcome1_df.drop(columns=["cancelled", "patient_id", "app_id"])

categorical_features = ["app_type", "sub_type"]

numeric_features = [
    "app_dist",
    "app_time_match",
    "conf_rate",
    "resp_rate",
    "sym_severity",
    "car_aval",
    "cancel_hist"
]

# One-hot encoding
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# Split into Train / Test  
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Quality check (should be identical)
print("Train outcome rate:", y_train.mean())
print("Test outcome rate:", y_test.mean())


# ---------------------------
# Step 3) Regularized Logsitic Regression
# ---------------------------
# Build Logistic Model (L2 regularized) 
log_reg = LogisticRegression(
    penalty="l2",   # ridge penalty  
    C=1.0,          # lower C = stronger regularization
    solver="lbfgs", # optimization algorithm standard for L2 logistic regression
    max_iter=2000   # allow enough iterations for convergence
)

# Set up pipeline
log_reg_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", log_reg)
])

# Train model 
log_reg_model.fit(X_train, y_train)

# Predictions 
y_pred_proba_lr = log_reg_model.predict_proba(X_test)[:, 1] # returns probability of class 1 (cancelled)
y_pred_lr = log_reg_model.predict(X_test) # converts probabilities into class labels

# Evaluation 
print("AUC:", roc_auc_score(y_test, y_pred_proba_lr)) # evaluates ranking ability across all thresholds (higher = better)
print("Log Loss:", log_loss(y_test, y_pred_proba_lr)) # evaluates probability calibration (lower = better) 
print("Accuracy:", accuracy_score(y_test, y_pred_lr)) # simple proportion correct

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr)) # provides precision (correctness among predicted positives), recall (ability to capture true positives, and F1 (balance of both)

# Interpretation check
feature_names = log_reg_model.named_steps["preprocess"].get_feature_names_out() # Get feature names after preprocessing / one-hot encoding 
coefs = log_reg_model.named_steps["model"].coef_[0] # extracts regression coefficients
coef_df = pd.DataFrame({     # combines feature names + coefficients
    "feature": feature_names,
    "coef": coefs
}).sort_values("coef", ascending=False) # sort from strongest positive to strongest negative effect

print(coef_df) 
# num__cancel_hist:     ~14% higher odds of cancellation per 0.1 increase in prior no-show rate
# cat__app_type_Intake: ~41% higher odds of cancellation for Intake vs Follow-up
# num__sym_severity:    ~17% higher odds of cancellation per 1-unit increase in severity
# cat__sub_type_Opi:    ~10% higher odds of cancellation for Opi vs reference group
# num__resp_rate:       ~1% higher odds of cancellation per 1-hour increase in response time
# num__app_dist:        ~1% higher odds of cancellation per 1 km increase in distance
# num__app_time_match:  ~5% lower odds of cancellation when appointment time is preferred (1 vs 0)
# cat__sub_type_Other:  ~27% lower odds of cancellation for Other vs reference group
# num__car_aval:        ~34% lower odds of cancellation when car is available (1 vs 0)
# cat__sub_type_Stim:   ~45% lower odds of cancellation for Stim vs reference group
# num__conf_rate:       ~53% lower odds of cancellation per 1-unit increase in confirmation rate (0 → 1)


# ---------------------------
# Step 4) XGBoost classifier
# ---------------------------
# Build XGB model 
xgb = XGBClassifier(
    n_estimators=300,      # 300 trees 
    learning_rate=0.05,    # slows learning (better generalization)
    max_depth=4,           # prevents overfitting 
    subsample=0.8,         # add robustness
    colsample_bytree=0.8,  # feature sampling per tree
    reg_lambda=1.0,        # L2 regularization
    random_state=42,       # reproducibility
    eval_metric="logloss", # proper probabilistic loss
    use_label_encoder=False
)

# Set up pipeline
xgb_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", xgb)
])

# Train model 
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = xgb_model.predict(X_test)


# ---------------------------
# Step 5: Calibration model + curves
# ---------------------------

calibrated_xgb = CalibratedClassifierCV(
    xgb_model.named_steps["model"],
    method="sigmoid",
    cv=5
)

calibrated_pipeline = Pipeline([
    ("preprocess", xgb_model.named_steps["preprocess"]),
    ("model", calibrated_xgb)
])

calibrated_pipeline.fit(X_train, y_train)
y_pred_proba_xgb_cal = calibrated_pipeline.predict_proba(X_test)[:, 1]

frac_pos_lr, mean_pred_lr = calibration_curve(y_test, y_pred_proba_lr, n_bins=10)
frac_pos_xgb, mean_pred_xgb = calibration_curve(y_test, y_pred_proba_xgb, n_bins=10)
frac_pos_xgb_cal, mean_pred_xgb_cal = calibration_curve(y_test, y_pred_proba_xgb_cal, n_bins=10)


# ---------------------------
# Step 6: Model comparison table
# ---------------------------

results_df = pd.DataFrame([
    {
        "model": "L2 Logistic Regression",
        "AUC": roc_auc_score(y_test, y_pred_proba_lr),
        "LogLoss": log_loss(y_test, y_pred_proba_lr),
        "Accuracy": accuracy_score(y_test, y_pred_lr),
        "Brier": brier_score_loss(y_test, y_pred_proba_lr)
    },
    {
        "model": "XGBoost",
        "AUC": roc_auc_score(y_test, y_pred_proba_xgb),
        "LogLoss": log_loss(y_test, y_pred_proba_xgb),
        "Accuracy": accuracy_score(y_test, y_pred_xgb),
        "Brier": brier_score_loss(y_test, y_pred_proba_xgb)
    },
    {
        "model": "XGBoost Calibrated",
        "AUC": roc_auc_score(y_test, y_pred_proba_xgb_cal),
        "LogLoss": log_loss(y_test, y_pred_proba_xgb_cal),
        "Accuracy": np.nan,
        "Brier": brier_score_loss(y_test, y_pred_proba_xgb_cal)
    }
])

print(results_df)
results_df.to_csv("outputs/models/out1_model_comparison.csv", index=False)


# ---------------------------
# Step 7: Combined Prediction table
# ---------------------------

predictions_df = pd.DataFrame({
    "actual": y_test.values,
    "pred_lr": y_pred_proba_lr,
    "pred_xgb": y_pred_proba_xgb,
    "pred_xgb_calibrated": y_pred_proba_xgb_cal
})

print(predictions_df.head())
predictions_df.to_csv("outputs/models/out1_model_predictions.csv", index=False)


# ---------------------------
# Step 8: Diagnostic plots
# ---------------------------

feature_names = xgb_model.named_steps["preprocess"].get_feature_names_out()
importances = xgb_model.named_steps["model"].feature_importances_

fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(fi_df["feature"], fi_df["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig("outputs/plots/out1_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

X_test_transformed = xgb_model.named_steps["preprocess"].transform(X_test)

if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

explainer = shap.TreeExplainer(xgb_model.named_steps["model"])
shap_vals = explainer.shap_values(X_test_transformed)

shap.summary_plot(
    shap_vals,
    X_test_transformed,
    feature_names=feature_names,
    show=False
)
plt.savefig("outputs/plots/out1_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

shap.dependence_plot(
    "num__sym_severity",
    shap_vals,
    X_test_transformed,
    feature_names=feature_names,
    show=False
)
plt.savefig("outputs/plots/out1_shap_symsev.png", dpi=150, bbox_inches="tight")
plt.close()


# ---------------------------
# Step 9: Calibration plot
# ---------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

axes[0].plot(mean_pred_lr, frac_pos_lr, marker="o", label="Logistic")
axes[0].plot(mean_pred_xgb, frac_pos_xgb, marker="o", label="XGBoost")
axes[0].plot([0, 1], [0, 1], linestyle="--", color="black")
axes[0].set_title("Logistic vs XGBoost")
axes[0].set_xlabel("Mean predicted probability")
axes[0].set_ylabel("True fraction of positives")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(mean_pred_lr, frac_pos_lr, marker="o", label="Logistic")
axes[1].plot(mean_pred_xgb_cal, frac_pos_xgb_cal, marker="o", label="XGBoost Calibrated")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="black")
axes[1].set_title("Logistic vs Calibrated XGB")
axes[1].set_xlabel("Mean predicted probability")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(mean_pred_xgb, frac_pos_xgb, marker="o", label="XGB Pre-Cal")
axes[2].plot(mean_pred_xgb_cal, frac_pos_xgb_cal, marker="o", label="XGB Calibrated")
axes[2].plot([0, 1], [0, 1], linestyle="--", color="black")
axes[2].set_title("XGB Pre vs Post Calibration")
axes[2].set_xlabel("Mean predicted probability")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/plots/out1_calibration.png", dpi=150, bbox_inches="tight")
plt.close()