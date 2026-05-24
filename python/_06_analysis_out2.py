#####################################################
# OUTCOME 2 ANALYSIS: THERAPIST DEMAND FORECASTING
#####################################################
# Goal:
#   Forecast next-month therapist demand using:
#   1. Seasonal naive baseline
#   2. SARIMAX
#   3. XGBoost regressor
#
# Key design choice:
#   Use time-aware train/test splitting to avoid temporal leakage.
#####################################################

# ---------------------------
# Libraries
# ---------------------------
from _04_sql_import import load_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Suppress SARIMAX warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=FutureWarning)


# ---------------------------
# Helper functions
# ---------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(y_true, y_pred, model_name):
    return {
        "model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred)
    }


# ---------------------------
# Step 1: Load data from PostgreSQL
# ---------------------------
outcome1_df, outcome2_df, patients, appointments, demand = load_data()

df = demand.copy()

# Quality check
print(df.head())
print(df.info())


# ---------------------------
# Step 2: Data cleaning / formatting
# ---------------------------
df["mon"] = pd.to_datetime(df["mon"])

df = df.sort_values(["prov", "urb_rur", "mon"]).reset_index(drop=True)

# Create region identifier for grouped time series
df["region"] = df["prov"] + "_" + df["urb_rur"]

# Optional calendar features for XGBoost
df["month"] = df["mon"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Confirm panel structure
print(
    df.groupby("region")
    .agg(
        n_months=("mon", "count"),
        first_month=("mon", "min"),
        last_month=("mon", "max")
    )
)


# ---------------------------
# Step 3: Time-aware train/test split
# ---------------------------
# Train on earlier months; test on future months.
# This avoids temporal leakage.

train_df = df[df["mon"] < "2026-01-01"].copy()
test_df = df[df["mon"] >= "2026-01-01"].copy()

print("\nTraining months:", train_df["mon"].min(), "to", train_df["mon"].max())
print("Testing months:", test_df["mon"].min(), "to", test_df["mon"].max())

print("Train rows:", train_df.shape[0])
print("Test rows:", test_df.shape[0])


# ---------------------------
# Step 4: Seasonal naive baseline
# ---------------------------
# Baseline prediction:
#   next-month demand is approximated using same-month demand from last year.
#
# In this simulated dataset, sess_lag12 is the same-month previous-year session count.
# This is a strong and appropriate benchmark for seasonal forecasting.

test_df["pred_naive"] = test_df["sess_lag12"]

naive_results = evaluate_model(
    y_true=test_df["pred_demand"],
    y_pred=test_df["pred_naive"],
    model_name="Seasonal Naive"
)

print("\nSeasonal Naive Results:")
print(naive_results)


# ---------------------------
# Step 5: SARIMAX model
# ---------------------------
# SARIMAX is fit separately for each region.
# This avoids forcing one single time-series model across unrelated regional panels.

sarimax_predictions = []

exog_vars = [
    "new_pt",
    "relaps_rate",
    "od_flag",
    "szn_flag"
]

for region in df["region"].unique():

    region_train = train_df[train_df["region"] == region].copy()
    region_test = test_df[test_df["region"] == region].copy()

    # Set datetime index and frequency for SARIMAX
    region_train = region_train.set_index("mon")
    region_train.index.freq = "MS"  # Monthly start frequency
    
    region_test = region_test.set_index("mon")
    region_test.index.freq = "MS"

    y_train = region_train["pred_demand"]
    y_test = region_test["pred_demand"]

    X_train_exog = region_train[exog_vars]
    X_test_exog = region_test[exog_vars]

    # SARIMAX order choices:
    #   order=(1,0,0): simple autoregressive component
    #   seasonal_order=(1,0,0,12): annual seasonality
    #
    # Dataset is small (only 12 months training), so this is intentionally conservative.
    # Removed differencing (d=0, D=0) since we have limited data
    model = SARIMAX(
        y_train,
        exog=X_train_exog,
        order=(1, 0, 0),
        seasonal_order=(1, 0, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fit = model.fit(disp=False)

    forecast = fit.forecast(
        steps=len(region_test),
        exog=X_test_exog
    )

    # Reset index for merging later
    region_preds = region_test.reset_index()[["mon", "pred_demand"]].copy()
    region_preds["region"] = region
    region_preds["prov"] = region.split("_")[0]
    region_preds["urb_rur"] = region.split("_")[1]
    region_preds["pred_sarimax"] = forecast.values

    sarimax_predictions.append(region_preds)

sarimax_df = pd.concat(sarimax_predictions, ignore_index=True)

sarimax_results = evaluate_model(
    y_true=sarimax_df["pred_demand"],
    y_pred=sarimax_df["pred_sarimax"],
    model_name="SARIMAX"
)

print("\nSARIMAX Results:")
print(sarimax_results)


# ---------------------------
# Step 6: XGBoost regressor 
# ---------------------------
# XGBoost uses lagged demand, regional identifiers, seasonality,
# and external demand drivers.

target = "pred_demand"

xgb_features = [
    "prov",
    "urb_rur",
    "new_pt",
    "relaps_rate",
    "od_flag",
    "szn_flag",
    "sess_lag1",
    "sess_lag2",
    "sess_lag12",
    "month_sin",
    "month_cos"
]

X_train = train_df[xgb_features]
y_train = train_df[target]

X_test = test_df[xgb_features]
y_test = test_df[target]

categorical_features = ["prov", "urb_rur"]

numeric_features = [
    "new_pt",
    "relaps_rate",
    "od_flag",
    "szn_flag",
    "sess_lag1",
    "sess_lag2",
    "sess_lag12",
    "month_sin",
    "month_cos"
]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    objective="reg:squarederror"
)

xgb_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", xgb)
])

xgb_model.fit(X_train, y_train)

test_df["pred_xgb"] = xgb_model.predict(X_test)

xgb_results = evaluate_model(
    y_true=y_test,
    y_pred=test_df["pred_xgb"],
    model_name="XGBoost"
)

print("\nXGBoost Results:")
print(xgb_results)


# ---------------------------
# Step 7: Compare model performance
# ---------------------------
results_df = pd.DataFrame([
    naive_results,
    sarimax_results,
    xgb_results
]).sort_values("RMSE")

print("\nModel Comparison:")
print(results_df)

results_df.to_csv("outputs/models/out2_model_comparison.csv", index=False)


# ---------------------------
# Step 8: Combined Prediction table
# ---------------------------
plot_df = test_df[
    ["region", "prov", "urb_rur", "mon", "pred_demand", "pred_naive", "pred_xgb"]
].merge(
    sarimax_df[["region", "mon", "pred_sarimax"]],
    on=["region", "mon"],
    how="left"
)

print("\n=== Sample Predictions ===")
print(plot_df.head())

plot_df.to_csv("outputs/models/out2_model_predictions.csv", index=False)


# ---------------------------
# Step 9: Faceted actual vs predicted demand plot
# ---------------------------
regions = sorted(plot_df["region"].unique())

fig, axes = plt.subplots(
    nrows=4,
    ncols=2,
    figsize=(16, 18),
    sharex=True,
    sharey=True
)

axes = axes.flatten()

for ax, region in zip(axes, regions):

    region_plot = plot_df[plot_df["region"] == region].copy()

    ax.plot(
        region_plot["mon"],
        region_plot["pred_demand"],
        marker="o",
        label="Actual",
        linewidth=2
    )

    ax.plot(
        region_plot["mon"],
        region_plot["pred_naive"],
        marker="s",
        label="Seasonal Naive",
        alpha=0.7
    )

    ax.plot(
        region_plot["mon"],
        region_plot["pred_sarimax"],
        marker="^",
        label="SARIMAX",
        alpha=0.7
    )

    ax.plot(
        region_plot["mon"],
        region_plot["pred_xgb"],
        marker="d",
        label="XGBoost",
        alpha=0.7
    )

    ax.set_title(region)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

# Shared labels
fig.suptitle("Therapist Demand Forecasting by Province and Region (Urban / Rural)", fontsize=16)
fig.supxlabel("Month")
fig.supylabel("Next-Month Therapist Sessions")

# One shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    frameon=False
)

plt.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.savefig("outputs/plots/out2_forcasting.png", dpi=150)
plt.close()


# ---------------------------
# Step 10: XGBoost feature importance
# ---------------------------
feature_names = xgb_model.named_steps["preprocess"].get_feature_names_out()
importances = xgb_model.named_steps["model"].feature_importances_

fi_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nXGBoost Feature Importance:")
print(fi_df)


plt.figure(figsize=(8, 5))
plt.barh(fi_df["feature"], fi_df["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("XGBoost Feature Importance: Therapist Demand Forecasting")
plt.tight_layout()
plt.savefig("outputs/plots/out2_feature_importance.png", dpi=150)
plt.close()
