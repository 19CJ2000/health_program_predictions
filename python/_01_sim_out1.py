#####################################################
# DATA SIMULATION PART 1: OUTCOME 1 DATASET
#####################################################

# ---------------------------
# Libraries 
# ---------------------------
import numpy as np
import pandas as pd


# ---------------------------
# Step a) Patient-level variables
# ---------------------------
# Reproducibilitys
np.random.seed(42)  
n_patients = 5000

# Patient-level dataframe 
patients = pd.DataFrame({
    "patient_id": np.arange(1, n_patients + 1),   # unique patient IDs

    "sub_type": np.random.choice(["Dep", "Stim", "Opi", "Other"], size=n_patients), # random selection of any 4 levels

    "sym_severity": np.random.randint(0, 6, size=n_patients),  # random selection from 0–5 scale

    "car_aval": np.random.binomial(1, 0.55, size=n_patients),  # 55% have access
})
print(patients.head(10))



# ---------------------------
# Step b) Cancel_hist (latent prior behavior influenced by patient traits)
# ---------------------------
# Assign baseline effect per substance use group (sub_type)
sub_type_effect = {
    "Dep": 0.4,   # addicts of depressants and opioids more likely to have higher prior cancellation risk 
    "Opi": 0.3, 
    "Stim": -0.3, 
    "Other": 0.0}  

patients["sub_effect"] = patients["sub_type"].map(sub_type_effect) # add to pateints df_outcome1

# Linear predictor from patient-level variables
weights_latent = {
    "intercept": -1.5,
    "sym_severity": 0.3,
    "car_aval": -0.5,
}

lp_patient = (
    weights_latent["intercept"] +
    weights_latent["sym_severity"] * patients["sym_severity"] +
    weights_latent["car_aval"] * patients["car_aval"] +
    patients["sub_effect"]  # sub_type contribution
)

# Transform to [0,1] using sigmoid
mean_beta = 1 / (1 + np.exp(-lp_patient))

# Beta draw for stochastic cancel_hist
phi = 10   # for noise
alpha = mean_beta * phi
beta = (1 - mean_beta) * phi
patients["cancel_hist"] = np.random.beta(alpha, beta)

# Quality check
print(patients.head())
print(patients["cancel_hist"].describe())
print(patients.groupby("sub_type")["cancel_hist"].mean()) # mean cancel_hist by sub_type
print(patients.groupby("sym_severity")["cancel_hist"].mean()) # mean cancel_hist by sym_severity
print(patients.groupby("car_aval")["cancel_hist"].mean()) # mean cancel_hist by car_aval



# ---------------------------
# Step c) Appointment-level variables
# ---------------------------
# Appointment-level dataframe 
appointments = pd.DataFrame({
    "app_id": np.arange(1, n_patients + 1),

    "app_type": np.random.choice(["Intake", "Follow-up"], size=n_patients),

    "app_dist": np.round(np.minimum(np.random.exponential(scale=5, size=n_patients), 19.9) + 0.1, 2),

    "resp_rate": np.round(np.minimum(np.random.exponential(scale=12, size=n_patients), 71.5) + 0.5, 2),  # max 3 days (72 hours) for sim sake

    "app_time_match": np.random.binomial(1, 0.6, size=n_patients),  # 60% have an appointment booked at a time that they preffered 
})
print(appointments.head(60))
print(appointments["app_dist"].describe())



# ---------------------------
# Step d) Merge patient and appointment data
# ---------------------------
# Randomly assign patients to appointments 
appointments["patient_id"] = np.random.choice(patients["patient_id"], size=n_patients, replace=False)
df_outcome1 = pd.merge(appointments, patients, on="patient_id")
print(df_outcome1.head(20))



# ---------------------------
# Step e) # New Var: conf_rate (appointment confirmation rate via 2 calls, influenced by cancel_hist)
# ---------------------------
def assign_conf_rate(ch):
    if ch < 0.15:
        # Reliable patients → mostly confirm
        return np.random.choice([0.5, 1.0], p=[0.2, 0.8])
    
    elif ch < 0.40:
        # Middle group → mixed behavior
        return np.random.choice([0.0, 0.5, 1.0], p=[0.2, 0.5, 0.3])
    
    else:
        # High-risk patients → rarely confirm
        return np.random.choice([0.0, 0.5], p=[0.7, 0.3])

df_outcome1["conf_rate"] = df_outcome1["cancel_hist"].apply(assign_conf_rate)
print(df_outcome1.head(20))



# ---------------------------
# Step f) Outcome generation
# ---------------------------
# Assign baseline effect per appointment type group (app_type)
app_type_effect = {
    "Intake": 0.3,    # ~ 3% higher odds of cancellation if not a follow-up 
    "Follow-up": 0.0
}
df_outcome1["app_effect"] = df_outcome1["app_type"].map(app_type_effect) # add to pateints df_outcome1
print(df_outcome1.head(20))

# Logistic function: to determine cancel probability
# Weights for illustration; can be tuned
weights = {
    "intercept":       -1.5,   # baseline log-odds → ~18% cancellation probability when all predictors = 0
    "app_dist":        0.03,   # ~3% higher odds per of cancellation additional km (distance barrier)
    "app_time_match": -0.60,   # ~39% lower odds of cancellation if preferrable appointment times align
    "resp_rate":       0.01,   # ~1% higher odds of cancellation per additional hour delay (Longer response time = increased cancellation risk)
    "sym_severity":     0.2,   # ~22% higher odds of cancellation per 1-unit increase in severity (0–5 scale)
    "car_aval":        -0.5,   # ~39% lower odds of cancellation (access effect) if car is accessible 
    "cancel_hist":      1.4,   # ~15% higher odds of cancellation per 10 percentage point increase in prior no-show rate
    "conf_rate":       -0.8    # strong protective effect
}

# Linear predictor
lp = (
    weights["intercept"] +

    # main effects
    df_outcome1["app_effect"] +  # for app_type
    weights["app_dist"] * df_outcome1["app_dist"] +
    weights["app_time_match"] * df_outcome1["app_time_match"] +
    weights["resp_rate"] * df_outcome1["resp_rate"] +
    df_outcome1["sub_effect"] +  # for sub_type 
    weights["sym_severity"] * df_outcome1["sym_severity"] +
    weights["car_aval"] * df_outcome1["car_aval"] +
    weights["cancel_hist"] * df_outcome1["cancel_hist"] +
    weights["conf_rate"] * df_outcome1["conf_rate"] +

    # Interactions
    0.08 * df_outcome1["app_dist"] * (1 - df_outcome1["car_aval"]) +      # Interation 1) Distance × No car
    1.2 * (df_outcome1["app_dist"] > 3) * (1 - df_outcome1["car_aval"]) + #   set a threshold for explicit non-linearity in interaction 1
    0.9 * (df_outcome1["sym_severity"] >= 3) * (df_outcome1["app_time_match"] == 0) # Interaction 2) Symptom Severity x Time Alignment
)

# Logistic transformation
df_outcome1["prob_cancel"] = 1 / (1 + np.exp(-lp)) # the probability each appointment will eventually be cancelled

# Generate actual outcome
df_outcome1["cancelled"] = np.random.binomial(1, df_outcome1["prob_cancel"])

# Round only float columns
cols = ["cancel_hist","prob_cancel"]
df_outcome1[cols] = df_outcome1[cols].round(2)

# Reorder columns (and drop app_effect + sub_effect)
df_outcome1 = df_outcome1[
    [
        # appointment features
        "app_id",
        "app_type",
        "app_dist",
        "app_time_match",
        "conf_rate",
        "resp_rate",

        # patient features
        "patient_id",
        "sub_type",
        "sym_severity",
        "car_aval",
        "cancel_hist",

        # outcome 
        "cancelled"
    ]
]

# Check output
print(df_outcome1.head(20))
print(df_outcome1["cancelled"].value_counts(normalize=True)) # 1 = cancelled, 0 = attended


# Save as csv file 
df_outcome1.to_csv("data/df_outcome1.csv", index=False) # examine in SQL after rdm schema creation 
