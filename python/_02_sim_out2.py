#####################################################
# DATA SIMULATION PART 2: OUTCOME 2 DATASET
#####################################################

# ---------------------------
# Libraries 
# ---------------------------
import numpy as np
import pandas as pd


# ---------------------------
# Step a) province, urb/rur, month variables
# ---------------------------
# Reproducibility
np.random.seed(42)  

provinces = [ "BC", "AB", "ON", "QC"]

urban_rural = ["U", "R"]

months = pd.date_range("2021-01-01", "2026-12-01", freq="MS")
n_months = len(months)

# Skeletal dataframe 
geographic = pd.DataFrame([
    (prov, ur, m)
    for prov in provinces
    for ur in urban_rural
    for m in months
], columns=["prov", "urb_rur", "mon"])

geographic = geographic.sort_values(["prov", "urb_rur", "mon"]).reset_index(drop=True)

print(geographic.head(300))

# Assign baseline demand per province
prov_effect = {
    "BC": 0.08,
    "AB": -0.05, 
    "ON": 0.10, 
    "QC": 0.05
    }
geographic["prov_eff"] = geographic["prov"].map(prov_effect) # add to geographic df

# Assign baseline demand per urban/rural region
urb_effect = {
    "U": 0.15, 
    "R": -0.10
    }
geographic["urb_eff"] = geographic["urb_rur"].map(urb_effect) # add to geographic df 

print(geographic.head(60))



# ---------------------------
# Step c) New Var: Patient intake (trend + noise + regional scaling)
# ---------------------------
# Show system-wide growth in demand over time
base_trend = np.linspace(100, 160, n_months)

# Repeat monthly trend for each region (province x urb/rur)
geographic["new_pt"] = np.tile(base_trend, len(provinces) * len(urban_rural)) 

# Introduce region-time level variability via random noise
geographic["new_pt"] = geographic["new_pt"] * np.random.normal(1, 0.08, len(geographic)) 
print(geographic.head(60))



# ---------------------------
# Step d) New Var: Relapse rate (realistic baseline + regional structure)
# ---------------------------
# Baseline ~35% (aligned with literature ~40–60%, slightly conservative)
relapse_base = 0.35  

# Province-specific effects (ordered: BC > AB > ON > QC)
relapse_prov_effect = {
    "BC": 0.08,   # highest burden
    "AB": 0.05,
    "ON": 0.02,
    "QC": -0.02  
}

# Urban vs Rural effects (slightly higher in rural)
relapse_urb_effect = {
    "U": -0.015,
    "R": 0.015
}

# Generate relapse with structure + noise
geographic["relaps_rate"] = (
    relapse_base +
    geographic["prov"].map(relapse_prov_effect) +
    geographic["urb_rur"].map(relapse_urb_effect) +
    np.random.normal(0, 0.015, len(geographic))  # modest noise
)

# Clip to realistic bounds (avoid extreme tails)
geographic["relaps_rate"] = geographic["relaps_rate"].clip(0.25, 0.60)
print(geographic.head(60))
print(geographic.groupby("prov")["relaps_rate"].mean())
print(geographic.groupby("urb_rur")["relaps_rate"].mean())



# ---------------------------
# Step e) New vars: Flags
# ---------------------------
# Overdose alert flag (rare events that trigger spike in therapist demand)
geographic["od_flag"] = np.random.binomial(1, 0.05, len(geographic))

# Event flag (seasonality): Oct to Feb higher demands periods
geographic["szn_flag"] = geographic["mon"].dt.month.isin([10, 11, 12, 1, 2]).astype(int)
print(geographic.head(20))



# ---------------------------
# Step f) New Vars: sessions (time-dependent process) 
# ---------------------------
geographic = geographic.sort_values(["prov", "urb_rur", "mon"]) # sort
geographic["sessions"] = np.nan # insert blank session variable to fill 
print(geographic.head(60))


# Baseline starting demand per region
baseline_map = {
    ("BC", "U"): 140,
    ("BC", "R"): 95,
    ("AB", "U"): 110,
    ("AB", "R"): 75,
    ("ON", "U"): 150,
    ("ON", "R"): 100,
    ("QC", "U"): 135,
    ("QC", "R"): 90
}

# Loop over each region → simulate 8 independent time series
#   Must run entire script because of how pandas indexing works with .loc and the DataFrame state.
for (prov, ur), idx in geographic.groupby(["prov", "urb_rur"]).groups.items(): 
    
    idx = list(idx)                    # row indices for this region (ordered in time)
    base = baseline_map[(prov, ur)]    # region-specific starting demand
    
    for i, row in enumerate(idx):
        
        if i == 0:
            geographic.loc[row, "sessions"] = base   # t = 0 (initialize series)
        
        elif i == 1:
            # t = 1 (introduce slight variation before full lag structure kicks in)
            geographic.loc[row, "sessions"] = geographic.loc[idx[i-1], "sessions"] * 0.98
        
        else:
            # ---- Autoregressive structure ----
            lag1 = geographic.loc[idx[i-1], "sessions"]   # previous month demand (t-1)
            lag2 = geographic.loc[idx[i-2], "sessions"]   # two months prior (t-2)
            
            # ---- External/system drivers ----
            relapse = geographic.loc[row, "relaps_rate"]  # higher relapse → higher demand
            od_flag = geographic.loc[row, "od_flag"]      # overdose shock (binary spike)
            new_pt = geographic.loc[row, "new_pt"]        # new patient intake (growth driver)
            szn_flag = geographic.loc[row, "szn_flag"]    # seasonal indicator (Oct–Feb)
            
            # Random noise (unobserved variability)
            noise = np.random.normal(0, 6)
            
            # Core demand equation (before seasonality)
            base_value = (
                0.65 * lag1 +                           # strong short-term persistence
                0.20 * lag2 +                           # additional smoothing
                (1 + relapse) * 20 +                    # relapse-driven demand pressure
                18 * od_flag +                          # shock increase during OD alerts
                0.25 * new_pt +                         # capacity-adjusted demand (not a 1:1 conversion from patients filling out intake forms, to sessions)
                geographic.loc[row, "prov_eff"] * 50 +    # province-level effect
                geographic.loc[row, "urb_eff"] * 40 +     # urban vs rural effect
                noise                                   # stochastic variation
            )
            
            # Multiplicative seasonality: +10% demand in peak months
            season_multiplier = 1 + 0.10 * szn_flag
            
            value = base_value * season_multiplier      # scale demand during peak periods
            
            # Enforce minimum demand level (avoid unrealistic low values)
            geographic.loc[row, "sessions"] = max(value, 10)

print(geographic.head(60))



# ---------------------------
# Step g) New Vars: Lags 
# ---------------------------
geographic["sess_lag1"] = geographic.groupby(["prov", "urb_rur"])["sessions"].shift(1)
geographic["sess_lag2"] = geographic.groupby(["prov", "urb_rur"])["sessions"].shift(2)
geographic["sess_lag12"] = geographic.groupby(["prov", "urb_rur"])["sessions"].shift(12)



# ---------------------------
# Step h) New Var: Next month demand prediction (Final target variable)
# ---------------------------
geographic["pred_demand"] = geographic.groupby(["prov", "urb_rur"])["sessions"].shift(-1)

# ---------------------------
# Step i) Final Clean Up
# ---------------------------

df_outcome2 = geographic[[
    "prov", "urb_rur", "mon",
    "new_pt", "relaps_rate", "od_flag", "szn_flag",
    "sess_lag1", "sess_lag2", "sess_lag12",
    "pred_demand"
]].dropna().reset_index(drop=True)

# Round only float columns
cols = ["new_pt","relaps_rate","sess_lag1","sess_lag2","sess_lag12"]
df_outcome2[cols] = df_outcome2[cols].round(0).astype(int)
df_outcome2["pred_demand"] = df_outcome2["pred_demand"].round(2)

# Check output
print(df_outcome2.head(60))
print(df_outcome2.groupby("prov")["sess_lag1"].mean())    # mean sess_lag1 by province
print(df_outcome2.groupby("urb_rur")["sess_lag1"].mean()) # mean sess_lag1 by urb/rur
print(df_outcome2.groupby("prov")["new_pt"].mean())       # mean new_patient by province

# Save as csv file 
df_outcome2.to_csv("data/df_outcome2.csv", index=False) # examine in SQL after rdm schema creation 
