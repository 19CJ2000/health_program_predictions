--------------------------------------------------
-- SECTION 1: CREATE RAW IMPORT TABLES
--------------------------------------------------
CREATE TABLE outcome1_df (
    app_id          INT,
    app_type        VARCHAR(20),
    app_dist        NUMERIC,
	app_time_match  INT,
    conf_rate       NUMERIC,
    resp_rate       NUMERIC,
    patient_id      INT,
    sub_type        VARCHAR(20),
    sym_severity    INT,
    car_aval        INT,
    cancel_hist     NUMERIC,
    cancelled       INT
); -- Then import data from local drive via GUI 

CREATE TABLE outcome2_df (
    prov            VARCHAR(10),
    urb_rur         VARCHAR(5),
    mon             DATE,
    new_pt          INT,
    relaps_rate     NUMERIC,
    od_flag         INT,
    szn_flag        INT,
    sess_lag1       INT,
    sess_lag2       INT,
    sess_lag12      INT,
    pred_demand     NUMERIC
); -- Then import data from local drive via GUI 


--------------------------------------------------
-- SECTION 2: DATA IMPORT
--------------------------------------------------
-- Import via pgAdmin GUI


--------------------------------------------------
-- SECTION 3: IMPORT VALIDATION
--------------------------------------------------
SELECT COUNT(*) FROM outcome1_df;
SELECT COUNT(*) FROM outcome2_df;


--------------------------------------------------
-- SECTION 4: DROP OLD RELATIONAL TABLES
--------------------------------------------------
DROP TABLE IF EXISTS appointments;
DROP TABLE IF EXISTS patients;
DROP TABLE IF EXISTS demand;


--------------------------------------------------
-- SECTION 5: CREATE RELATIONAL TABLES
--------------------------------------------------
CREATE TABLE patients (
    patient_id      INT PRIMARY KEY,
    sub_type        VARCHAR(20) NOT NULL,
    sym_severity    INT NOT NULL CHECK (sym_severity BETWEEN 0 AND 5),
    car_aval        INT NOT NULL CHECK (car_aval IN (0, 1)),
    cancel_hist     NUMERIC NOT NULL CHECK (cancel_hist BETWEEN 0 AND 1)
);


CREATE TABLE appointments (
    app_id          INT PRIMARY KEY,
    patient_id      INT NOT NULL REFERENCES patients(patient_id),
    app_type        VARCHAR(20) NOT NULL CHECK (app_type IN ('Intake', 'Follow-up')),
    app_dist        NUMERIC NOT NULL CHECK (app_dist >= 0),
    app_time_match  INT NOT NULL CHECK (app_time_match IN (0, 1)),
    conf_rate       NUMERIC NOT NULL CHECK (conf_rate BETWEEN 0 AND 1),
    resp_rate       NUMERIC NOT NULL CHECK (resp_rate >= 0),
    cancelled       INT NOT NULL CHECK (cancelled IN (0, 1))
);


CREATE TABLE demand (
    prov            VARCHAR(10) NOT NULL CHECK (prov IN ('BC', 'AB', 'ON', 'QC')),
    urb_rur         VARCHAR(5) NOT NULL CHECK (urb_rur IN ('U', 'R')),
    mon             DATE NOT NULL,
    new_pt          INT NOT NULL CHECK (new_pt >= 0),
    relaps_rate     NUMERIC NOT NULL CHECK (relaps_rate BETWEEN 0 AND 1),
    od_flag         INT NOT NULL CHECK (od_flag IN (0, 1)),
    szn_flag        INT NOT NULL CHECK (szn_flag IN (0, 1)),
    sess_lag1       INT NOT NULL CHECK (sess_lag1 >= 0),
    sess_lag2       INT NOT NULL CHECK (sess_lag2 >= 0),
    sess_lag12      INT NOT NULL CHECK (sess_lag12 >= 0),
    pred_demand     NUMERIC NOT NULL CHECK (pred_demand >= 0),
    PRIMARY KEY (prov, urb_rur, mon)  -- each province/urban-rural/month combination should only appear once.
);


--------------------------------------------------
-- SECTION 6: POPULATE RELATIONAL TABLES
--------------------------------------------------
INSERT INTO patients
SELECT DISTINCT
    patient_id, sub_type, sym_severity,
    car_aval, cancel_hist
FROM outcome1_df;


INSERT INTO appointments 
SELECT app_id, patient_id, app_type, app_dist, app_time_match,
    conf_rate, resp_rate, cancelled
FROM outcome1_df;


INSERT INTO demand
SELECT * 
FROM outcome2_df;


--------------------------------------------------
-- SECTION 7: DATA QUALITY CHECKS
--------------------------------------------------

-- Confirm Proper Population of New Relational Tables
SELECT * FROM patients;
SELECT * FROM appointments;
SELECT * FROM demand;


-- Row Counts 
SELECT COUNT(*) AS n_rows FROM outcome1_df;
SELECT COUNT(*) AS n_rows FROM outcome2_df;
SELECT COUNT(*) AS n_rows FROM patients;
SELECT COUNT(*) AS n_rows FROM appointments;
SELECT COUNT(*) AS n_rows FROM demand;


-- Duplication Checks
SELECT app_id, COUNT(*) AS n
FROM appointments
GROUP BY app_id
HAVING COUNT(*) > 1;

SELECT patient_id, COUNT(*) AS n
FROM patients
GROUP BY patient_id
HAVING COUNT(*) > 1;

SELECT prov, urb_rur, mon, COUNT(*) AS n
FROM demand
GROUP BY prov, urb_rur, mon
HAVING COUNT(*) > 1;


-- Missingness Checks 
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE patient_id IS NULL) AS missing_patient_id,
    COUNT(*) FILTER (WHERE app_type IS NULL) AS missing_app_type,
    COUNT(*) FILTER (WHERE cancelled IS NULL) AS missing_cancelled
FROM appointments;

SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE prov IS NULL) AS missing_prov,
    COUNT(*) FILTER (WHERE urb_rur IS NULL) AS missing_urb_rur,
    COUNT(*) FILTER (WHERE mon IS NULL) AS missing_month,
    COUNT(*) FILTER (WHERE pred_demand IS NULL) AS missing_pred_demand
FROM demand;


-- Outcome 1 Distribution Check
SELECT
    cancelled,
    COUNT(*) AS n,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM appointments
GROUP BY cancelled;


-- Outcome 2 Distribution Check
SELECT
    prov,
    urb_rur,
    COUNT(*) AS n_months,
    MIN(mon) AS first_month,
    MAX(mon) AS last_month
FROM demand
GROUP BY prov, urb_rur
ORDER BY prov, urb_rur;

