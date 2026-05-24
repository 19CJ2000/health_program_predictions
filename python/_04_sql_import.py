#####################################################
# DATA IMPORT FROM SQL
#####################################################

# ---------------------------
# Libraries
# ---------------------------
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

# Make function to load all tables from PostgreSQL into dataframes
def load_data():

    user = os.getenv("DB_USER")
    password = quote_plus(os.getenv("DB_PASSWORD"))
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db = quote_plus(os.getenv("DB_NAME"))

    # Connect to database
    engine = create_engine(
        f"postgresql://{user}:{password}@{host}:{port}/{db}"
    )    
    
    # Load tables
    outcome1_df  = pd.read_sql('SELECT * FROM outcome1_df', engine)
    outcome2_df  = pd.read_sql('SELECT * FROM outcome2_df', engine)
    patients     = pd.read_sql('SELECT * FROM patients', engine)
    appointments = pd.read_sql('SELECT * FROM appointments', engine)
    demand       = pd.read_sql('SELECT * FROM demand', engine)

    return outcome1_df, outcome2_df, patients, appointments, demand


# Call function and unpack results
outcome1_df, outcome2_df, patients, appointments, demand = load_data()

