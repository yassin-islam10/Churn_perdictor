from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR/"data"/"Telco-Customer-Churn.csv"

Target = "Churn"

cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod','SeniorCitizen']
num_cols = ['tenure','MonthlyCharges','TotalCharges']

def load_data(path = DATA_PATH):
    df = pd.read_csv(path)
    df = df.drop(columns=['customerID'], errors = "ignore")
    df["TotalCharges"]= pd.to_numeric(df["TotalCharges"], errors='coerce')
    if df["SeniorCitizen"].dtype != "object":
        df["SeniorCitizen"] = df["SeniorCitizen"].map({1:"Yes", 0:"No"})
    df[Target] = df[Target].map({"Yes":1, "No":0})
    return df

def split_xy(df):
    X = df.drop(Target, axis = 1)
    Y = df[Target]
    return X,Y

def get_preprocessor():
 num_pipe = Pipeline([
 ("imputer", SimpleImputer(strategy = "median")),
 ("Scalar", StandardScaler())
 ])
 cat_pipe = Pipeline([
 ("imputer", SimpleImputer(strategy = "most_frequent")),
 ("encoder", OneHotEncoder(handle_unknown = 'ignore'))  
 ])
 return ColumnTransformer([
 ("num", num_pipe, num_cols),
 ("cat", cat_pipe, cat_cols)

 ])


