import pandas as pd
import sklearn.model_selection as train_test_split
from sklearn.linear_model import LogisticRegression,DecisionTreeClassifier

DATA_PATH =r"data\credit_risk_dataset.csv"
MODEL_PATH ="models\\model.joblib"
REPORT_PATH = "reports\\comparison.csv"

FEATURES =["annual_income",
           "credit_score",
           "loan_ammount",
           "existing_debt",
           "late_payments",
           "previous_defaults"]



TARGET="credit_risk"
df=pd.read_csv(DATA_PATH)
print(df.head())

print("Rows in dataset:", len(df))

print("Risk Classes:", df [TARGET].unique())

X=df [FEATURES]

y=df [TARGET]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

models={LogisticRegression: LogisticRegression(),
        DecisionTreeClassifier: DecisionTreeClassifier()}
    
    
