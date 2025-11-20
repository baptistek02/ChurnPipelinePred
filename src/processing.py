import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_clean(file_path):
    df = pd.read_csv(file_path, sep=",")

    # CLean NA and duplicates
    df = df.dropna()
    df = df.drop_duplicates(subset=['customerID'], keep="last")
    df = df[df["TotalCharges"] != " "]

    # Clean outliers
    Q1 = df["MonthlyCharges"].quantile(q=0.25)
    Q3 = df["MonthlyCharges"].quantile(q=0.75)
    QR = Q3 - Q1
    df = df[(df["MonthlyCharges"] >= Q1 - 1.5 * QR) & (df["MonthlyCharges"] <= Q3 + 1.5 * QR)]

    df["TotalCharges"] = df["TotalCharges"].astype("float64")
    Q1 = df["TotalCharges"].quantile(q=0.25)
    Q3 = df["TotalCharges"].quantile(q=0.75)
    QR = Q3 - Q1
    df = df[(df["TotalCharges"] >= Q1 - 1.5 * QR) & (df["TotalCharges"] <= Q3 + 1.5 * QR)]

    # Encoding
    df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
    df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
    df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df,
                        columns=['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                 'DeviceProtection',
                                 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
                                 ], drop_first=False, dtype=int)

    # Clear and make features
    df = df.drop(columns=["customerID"])

    return df

def split_data(df, target="Churn"):

    y = df[target]
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    X_train_normalized = (X_train - X_train_mean) / X_train_std
    X_test_normalized = (X_test - X_train_mean) / X_train_std

    return X_train_normalized, X_test_normalized, y_train, y_test




