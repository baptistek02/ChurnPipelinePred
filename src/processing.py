import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.utils import detect_components, detect_cluster_column

def load_and_clean(file_path):
    """
    Load a csv file and clean it up
    :param file_path:
    :return: cleaned dataframe
    """
    df = pd.read_csv("data/data.csv", sep=",")
    df = clean_na_duplicates(df)
    df = clean_outliers(df)
    df = encode(df)
    df = clear_features(df)

    return df

def clean_na_duplicates(df):
    df = df.copy()

    # CLean NA and duplicates
    df = df.dropna()
    df = df.drop_duplicates(subset=['customerID'], keep="last")
    df = df[df["TotalCharges"] != " "]

    return df

def clean_outliers(df):
    df = df.copy()

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

    return df

def encode(df):
    df = df.copy()

    # Encoding
    df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
    df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
    df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    df = pd.get_dummies(df,
                        columns=['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                 'DeviceProtection',
                                 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
                                 ], drop_first=False, dtype=int)
    return df

def clear_features(df):
    df = df.copy()

    # Clear and make features
    df = df.drop(columns=["customerID"])
    return df



def split_data(df, target="Churn"):
    """
    Split a dataframe into 2 sets : train set and test set
    Apply SMOTE on train test to make it balanced
    :param df: cleaned df
    :param target:
    :return: normalized features set (train, test) and target (train, test)
    """

    y = df[target]
    X = df.drop(columns=[target])

    try:
        X = X.drop(columns=detect_components(df))
    except Exception as e:
        print("Impossible to delete components cols:", e)

    try:
        X = pd.get_dummies(X, columns=[detect_cluster_column(df)], drop_first=True, dtype=int)
    except Exception as e:
        print("Impossible to get dummies for cluster cols:", e)

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




