import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(filepath="data/synthetic_telematics.csv"):
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df):
    # Check missing values
    if df.isnull().sum().any():
        df = df.dropna()

    features = df.copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, scaler