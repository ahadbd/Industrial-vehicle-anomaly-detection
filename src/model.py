import pandas as pd
from sklearn.ensemble import IsolationForest
from preprocessing import load_data, preprocess_data

def train_isolation_forest(scaled_data, contamination=0.05, random_state=42):
    """
    Train Isolation Forest model on scaled data.
    contamination = expected proportion of anomalies
    """
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state
    )
    model.fit(scaled_data)
    return model

def detect_anomalies(model, scaled_data):
    """
    Predict anomalies: -1 = anomaly, 1 = normal
    """
    predictions = model.predict(scaled_data)
    return predictions

if __name__ == "__main__":
    # Load and preprocess
    df = load_data()
    scaled_data, scaler = preprocess_data(df)

    # Train model
    model = train_isolation_forest(scaled_data)

    # Detect anomalies
    df['anomaly'] = detect_anomalies(model, scaled_data)

    # Save results
    df.to_csv("data/anomaly_results.csv", index=False)
    print("Anomaly detection complete. Results saved to data/anomaly_results.csv")

    import pandas as pd

df = pd.read_csv("data/anomaly_results.csv")
print(df.shape)  # should be (1500, 7)
print(df['anomaly'].value_counts())