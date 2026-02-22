import numpy as np
import pandas as pd

def generate_telematics_data(n_samples=1500, random_state=42):
    np.random.seed(random_state)

    # Normal operational behavior
    speed = np.random.normal(loc=20, scale=5, size=n_samples)
    speed = np.clip(speed, 0, 40)

    engine_rpm = np.random.normal(loc=1500, scale=300, size=n_samples)
    engine_rpm = np.clip(engine_rpm, 600, 2500)

    fuel_rate = np.random.normal(loc=15, scale=3, size=n_samples)
    fuel_rate = np.clip(fuel_rate, 5, 30)

    idle_time = np.random.exponential(scale=30, size=n_samples)
    idle_time = np.clip(idle_time, 0, 300)

    load_weight = np.random.normal(loc=10, scale=3, size=n_samples)
    load_weight = np.clip(load_weight, 0, 25)

    acceleration = np.random.normal(loc=0, scale=1.5, size=n_samples)

    data = pd.DataFrame({
        "speed": speed,
        "engine_rpm": engine_rpm,
        "fuel_rate": fuel_rate,
        "idle_time": idle_time,
        "load_weight": load_weight,
        "acceleration": acceleration
    })

    # Inject anomalies (5% abnormal behavior)
    n_anomalies = int(0.05 * n_samples)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)

    # High RPM + low speed anomaly
    data.loc[anomaly_indices, "engine_rpm"] = np.random.uniform(2200, 2500, n_anomalies)
    data.loc[anomaly_indices, "speed"] = np.random.uniform(0, 5, n_anomalies)
    data.loc[anomaly_indices, "fuel_rate"] = np.random.uniform(25, 35, n_anomalies)

    return data


if __name__ == "__main__":
    df = generate_telematics_data()
    df.to_csv("data/synthetic_telematics.csv", index=False)
    print("Dataset generated and saved to data/synthetic_telematics.csv")