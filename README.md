# Industrial Vehicle Anomaly Detection

## Project Overview
Simulates industrial port vehicle telematics and applies unsupervised anomaly detection to identify operational inefficiencies such as excessive idle time, high fuel consumption, and abnormal engine behavior.

## Dataset
- Synthetic telematics data
- Columns: speed, engine_rpm, fuel_rate, idle_time, load_weight, acceleration
- 1500 rows with 5% anomalies injected

## Methodology
1. Data generation & preprocessing
2. Feature scaling (StandardScaler)
3. Isolation Forest for anomaly detection
4. Visualization with Matplotlib

## Results
- Scatter plots highlight anomalies in:
  - Speed vs Engine RPM
  - Fuel Rate vs Idle Time
- Summary:
  - Total points: 1500
  - Anomalies detected: ~75 (5%)

## Technologies Used
- Python, Pandas, NumPy, Scikit-learn, Matplotlib, Jupyter Notebook

## Future Improvements
- Use real port vehicle telematics data
- Experiment with additional ML models
- Build interactive dashboard for anomaly monitoring