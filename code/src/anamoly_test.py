import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values, encoding categorical variables, and scaling numerical features.
    """
    # Handle missing values for numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    # Select relevant features
    features = data[['expected_value', 'actual_value', 'transaction_type']].copy()

    # Encode categorical variables
    features['transaction_type'] = features['transaction_type'].astype('category').cat.codes

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features

def train_anomaly_model(data: pd.DataFrame, contamination: float = 0.05) -> IsolationForest:
    """
    Train Isolation Forest model for anomaly detection.
    """
    # Preprocess the data
    scaled_features = preprocess_data(data)

    # Adjust contamination level dynamically
    contamination = min(0.1, max(0.01, 50 / len(data)))  

    # Train model
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(scaled_features)

    return model

def detect_anomalies(model: IsolationForest, data: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalies in transaction data and return results with anomaly labels.
    """
    # Preprocess the data
    scaled_features = preprocess_data(data)

    # Predict anomalies (-1 = anomaly, 1 = normal)
    data['anomaly'] = model.predict(scaled_features)
    data['anomaly_label'] = data['anomaly'].map({-1: 'Anomaly', 1: 'Normal'})

    return data

def run_anomaly_detection(data: pd.DataFrame, output_csv="reconciliation_with_anomalies.csv"):
    """
    Run anomaly detection and save results to a CSV file.
    """
    # Train the model
    model = train_anomaly_model(data)

    # Detect anomalies
    results = detect_anomalies(model, data)

    # Save results to CSV
    results.to_csv(output_csv, index=False)
    print(f"Anomaly detection results saved to {output_csv}")

    return results
