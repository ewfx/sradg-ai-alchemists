import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data before feeding it into the anomaly detection model.
    This includes handling missing values, encoding categorical variables,
    and scaling numerical features.

    Args:
    - data (pd.DataFrame): The input transaction data.

    Returns:
    - pd.DataFrame: Processed and scaled features ready for the anomaly detection model.
    """
    # Handle missing values (simple approach, filling NaN with median)
    data = data.fillna(data.median())

    # Feature engineering: Extract relevant columns (example: amount and transaction_type)
    features = data[['amount', 'transaction_type']]  # Add more relevant features here

    # Encode categorical variables (transaction_type)
    features['transaction_type'] = features['transaction_type'].astype('category').cat.codes

    # Scale the features (StandardScaler for normalization)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features

def train_anomaly_model(data: pd.DataFrame, contamination: float = 0.05) -> IsolationForest:
    """
    Train the Isolation Forest model to detect anomalies in transaction data.

    Args:
    - data (pd.DataFrame): The input transaction data.
    - contamination (float): The expected proportion of outliers (anomalies) in the data.

    Returns:
    - IsolationForest: The trained anomaly detection model.
    """
    # Preprocess the data (e.g., scale numerical features)
    scaled_features = preprocess_data(data)

    # Initialize the Isolation Forest model
    model = IsolationForest(contamination=contamination, random_state=42)

    # Fit the model to the scaled features
    model.fit(scaled_features)

    return model

def detect_anomalies(model: IsolationForest, data: pd.DataFrame) -> pd.DataFrame:
    """
    Use the trained model to detect anomalies in new transaction data.

    Args:
    - model (IsolationForest): The trained anomaly detection model.
    - data (pd.DataFrame): The input transaction data.

    Returns:
    - pd.DataFrame: Data with an 'anomaly' column indicating anomalies (-1 for anomaly, 1 for normal).
    """
    # Preprocess the data
    scaled_features = preprocess_data(data)

    # Predict anomalies (-1 indicates anomaly, 1 indicates normal)
    data['anomaly'] = model.predict(scaled_features)

    # Map -1 to 'Anomaly' and 1 to 'Normal'
    data['anomaly_label'] = data['anomaly'].map({-1: 'Anomaly', 1: 'Normal'})

    return data

def evaluate_anomalies(data: pd.DataFrame):
    """
    Evaluate the anomaly detection results. This could be used for further analysis
    or visualizations of the anomalies detected in the data.

    Args:
    - data (pd.DataFrame): Data with anomaly labels.
    """
    # Count the number of anomalies
    anomaly_count = data[data['anomaly_label'] == 'Anomaly'].shape[0]
    print(f"Detected {anomaly_count} anomalies out of {data.shape[0]} transactions.")

    # Optionally, display some anomaly examples
    print("\nSome Anomalous Transactions:")
    print(data[data['anomaly_label'] == 'Anomaly'].head())

def run_anomaly_detection(data: pd.DataFrame, contamination: float = 0.05):
    """
    Main function to run the anomaly detection pipeline.

    Args:
    - data (pd.DataFrame): The input transaction data.
    - contamination (float): The proportion of expected anomalies.
    """
    # Train the anomaly detection model
    model = train_anomaly_model(data, contamination)

    # Detect anomalies in the data
    results = detect_anomalies(model, data)

    # Evaluate and print results
    evaluate_anomalies(results)

    return results

