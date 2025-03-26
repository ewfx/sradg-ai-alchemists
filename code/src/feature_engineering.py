import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load your dataset (replace with actual file or database)
def load_data(file_path):
    return pd.read_csv(file_path)

# Feature engineering function
def feature_engineering(df):
    # Example feature creation: difference between expected and actual values
    df["value_diff"] = df["expected_value"] - df["actual_value"]
    df["abs_value_diff"] = np.abs(df["value_diff"])

    # Create a ratio feature (avoid division by zero)
    df["value_ratio"] = np.where(df["expected_value"] != 0, 
                                 df["actual_value"] / df["expected_value"], 
                                 1)

    # Convert timestamps to numerical features
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Drop unnecessary columns
    df = df.drop(columns=["timestamp"])

    return df

# Anomaly detection using Isolation Forest
def detect_anomalies(df, contamination=0.05):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    df["anomaly_score"] = model.fit_predict(df_scaled)
    df["anomaly"] = df["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)

    return df

if __name__ == "__main__":
    # Load data
    file_path = "re_

