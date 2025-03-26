import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_raw_data(filename):
    """
    Load raw data from 'data/raw/' directory.
    """
    file_path = os.path.join("data", "raw", filename)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"Raw data file {filename} not found in 'data/raw/' folder.")

def clean_data(df):
    """
    Perform basic data cleaning: handle missing values, remove duplicates, and correct data types.
    """
    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing values (example: filling missing numeric values with median)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Convert timestamp to datetime format
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df

def feature_engineering(df):
    """
    Create new features and encode categorical variables.
    """
    # Compute transaction differences
    df["amount_difference"] = df["expected_value"] - df["actual_value"]

    # Encode categorical variables
    label_encoder = LabelEncoder()
    for col in ["transaction_type", "currency", "status"]:
        df[col] = label_encoder.fit_transform(df[col])

    # Extract time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    return df

def scale_features(df):
    """
    Normalize numeric columns using StandardScaler.
    """
    scaler = StandardScaler()
    numeric_columns = ["expected_value", "actual_value", "amount_difference", "hour", "day_of_week"]
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

def process_data(input_file, output_file):
    """
    Complete pipeline: Load raw data, clean, engineer features, scale, and save processed data.
    """
    print("Loading raw data...")
    df = load_raw_data(input_file)

    print("Cleaning data...")
    df = clean_data(df)

    print("Performing feature engineering...")
    df = feature_engineering(df)

    print("Scaling features...")
    df = scale_features(df)

    # Ensure processed data directory exists
    os.makedirs("data/processed", exist_ok=True)

    # Save processed data
    output_path = os.path.join("data", "processed", output_file)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

# Example usage
if __name__ == "__main__":
    process_data("reconciliation_data.csv", "cleaned_transactions.csv")
