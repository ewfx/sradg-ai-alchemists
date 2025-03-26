import pandas as pd
import numpy as np
import logging
import os

# Set up logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def load_data(file_path):
    """
    Load CSV data into a pandas DataFrame.
    """
    try:
        def load_data(filename):
    file_path = os.path.join("raw", filename)
    # return pd.read_csv(file_path)
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def save_results(df, file_path):
    """
    Save DataFrame results to a CSV file.
    """
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Results saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
        raise

def handle_missing_values(df, strategy="mean"):
    """
    Handle missing values in a DataFrame.
    """
    try:
        if strategy == "mean":
            df.fillna(df.mean(), inplace=True)
        elif strategy == "median":
            df.fillna(df.median(), inplace=True)
        elif strategy == "mode":
            df.fillna(df.mode().iloc[0], inplace=True)
        elif strategy == "drop":
            df.dropna(inplace=True)
        else:
            logging.warning(f"Unknown strategy {strategy}, skipping missing value handling.")
        logging.info(f"Missing values handled using strategy: {strategy}")
        return df
    except Exception as e:
        logging.error(f"Error handling missing values: {str(e)}")
        raise

def scale_features(df, columns):
    """
    Normalize specified columns using min-max scaling.
    """
    try:
        for col in columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        logging.info(f"Features {columns} scaled using Min-Max normalization.")
        return df
    except Exception as e:
        logging.error(f"Error scaling features: {str(e)}")
        raise

def create_directory(directory):
    """
    Create a directory if it does not exist.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Directory created: {directory}")
        return directory
    except Exception as e:
        logging.error(f"Error creating directory: {str(e)}")
        raise

