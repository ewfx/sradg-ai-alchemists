# __init__.py for the anomaly detection package

# Importing functions to make them easily accessible from the package
from .anomaly_detection import train_anomaly_model
from .feature_engineering import preprocess_data

# Optionally, you could also expose version information
__version__ = '1.0.0'

# Optionally, add any setup or initialization code if needed
def init():
    print("Bank Reconciliation Anomaly Detection package initialized")
