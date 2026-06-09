import pandas as pd

from src.utils import *
from src.preprocessing import *
from src.feature_engineering import *
from src.geolocation import *


def main():

    # Load data
    fraud = load_data("data/raw/Fraud_Data.csv")
    ip_map = load_data("data/raw/IpAddress_to_Country.csv")

    # Clean
    fraud = remove_duplicates(fraud)

    # Datetime conversion
    fraud = convert_datetime(fraud, ["signup_time", "purchase_time"])

    # Feature engineering
    fraud = create_time_features(fraud)
    fraud = create_time_since_signup(fraud)
    fraud = create_transaction_frequency(fraud)

    # IP conversion + geolocation
    fraud = convert_ip_column(fraud)
    fraud = map_ip_to_country(fraud, ip_map)

    # Save processed dataset
    fraud.to_csv("data/processed/fraud_clean.csv", index=False)

    print("Preprocessing complete. File saved to data/processed/")


if __name__ == "__main__":
    main()
