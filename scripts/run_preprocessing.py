import pandas as pd

from src.utils import *
from src.preprocessing import *
from src.feature_engineering import *
from src.geolocation import *


def preprocess_fraud():

    fraud = load_data(
        "data/raw/Fraud_Data.csv"
    )

    ip = load_data(
        "data/raw/IpAddress_to_Country.csv"
    )

    fraud = remove_duplicates(
        fraud
    )

    fraud = fill_missing(
        fraud
    )

    fraud = convert_datetime(
        fraud,
        [
            "signup_time",
            "purchase_time"
        ]
    )

    fraud = (
        create_time_features(
            fraud
        )
    )

    fraud = (
        create_time_since_signup(
            fraud
        )
    )

    fraud = (
        create_transaction_frequency(
            fraud
        )
    )

    fraud = (
        map_ip_to_country(
            fraud,
            ip
        )
    )

    fraud.to_csv(
        "data/processed/fraud_clean.csv",
        index=False
    )

    print(
        "Fraud dataset saved"
    )


def preprocess_credit():

    credit = load_data(
        "data/raw/creditcard.csv"
    )

    credit = (
        remove_duplicates(
            credit
        )
    )

    credit = (
        fill_missing(
            credit
        )
    )

    credit.to_csv(
        "data/processed/creditcard_clean.csv",
        index=False
    )

    print(
        "Credit dataset saved"
    )


def main():

    preprocess_fraud()

    preprocess_credit()


if __name__ == "__main__":
    main()
