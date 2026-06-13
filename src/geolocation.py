import pandas as pd


def map_ip_to_country(
    fraud_df,
    ip_df
):

    fraud_df = fraud_df.copy()
    ip_df = ip_df.copy()

    fraud_df["ip_address"] = pd.to_numeric(
        fraud_df["ip_address"],
        errors="coerce"
    )

    ip_df[
        "lower_bound_ip_address"
    ] = pd.to_numeric(
        ip_df[
            "lower_bound_ip_address"
        ],
        errors="coerce"
    )

    ip_df[
        "upper_bound_ip_address"
    ] = pd.to_numeric(
        ip_df[
            "upper_bound_ip_address"
        ],
        errors="coerce"
    )

    fraud_df = (
        fraud_df
        .dropna(
            subset=[
                "ip_address"
            ]
        )
    )

    ip_df = (
        ip_df
        .dropna()
    )

    fraud_df = (
        fraud_df
        .sort_values(
            "ip_address"
        )
    )

    ip_df = (
        ip_df
        .sort_values(
            "lower_bound_ip_address"
        )
    )

    merged = pd.merge_asof(
        fraud_df,
        ip_df,
        left_on="ip_address",
        right_on="lower_bound_ip_address",
        direction="backward"
    )

    return merged
