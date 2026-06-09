import pandas as pd


def ip_to_int(ip):
    """
    Convert IPv4 string to integer
    """
    parts = ip.split(".")
    return (
        int(parts[0]) * 256**3 +
        int(parts[1]) * 256**2 +
        int(parts[2]) * 256 +
        int(parts[3])
    )


def convert_ip_column(df, ip_col="ip_address"):
    df[ip_col] = df[ip_col].apply(ip_to_int)
    return df


def map_ip_to_country(fraud_df, ip_map_df):
    """
    Range-based lookup using merge_asof logic
    """

    fraud_df = fraud_df.sort_values("ip_address")
    ip_map_df = ip_map_df.sort_values("lower_bound_ip_address")

    merged = pd.merge_asof(
        fraud_df,
        ip_map_df,
        left_on="ip_address",
        right_on="lower_bound_ip_address",
        direction="backward"
    )

    # filter valid range
    merged = merged[
        merged["ip_address"] <= merged["upper_bound_ip_address"]
    ]

    return merged
