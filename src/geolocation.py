import pandas as pd


def ip_to_int(ip):
    if pd.isna(ip):
        return None

    ip = str(ip).strip()

    parts = ip.split(".")

    if len(parts) != 4:
        return None

    try:
        return (
            int(parts[0]) * 256**3 +
            int(parts[1]) * 256**2 +
            int(parts[2]) * 256 +
            int(parts[3])
        )
    except ValueError:
        return None


def convert_ip_column(df, ip_col="ip_address"):
    df = df.copy()

    df[ip_col] = df[ip_col].apply(ip_to_int)

    df = df.dropna(subset=[ip_col])

    return df


def map_ip_to_country(fraud_df, ip_map_df):

    fraud_df = fraud_df.copy()
    ip_map_df = ip_map_df.copy()

    fraud_df["ip_address"] = pd.to_numeric(
        fraud_df["ip_address"], errors="coerce")

    ip_map_df["lower_bound_ip_address"] = pd.to_numeric(
        ip_map_df["lower_bound_ip_address"], errors="coerce"
    )

    ip_map_df["upper_bound_ip_address"] = pd.to_numeric(
        ip_map_df["upper_bound_ip_address"], errors="coerce"
    )

    fraud_df = fraud_df.dropna(subset=["ip_address"])
    ip_map_df = ip_map_df.dropna(
        subset=["lower_bound_ip_address", "upper_bound_ip_address"]
    )

    fraud_df["ip_address"] = fraud_df["ip_address"].astype("int64")
    ip_map_df["lower_bound_ip_address"] = ip_map_df["lower_bound_ip_address"].astype(
        "int64")
    ip_map_df["upper_bound_ip_address"] = ip_map_df["upper_bound_ip_address"].astype(
        "int64")

    fraud_df = fraud_df.sort_values("ip_address")
    ip_map_df = ip_map_df.sort_values("lower_bound_ip_address")

    merged = pd.merge_asof(
        fraud_df,
        ip_map_df,
        left_on="ip_address",
        right_on="lower_bound_ip_address",
        direction="backward"
    )

    merged = merged[
        merged["ip_address"] <= merged["upper_bound_ip_address"]
    ]

    return merged
