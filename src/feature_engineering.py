def create_time_features(df):
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek

    return df


def create_time_since_signup(df):
    df["time_since_signup"] = (
        df["purchase_time"] -
        df["signup_time"]
    ).dt.total_seconds()

    return df


def create_transaction_frequency(df):
    transaction_count = (
        df.groupby("user_id")
        .size()
        .reset_index(name="transaction_count")
    )

    df = df.merge(
        transaction_count,
        on="user_id",
        how="left"
    )

    return df


def create_velocity_features(df):
    """
    Transaction velocity per user in short windows
    """

    df = df.sort_values(["user_id", "purchase_time"])

    df["txn_count_last_24h"] = df.groupby("user_id")["purchase_time"].transform(
        lambda x: x.rolling("24h").count()
    )

    return df
