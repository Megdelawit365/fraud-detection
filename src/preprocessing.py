import pandas as pd
from sklearn.preprocessing import (
    StandardScaler
)


def convert_datetime(
    df,
    columns
):

    for col in columns:
        df[col] = pd.to_datetime(
            df[col]
        )

    return df


def scale_columns(
    df,
    columns
):

    scaler = StandardScaler()

    df[columns] = scaler.fit_transform(
        df[columns]
    )

    return df


def encode_columns(
    df,
    columns
):

    return pd.get_dummies(
        df,
        columns=columns,
        drop_first=True
    )
