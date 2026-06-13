import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def remove_duplicates(df):
    return df.drop_duplicates()


def missing_values(df):
    return pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": (
            df.isnull().mean() * 100
        ).round(2)
    })


def fill_missing(df):

    numeric = df.select_dtypes(
        include="number"
    ).columns

    categorical = df.select_dtypes(
        exclude="number"
    ).columns

    for col in numeric:
        df[col] = df[col].fillna(
            df[col].median()
        )

    for col in categorical:
        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

    return df
