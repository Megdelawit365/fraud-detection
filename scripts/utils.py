import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def missing_values(df):
    return pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": round(df.isnull().mean() * 100, 2)
    }).sort_values("missing_count", ascending=False)


def duplicate_count(df):
    return df.duplicated().sum()


def remove_duplicates(df):
    return df.drop_duplicates()


def dataset_summary(df):
    return {
        "shape": df.shape,
        "columns": list(df.columns)
    }
