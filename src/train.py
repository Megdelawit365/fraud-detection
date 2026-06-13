from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def split_data(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)
