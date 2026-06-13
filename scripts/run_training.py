import pandas as pd

from src.train import split_data, apply_smote
from src.models import (
    get_logistic_model,
    get_random_forest,
    get_xgboost
)

from src.evaluation import (
    evaluate_model
)


def prepare_features(df):

    # Remove columns not useful for training
    remove = [
        "signup_time",
        "purchase_time",
        "device_id"
    ]

    existing = [
        c
        for c in remove
        if c in df.columns
    ]

    df = df.drop(
        columns=existing
    )

    # Encode categories
    categorical = (
        df.select_dtypes(
            include=[
                "object",
                "string"
            ]
        ).columns
    )

    df = pd.get_dummies(
        df,
        columns=categorical,
        drop_first=True
    )

    # Replace infinities
    df = df.replace(
        [float("inf"), -float("inf")],
        pd.NA
    )

    # Fill missing values
    for col in df.columns:

        if col == "class":
            continue

        if (
            str(
                df[col].dtype
            )
            in [
                "float64",
                "int64",
                "bool"
            ]
        ):

            df[col] = (
                df[col]
                .fillna(
                    df[col]
                    .median()
                )
            )

    # Last safety check
    df = df.fillna(0)

    return df


def main():

    df = pd.read_csv(
        "data/processed/fraud_clean.csv"
    )

    print(
        "Dataset:",
        df.shape
    )

    df = (
        prepare_features(
            df
        )
    )

    print(
        "\nRemaining NaNs:"
    )

    print(
        df.isna()
        .sum()
        .sum()
    )

    target = "class"

    X_train, X_test, y_train, y_test = (
        split_data(
            df,
            target
        )
    )

    print(
        "\nBefore SMOTE"
    )

    print(
        y_train.value_counts()
    )

    X_train, y_train = (
        apply_smote(
            X_train,
            y_train
        )
    )

    print(
        "\nAfter SMOTE"
    )

    print(
        y_train.value_counts()
    )

    models = {

        "Logistic":
        get_logistic_model(),

        "Random Forest":
        get_random_forest(),

        "XGBoost":
        get_xgboost()

    }

    for name, model in models.items():

        print(
            f"\nTraining {name}"
        )

        model.fit(
            X_train,
            y_train
        )

        results = (
            evaluate_model(
                model,
                X_test,
                y_test
            )
        )

        print(
            results["report"]
        )

        print(
            "F1:",
            results["f1"]
        )

        print(
            "AUC-PR:",
            results["auc_pr"]
        )


if __name__ == "__main__":
    main()
