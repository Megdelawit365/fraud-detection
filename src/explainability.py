import shap
import matplotlib.pyplot as plt
import pandas as pd


def get_feature_importance(model, feature_names):

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .head(10)
    )

    return importance_df


def plot_feature_importance(importance_df):

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        importance_df["feature"],
        importance_df["importance"]
    )

    plt.gca().invert_yaxis()

    plt.title(
        "Top 10 Feature Importance"
    )

    plt.show()


def generate_shap_values(
    model,
    X_train,
    X_test
):

    explainer = (
        shap.TreeExplainer(
            model
        )
    )

    shap_values = (
        explainer.shap_values(
            X_test
        )
    )

    return explainer, shap_values


def plot_summary(
    shap_values,
    X_test
):

    shap.summary_plot(
        shap_values,
        X_test
    )


def force_plot(
    explainer,
    shap_values,
    X_test,
    index
):

    shap.initjs()

    return shap.force_plot(
        explainer.expected_value,
        shap_values[index],
        X_test.iloc[index]
    )
