from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    average_precision_score
)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "f1": f1_score(y_test, y_pred),
        "auc_pr": average_precision_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred)
    }
