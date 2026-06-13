import joblib

from src.models import get_xgboost

model = get_xgboost()

joblib.dump(
    model,
    "models/xgboost.pkl"
)

print("saved")
