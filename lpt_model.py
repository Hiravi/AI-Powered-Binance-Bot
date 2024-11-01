import joblib
import pandas as pd


def pred(df: pd.DataFrame) -> int:
    # Load the trained model
    try:
        model = joblib.load('LPT_1hour_gradient_boosting_model_V2_220124.joblib')
    except FileNotFoundError:
        raise Exception("Model file not found. Please check the path.")

    # Make predictions
    prediction = model.predict(df)

    return prediction[0]  # 0: down, 1: UP, 2: no change on price
