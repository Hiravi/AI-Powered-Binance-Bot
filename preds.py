import joblib
import xgboost as xgb
def pred(df):
    preprocessor = joblib.load('preprocessor.pkl')
    feat = preprocessor.transform(df)
    loaded_model = xgb.Booster(model_file='regression.bin')
    result = loaded_model.predict(xgb.DMatrix(feat))
    result = result[1]
    return result # 0: down, 1:UP, 2:no change on price



