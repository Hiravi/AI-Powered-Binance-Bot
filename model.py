import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers
def pred(df):

    # Assuming df is your DataFrame containing the new data for prediction
    # Features (input)
    X_new = df[['open', 'high', 'low', 'close', 'volume']].values

    # Normalize the features using the same scaler as used during training
    scaler = StandardScaler()
    X_new_scaled = scaler.fit_transform(X_new)

    # Load the trained model
    model = keras.models.load_model('model_trained.h5')

    # Make predictions
    predictions = model.predict(X_new_scaled)

    result = predictions[0]
    return result # 0: down, 1:UP, 2:no change on price



