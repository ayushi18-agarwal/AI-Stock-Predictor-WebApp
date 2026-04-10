import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# LOAD DATA (MORE DATA)
data = yf.download("RELIANCE.NS", period="5y")

close_prices = data["Close"].values.reshape(-1, 1)

# SCALE
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices)

# CREATE DATASET
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# BETTER MODEL
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(60,1)))
model.add(LSTM(64))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# TRAIN (MORE EPOCHS)
model.fit(X_train, y_train, epochs=20, batch_size=16)

# SAVE
model.save("saved_model.h5")
joblib.dump(scaler, "scaler.pkl")

# METRICS
y_pred = model.predict(X_test)

y_pred_actual = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
r2 = r2_score(y_test_actual, y_pred_actual)

metrics = {
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("✅ Model + Metrics Saved")