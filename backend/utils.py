import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_data(stock):
    df = yf.download(stock, period="1y")
    return df[['Close']]

def preprocess(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaler