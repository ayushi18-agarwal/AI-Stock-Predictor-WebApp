from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import numpy as np
import joblib
import yfinance as yf
import os
import json
from tensorflow.keras.models import load_model

# -------------------------------
# PATH FIX (VERY IMPORTANT)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../fronthand"))

print("TEMPLATE PATH:", TEMPLATE_DIR)

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = load_model(os.path.join(BASE_DIR, "saved_model.h5"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# -------------------------------
# LOAD METRICS
# -------------------------------
metrics_path = os.path.join(BASE_DIR, "metrics.json")

if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)
else:
    metrics = {"mae": 0, "rmse": 0, "r2": 0}

# -------------------------------
# HOME ROUTE
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
# PREDICT ROUTE
# -------------------------------
@app.route("/predict")
def predict():
    stock = request.args.get("stock")

    if not stock:
        return jsonify({"error": "Enter stock symbol"})

    try:
        data = yf.download(stock, period="60d")

        if data is None or data.empty:
            return jsonify({"error": "Invalid stock"})

        if "Close" not in data.columns:
            return jsonify({"error": "Close data missing"})

        close_prices = data["Close"].values.reshape(-1, 1)

        if len(close_prices) < 60:
            return jsonify({"error": "Not enough data"})

        last_60 = close_prices[-60:]
        scaled = scaler.transform(last_60)

        X = scaled.reshape(1, 60, 1)

        pred = model.predict(X, verbose=0)[0][0]

        predicted_price = scaler.inverse_transform([[pred]])[0][0]
        current_price = close_prices[-1][0]

        decision = "BUY" if predicted_price > current_price else "SELL"

        return jsonify({
            "stock": stock,
            "current_price": float(current_price),
            "predicted_price": float(predicted_price),
            "decision": decision,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"]
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)})

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)