# AI-Stock-Predictor-WebApp
AI-powered stock price prediction web app using LSTM, Flask, and real-time Yahoo Finance data with interactive dashboard.
# 📈 AI Stock Predictor Web App

An end-to-end Machine Learning + Web Application that predicts stock prices using LSTM and provides intelligent BUY/SELL/HOLD decisions.

---

## 🚀 Features

- 📊 Real-time stock data using Yahoo Finance API
- 🤖 LSTM Deep Learning model for time-series prediction
- 📉 Predict next-day stock price
- 💡 Smart decision system (BUY / SELL / HOLD)
- 📈 Interactive chart visualization (Chart.js)
- 🌐 Flask-based backend + Web frontend
- 📏 Model evaluation metrics (MAE, RMSE, R²)

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript, Chart.js  
- **Backend:** Flask (Python)  
- **Machine Learning:** TensorFlow / Keras (LSTM)  
- **Data Source:** Yahoo Finance API (yfinance)  
- **Libraries:** NumPy, Pandas, Scikit-learn  

---

## 📁 Project Structure
backend/
│── app.py
│── train.py
│── saved_model.h5
│── scaler.pkl
│── metrics.json


### 5️⃣ Open in Browser
http://127.0.0.1:5000/
frontend/
│── index.html


---

## 📊 Sample Output

- Current Price: ₹1346  
- Predicted Price: ₹1349  
- Decision: BUY  
- R² Score: 0.90  

---

## 🧠 Model Details

- Uses **LSTM (Long Short-Term Memory)** network
- Trained on historical stock closing prices
- Input sequence: last 60 days
- Output: next day prediction

---

## 📈 Evaluation Metrics

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score (Model Accuracy)**

---

## 🔥 Future Improvements

- Multi-day forecasting
- Live candlestick charts
- User authentication system
- Deploy on AWS / Render

---

## 👩‍💻 Author

Ayushi Agarwal

---

## ⭐ If you like this project, give it a star!
