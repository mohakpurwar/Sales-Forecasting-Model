from flask import Flask, render_template, request,url_for
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from io import BytesIO
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)



# Function to forecast sales using ARIMA and Random Forest
def forecast_sales(df):
    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract additional features like 'Day', 'Month', 'Year' from 'Date'
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfWeek'] = df['Date'].dt.weekday

    # One-hot encode categorical columns (Product Category, Store Location, Promotions, Weather)
    df = pd.get_dummies(df, columns=['Product Category', 'Store Location', 'Promotions', 'Weather'], drop_first=True)

    # Prepare the data
    X = df.drop(columns=['Sales', 'Date'])  # Dropping the target column 'Sales' and 'Date'
    y = df['Sales'].values

    # ARIMA Model
    arima_model = ARIMA(y, order=(5, 1, 0))
    arima_model_fit = arima_model.fit()

    # Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    # Make Predictions for each date
    arima_predictions = arima_model_fit.forecast(steps=len(df))
    rf_predictions = rf_model.predict(X)
    
    # Combine predictions: average of ARIMA and Random Forest
    combined_predictions = (arima_predictions + rf_predictions) / 2

    # Prepare the result for displaying in the table
    forecast_data = []
    for i in range(len(df)):
        forecast_data.append({
            "Date": df['Date'].iloc[i].strftime('%Y-%m-%d'),
            "Actual Sales": df['Sales'].iloc[i],
            "ARIMA Predictions": round(arima_predictions[i], 2),
            "Random Forest Predictions": round(rf_predictions[i], 2),
            "Combined Predictions": round(combined_predictions[i], 2)
        })

    return forecast_data

    

@app.route("/", methods=["GET", "POST"])
def index():
    forecast_data = []
    if request.method == "POST":
        file = request.files["file"]
        if file:

            # Read the uploaded Excel file
            df = pd.read_excel(file)
            
            # Get the forecasted data
            forecast_data = forecast_sales(df)
            
            
        
    return render_template("index.html", forecast_data=forecast_data)

           
    
if __name__ == "__main__":
    app.run(debug=True)
