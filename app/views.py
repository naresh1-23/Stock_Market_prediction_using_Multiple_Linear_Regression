import numpy as np
import pandas as pd
from model.Train import clean_data
from django.shortcuts import render
from model.Regression import DataScaler, MultiLinearRegression


def home(request):
    data = {}
    if request.method == "POST":
        # df_train = pd.read_csv("./OHLC.csv")
        # X_train = clean_data(df_train)
        # Y_train = df_train["Close"].values

        # # Initialize scaler and model
        scaler = DataScaler(np.array([728.11551831, 738.12126082, 719.44240641, 8927.88882468]),
                            np.array([1589.11921085, 1604.52712571, 1577.02344667, 46659.45166046]))
        # X_scaled = scaler.fit_transform(X_train)

        # model = MultiLinearRegression(learning_rate=0.01, num_iterations=1000)
        # model.fit(X_scaled, Y_train)

        # # Train predictions
        # y_train_predicted = model.predict(X_scaled[:5])
        # print("Training Predictions:")
        # print(y_train_predicted)
        # print("Actual Values:")
        # print(Y_train[:5])

        # # Load and clean test data
        # df_test = pd.read_csv("./2020-01-01.csv")
        # X_test = clean_data(df_test)
        # Y_test = df_test["Close"].values

        # # Transform test data using the same scaler
        # X_test_scaled = scaler.transform(X_test)

        # # Test predictions
        # y_predicted = model.predict(X_test_scaled)
        # print("Test Predictions:")
        # print(y_predicted[:5])
        # print("Actual Test Values:")
        # print(Y_test[:5])
        # Collect input values from the POST request
        open_val = float(request.POST["open"])
        high_val = float(request.POST["high"])
        low_val = float(request.POST["low"])
        vol_val = float(request.POST["Vol"])

        # Prepare the input as a 2D array
        X = np.array([[open_val, high_val, low_val, vol_val]])
        X_scaled = scaler.transform(X)
        # Define the trained model with precomputed weights and bias
        weights = np.array([5.03229441e+02, 5.49317695e+02, 5.38591693e+02,
                           2.85364985e-01])  # Replace with your weights
        bias = 729.2879487128386  # Replace with your bias
        model = MultiLinearRegression(weights=weights, bias=bias)

        # Use the model to make predictions
        result = model.predict(X_scaled)[0]  # Extract the single prediction

        # Prepare the response data
        data = {
            "open": open_val,
            "high": high_val,
            "low": low_val,
            "vol": vol_val,
            "result": result,
        }

    # Render the template with the data
    return render(request, "home.html", {"data": data})
