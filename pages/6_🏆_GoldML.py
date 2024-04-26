import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Sample data generation function
def generate_data(start_date, end_date):
    dates = []
    prices = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        price = 50 + np.random.randn() * 10
        prices.append(price)
        current_date += timedelta(days=1)
    return np.array(dates), np.array(prices)

# Function to train SVR model and plot predictions
def train_and_plot_svr(dates, gold_prices, C, gamma, epsilon):
    plt.figure(figsize=(12,6))
    plt.plot(dates, gold_prices, label='Gold Prices')
    
    # Generate indices for dates
    indices = np.arange(len(dates)).reshape(-1, 1)
    
    model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
    model.fit(indices, gold_prices)

    # Generate indices for all dates
    all_indices = np.arange(len(dates) + 365).reshape(-1, 1)
    
    # Predict prices for all dates
    predicted_prices_all = model.predict(all_indices)
    
    plt.plot(dates, predicted_prices_all[:len(dates)], color='red', label='Predicted Line')

    plt.title("Gold Prices Over Time (SVR)")
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.tight_layout()
    return plt

# Streamlit app
def main():
    st.title('Gold Price Prediction with SVR')
    st.sidebar.title('SVR Hyperparameters')
    
    # Generate sample data
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2022, 12, 31)
    dates, gold_prices = generate_data(start_date, end_date)
    
    # Slider for C
    C = st.sidebar.slider('C', 0.1, 10.0, 1.0)
    
    # Slider for gamma
    gamma = st.sidebar.slider('Gamma', 0.001, 1.0, 0.1)
    
    # Slider for epsilon
    epsilon = st.sidebar.slider('Epsilon', 0.1, 5.0, 1.0)
    
    # Train and plot SVR model
    plt = train_and_plot_svr(dates, gold_prices, C, gamma, epsilon)
    st.pyplot(plt)

    # Calculate and display MSE
    indices = np.arange(len(dates)).reshape(-1, 1)
    model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
    model.fit(indices, gold_prices)

    predicted_prices = model.predict(indices)
    mse = mean_squared_error(gold_prices, predicted_prices)
    
    st.write(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    main()