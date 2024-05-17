import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# Data Ingestion
data_path = "commodities_12_22.csv"
# Read the CSV file into a pandas dataframe
df = pd.read_csv(data_path)

# Data Cleaning
commoditiesDf = df.dropna(axis=0)

# Convert the 'Date' column to datetime objects
commoditiesDf['Date'] = pd.to_datetime(commoditiesDf['Date']).dt.date

# Gold Table
goldDf = commoditiesDf[['Date', 'Gold']]

# Set up Streamlit App
st.title('\tGold Prices Visualization and Prediction')


def visualize(goldDf, date_range):
    # Filter data based on selected date range
    goldDf_filtered = goldDf[(goldDf['Date'] >= date_range[0]) & (goldDf['Date'] <= date_range[1])]

    # Convert date strings to datetime objects
    dates = pd.to_datetime(goldDf_filtered['Date'])
    gold_prices = goldDf_filtered['Gold'].values

    # Ensure dates are sorted in ascending order
    dates_sorted, gold_prices_sorted = zip(*sorted(zip(dates, gold_prices)))

    # Split data into features and target
    X = np.array(mdates.date2num(dates_sorted)).reshape(-1, 1)
    y = gold_prices_sorted

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # MLP Regressor model
    mlp_model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', random_state=42)
    mlp_model.fit(X_train_scaled, y_train)

    # Predict using the trained models
    predicted_prices_train = mlp_model.predict(X_train_scaled)
    predicted_prices_test = mlp_model.predict(X_test_scaled)

    # Predict using the trained models for entire date range
    predicted_prices = mlp_model.predict(scaler.transform(X))

    # Plotting - combined visualizations
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))

    # Plot 1: Gold Prices with Line of Best Fit
    ax1 = axes[0]
    ax1.plot(dates_sorted, gold_prices_sorted, label='Gold Prices')
    coefficients = np.polyfit(mdates.date2num(dates_sorted), gold_prices_sorted, 1)
    line_of_best_fit = np.poly1d(coefficients)
    ax1.plot(dates_sorted, line_of_best_fit(mdates.date2num(dates_sorted)), color='red', label='Line of Best Fit', linestyle='--')
    ax1.set_title("Gold Prices Over Time")
    ax1.set_ylabel('Price')
    ax1.set_xlabel('Date')
    ax1.legend()
    ax1.grid(True)

    # Calculate average price
    average_price = np.mean(gold_prices_sorted)
    st.markdown(f"<p style='font-size:18px;font-weight:bold;'>Average Gold Price: ${average_price:.2f}</p>", unsafe_allow_html=True)

    # Plot 2: Histogram and KDE
    ax2 = axes[1]
    sns.histplot(gold_prices_sorted, kde=True,
