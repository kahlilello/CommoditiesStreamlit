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

# Silver Table
silverDf = commoditiesDf[['Date','Silver']]

# Set up Streamlit App
st.title('\tSilver Prices Visualization')


# Function visualize Gold Table w/ LoBF
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


    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # Plot the scatter points for gold prices and line of best fit
    ax1 = axes[0]
    ax1.plot(dates_sorted, gold_prices_sorted, label='Gold Prices')
    ax1.plot(dates, predicted_prices, color='orange', label='MLP (Model)')
    ax1.set_title("Silver Prices Over Time")
    ax1.set_ylabel('Price') 
    ax1.set_xlabel('Date')
    ax1.legend()
    ax1.grid(True)

    # Calculate and display mean squared error
    mse_train = mean_squared_error(y_train, predicted_prices_train)
    mse_test = mean_squared_error(y_test, predicted_prices_test)
    st.markdown(f"<p style='font-size:18px;font-weight:bold;'>Mean Squared Error (Training): {mse_train:.2f}</p>", unsafe_allow_html=True)

    # Histogram and KDE
    ax2 = axes[1]
    sns.histplot(gold_prices_sorted, kde=True, color="skyblue", ax=ax2)
    ax2.set_title("Silver Price Distribution (Histogram & KDE)")
    ax2.set_xlabel("Silver Price")
    ax2.set_ylabel("Frequency")

    plt.tight_layout()

    return fig

# Get min and max date from dataframe
min_date = silverDf['Date'].min()
max_date = silverDf['Date'].max()

# Create a slider to select date range 
date_range = st.slider('Select a date range', min_value=min_date, max_value=max_date)

# Visualize the Silver Prices
fig = visualize(silverDf, date_range)
st.pyplot(fig)



st.markdown(f"<p style='font-size:24px;font-weight:bold;'> Machine Learning Silver </p>", unsafe_allow_html=True)

# Visualize the Gold Prices
fig2 = MLvisualize(silverDf, date_range)
st.pyplot(fig2)