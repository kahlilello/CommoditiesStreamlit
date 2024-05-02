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

# Convert the 'Date' col to datetime obj
commoditiesDf['Date'] = pd.to_datetime(commoditiesDf['Date']).dt.date

# Copper Table
copperDf = commoditiesDf['Date', 'Copper']

# Natural Gas Table
naturalgasDf = commoditiesDf['Date', 'Natural Gas']


# Set up Streamlit App
st.title('\tCopper & Natural Gas Visualization')


# Function visualization Copper & Natural Gas w/ LoBF and MLP
def visualize(data, commodity, date_range):
    # Filter Data based on selected date range
    data_filtered = data[(data['Date'] >= date_range[0]) & (data['Date'] <= date_range[1])]

    # Convert date strings to datetime objects
    dates = pd.to_datetime(data_filtered['Date'])
    prices = data_filtered[commodity].values

    # Ensure dates are sorted in ascending order
    dates_sorted, prices_sorted = zip(*sorted(zip(dates, prices)))

    # Split data into features and targets
    X = np.array(mdates.date2num(dates_sorted)).reshape(-1, 1)
    y = prices_sorted

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # MLP Regressor model
    mlp_model = MLPRegressor(hidden_later_sizes=(100, 100), activation = 'relu', solver = 'adam', random_state = 42)
    mlp_model.fit(X_scaled, y)

    