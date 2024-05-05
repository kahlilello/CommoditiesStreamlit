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

    # Predict using the trained model for entire date range
    predicted_prices = mlp_model.predict(X_scaled)

    # Plotting 
    fig, axes = plt.subplots(4, 1, figsize=(12,20))

    # Plot the scatter points for prices and MLP predictions
    ax1 = axes[0]
    ax1.plot(dates_sorted, prices_sorted, label=f'{commodity} Prices')
    ax1.plot(dates_sorted, predicted_prices, color = 'orange', label = 'MLP (Model)')
    ax1.set_title(f"{commodity} Prices Over Time with MLP Prediction")
    ax1.set_ylabel('Price')
    ax1.set_xlabel('Date')
    ax1.legend()
    ax1.grid(True)

    # Calculate and display mean squared error
    mse = mean_squared_error(y, predicted_prices)
    st.markdown(f"<p style='font-size:18px;font-weight:bold;'>{commodity} Mean Squared Error: {mse:.2f}</p>", unsafe_allow_html = True)

    # Histogram and KDE
    ax2 = axes[1]
    sns.histplot(prices_sorted, kde=True, color="skyblue", ax=ax2)
    ax2.set_title(f"{commodity} Price Distribution (Historgram & KDE)")
    ax2.set_xlabel(f"{commodity} Price")
    ax2.set_ylabel("Frequency")

    # Box Plot
    ax3 = axes[2]
    sns.boxplot(x=prices_sorted, ax=ax3, orient='h', color='lightblue')
    ax3.set_title(f"{commodity} Price Distribution (Box Plot)")
    ax3.set_xlabel(f"{commodity} Price")

    # Heatmap
    ax4 = axes[3]
    numeric_dates = mdates.date2num(dates_sorted)
    data = np.vstack((numeric_dates, prices_sorted)).T
    correlation_matrix = np.corrcoef(data, rowvar=False)
    sns.heatmap(correlation_matrix, annot = True, cmap = "TlGnBu", ax = ax4, xticklabels=["Date", f"{commodity} Price"], yticklabels = ["Date", f"{commodity} Price"])
    ax4.set_title(f"{commodity} Price Correlation Heatmap")

    plt.tight_layout()

    return fig

# Get Minimum and Maximum date from the dataframe
min_date = copperDf['Date'].min()
max_date = copperDf['Date'].max()