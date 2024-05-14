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

    # Split data into features and target (for prediction section)
    X = np.array(mdates.date2num(dates_sorted)).reshape(-1, 1)
    y = gold_prices_sorted

    # MLP Regressor model (prediction section)
    # Removed for brevity - can be included if desired

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
    sns.histplot(gold_prices_sorted, kde=True, color="skyblue", ax=ax2)
    ax2.set_title("Gold Price Distribution (Histogram & KDE)")
    ax2.set_xlabel("Gold Price")
    ax2.set_ylabel("Frequency")

    # Plot 3: Box Plot
    ax3 = axes[2]
    sns.boxplot(x=gold_prices_sorted, ax=ax3, orient='h', color='lightblue')
    ax3.set_title("Gold Price Distribution (Box Plot)")
    ax3.set_xlabel("Gold Price")

    plt.tight_layout()

    return fig


# Get minimum and maximum date from the dataframe
min_date = goldDf['Date'].min()
max_date = goldDf['Date'].max()

# Create a slider to select date range
date_range = st.slider('Select a date range', min_value=min_date, max_value=max_date, value=(min_date, max_date))

# Visualize the Gold Prices
fig = visualize(goldDf, date_range)
st.pyplot(fig)

# Prediction section (commented out for brevity)
# st.subheader