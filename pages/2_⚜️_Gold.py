import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from datetime import datetime
import streamlit as st

# Data Ingestion
data_path = "commodities_12_22.csv"
# Read the CSV file into a pandas dataframe
df = pd.read_csv(data_path)

# Data Cleaning 
commoditiesDf = df.dropna(axis=0)

# Convert the 'Date' column to datetime objects
commoditiesDf['Date'] = pd.to_datetime(commoditiesDf['Date']).dt.date

# Gold Table
goldDf = commoditiesDf[['Date','Gold']]

# Set up Streamlit App
st.title('\tGold Prices Visualization')

# Function visualize Gold Table w/ LoBF
def visualize(goldDf, date_range):
    # Filter data based on selected date range
    goldDf_filtered = goldDf[(goldDf['Date'] >= date_range[0]) & (goldDf['Date'] <= date_range[1])]

    # Convert date strings to datetime objects
    dates = pd.to_datetime(goldDf_filtered['Date'])
    gold_prices = goldDf_filtered['Gold'].values

    # Ensure dates are sorted in ascending order
    dates_sorted, gold_prices_sorted = zip(*sorted(zip(dates, gold_prices)))

    # Plotting
    fig, ax = plt.subplots(figsize=(12,6))

    # Plot the scatter points for gold prices
    ax.plot(dates_sorted, gold_prices_sorted, label='Gold Prices')

    # Calculate the coefficients of the line of best fit (1st-degree polynomial)
    coefficients = np.polyfit(mdates.date2num(dates_sorted), gold_prices_sorted, 1)

    # Create the line of best fit equation
    line_of_best_fit = np.poly1d(coefficients)

    # Plot the line of best fit within the selected date range
    ax.plot(dates_sorted, line_of_best_fit(mdates.date2num(dates_sorted)), color='red', label='Line of Best Fit', linestyle='--')

    # Set x-axis limits
    ax.set_xlim(min(dates_sorted), max(dates_sorted))

    # Display date format on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()  # Rotate date labels for better readability

    ax.set_title("Gold Prices Over Time")
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True)

    # Calculate average price
    average_price = np.mean(gold_prices_sorted)

    # Display average price above the slider
    st.markdown(f"<p style='font-size:18px;font-weight:bold;'>Average Gold Price: ${average_price:.2f}</p>", unsafe_allow_html=True)
    
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