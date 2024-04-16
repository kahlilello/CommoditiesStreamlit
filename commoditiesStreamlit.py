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


# Set up Streamlit App
st.title('\tCommodities Prices Visualization')



dates = pd.to_datetime(commoditiesDf['Date'])
other_columns = commoditiesDf.columns[1:]

fig, ax = plt.subplots(figsize=(12,6))

for col in other_columns:

    commodity_prices = commoditiesDf[col].values


    # Ensure dates are sorted in ascending order
    dates_sorted, commodity_prices_sorted = zip(*sorted(zip(dates, commodity_prices)))

    # Plot the scatter points for gold prices
    ax.plot(dates_sorted, commodity_prices_sorted, label=col)

    # Calculate the coefficients of the line of best fit (1st-degree polynomial)
    coefficients = np.polyfit(mdates.date2num(dates_sorted), commodity_prices_sorted, 1)


    # Create the line of best fit equation
    line_of_best_fit = np.poly1d(coefficients)

    # Plot the line of best fit within the selected date range
    ax.plot(dates_sorted, line_of_best_fit(mdates.date2num(dates_sorted)), color='red', label='Line of Best Fit', linestyle='--')


# Set x-axis limits
ax.set_xlim(min(dates_sorted), max(dates_sorted))

# Display date format on x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate()  # Rotate date labels for better readability

ax.set_title("Commodities Prices Over Time")
ax.set_ylabel('Price')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True)


plt.tight_layout()

st.pyplot(fig)

