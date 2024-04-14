import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#matplotlib.use('TkAgg')  # Use the TkAgg backend (or another suitable backend)


import seaborn as sns
import numpy as np
from scipy import stats
from datetime import datetime
from datetime import date 
import streamlit as st


# Data Ingestion
data_path = "commodities_12_22.csv"
# Read the CSV file into a pandas dataframe
df = pd.read_csv(data_path)

# Data Cleaning 
commoditiesDf = df.dropna(axis=0)

#Gold Table
goldDf = commoditiesDf[['Date','Gold']]

# Convert date strings to datetime objects
dates = pd.to_datetime(goldDf['Date'])
gold_prices = goldDf['Gold'].values

# Ensure dates are sorted in ascending order
dates_sorted, gold_prices_sorted = zip(*sorted(zip(dates, gold_prices)))


# Assuming you have already imported your data and created `dates` and `gold_prices`
plt.figure(figsize=(12,6))

# Plot the scatter points for gold prices
plt.plot(dates, gold_prices, label='Gold Prices')

# Calculate the coefficients of the line of best fit (1st-degree polynomial)
coefficients = np.polyfit(mdates.date2num(dates), gold_prices, 1)

# Create the line of best fit equation
line_of_best_fit = np.poly1d(coefficients)

# Plot the line of best fit
plt.plot(dates, line_of_best_fit(mdates.date2num(dates)), color='red', label='Line of Best Fit', linestyle='--')


# Set x-axis limits
plt.xlim(min(dates_sorted), max(dates_sorted))

# Display date format on x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

plt.title("Gold Prices Over Time")
plt.ylabel('Price')
plt.xlabel('Date')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()