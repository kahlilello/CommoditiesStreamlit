import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
print(goldDf)

# Visualize Gold Table w/ LoBF

dates = goldDf['Date'].values
gold_prices = goldDf['Gold'].values


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

plt.title("Gold Prices Over Time")
plt.ylabel('Price')
plt.xlabel('Date')
plt.legend()
plt.grid(True)

# Format the x-axis to display dates
#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.tight_layout()
plt.show()