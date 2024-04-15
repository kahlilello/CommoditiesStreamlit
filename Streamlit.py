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

# Set up Streamlit App
st.title('\tGold Prices Visualization')



# Function visualize Gold Table w/ LoBF
def visualize(goldDf):
    # Convert date strings to datetime objects
    dates = pd.to_datetime(goldDf['Date'])
    gold_prices = goldDf['Gold'].values

    # Ensure dates are sorted in ascending order
    dates_sorted, gold_prices_sorted = zip(*sorted(zip(dates, gold_prices)))

    # Assuming you have already imported your data and created `dates` and `gold_prices`
    # Added ax this is apparently good practice
    fig, ax = plt.subplots(figsize=(12,6))

    # Plot the scatter points for gold prices
    ax.plot(dates_sorted, gold_prices_sorted, label='Gold Prices')

    # Calculate the coefficients of the line of best fit (1st-degree polynomial)
    coefficients = np.polyfit(mdates.date2num(dates_sorted), gold_prices_sorted, 1)

    # Create the line of best fit equation
    line_of_best_fit = np.poly1d(coefficients)

     # Plot the line of best fit
    ax.plot(dates_sorted, line_of_best_fit(mdates.date2num(dates_sorted)), color='red', label='Line of Best Fit', linestyle='--')


    # Set x-axis limits
    ax.set_xlim(min(dates_sorted), max(dates_sorted))

    # Display date format on x-axis
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

    # Display date format on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()  # Rotate date labels for better readability


    # plt.title("Gold Prices Over Time")
    # plt.ylabel('Price')
    # plt.xlabel('Date')
    # plt.legend()
    # plt.grid(True)


    ax.set_title("Gold Prices Over Time")
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True)


    plt.tight_layout()

    # Display the plot using Streamlit
    #st.pyplot()

    return fig

    

# Visualize the Gold Prices
fig = visualize(goldDf)
st.pyplot(fig)


# # Create the Slider
# max_date = pd.to_datetime(goldDf['Date']).max()
# min_date = pd.to_datetime(goldDf['Date']).min()

# values = st.slider(
#         "Gold Prices Date Range",
#         min_value = min_date,
#         max_value = max_date,
#         value = (min_date, max_date)
#     )

# # Filter Data based on selected date range
# filtered_goldDf = goldDf.loc[(goldDf['Date'] >= values[0]) & (goldDf['Date'] <= values[1])]

# # Display filtered data
# st.write(filtered_goldDf)

# Everything below is from the previous work, may not be needed

# gold_prices['Date'] = gold_prices.to_datetime(gold_prices["Date"]).dt.date
# gold_prices = gold_prices.loc[(gold_prices['Date'] >= values[0])] & (gold_prices['Date']<= values[1])

# # Creating column to distinguish month/year date
# gold_prices['MONTH'] = gold_prices.DatetimeIndex(gold_prices['Date']).month.map("{:02}".format).astype(str)
# gold_prices['YEAR'] = gold_prices.DatetimeIndex(gold_prices['Date']).year.map("{:02}".format).astype(str)
# gold_prices['MONTH_YEAR'] = gold_prices['YEAR'] + '-' + gold_prices['MONTH']
