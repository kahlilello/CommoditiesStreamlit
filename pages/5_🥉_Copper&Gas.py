import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import streamlit as st

# Data Ingestion
data_path = "commodities_12_22.csv"
# Read the CSV file into a pandas dataframe
df = pd.read_csv(data_path)

# Data Cleaning 
commoditiesDf = df.dropna(axis=0)

# Convert the 'Date' column to datetime objects
commoditiesDf['Date'] = pd.to_datetime(commoditiesDf['Date']).dt.date

# Copper Table
copperDf = commoditiesDf[['Date','Copper']]
# Natural Gas Table
naturalGasDf = commoditiesDf[['Date','Natural Gas']]

# Set up Streamlit App
st.title('\tOil Prices Visualization')

# Function visualize Oil Prices w/ LoBF
def visualize(data, commodity, date_range):
    # Filter data based on selected date range
    data_filtered = data[(data['Date'] >= date_range[0]) & (data['Date'] <= date_range[1])]

    # Convert date strings to datetime objects
    dates = pd.to_datetime(data_filtered['Date'])
    prices = data_filtered[commodity].values

    # Ensure dates are sorted in ascending order
    dates_sorted, prices_sorted = zip(*sorted(zip(dates, prices)))

    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(12, 20))

    # Plot the scatter points for prices and line of best fit
    ax1 = axes[0]
    ax1.plot(dates_sorted, prices_sorted, label=f'{commodity} Prices')
    coefficients = np.polyfit(mdates.date2num(dates_sorted), prices_sorted, 1)
    line_of_best_fit = np.poly1d(coefficients)
    ax1.plot(dates_sorted, line_of_best_fit(mdates.date2num(dates_sorted)), color='red', label='Line of Best Fit', linestyle='--')
    ax1.set_title(f"{commodity} Prices Over Time")
    ax1.set_ylabel('Price')
    ax1.set_xlabel('Date')
    ax1.legend()
    ax1.grid(True)

    # Calculate average price
    average_price = np.mean(prices_sorted)
    st.markdown(f"<p style='font-size:18px;font-weight:bold;'>Average {commodity} Price: ${average_price:.2f}</p>", unsafe_allow_html=True)

    # Histogram and KDE
    ax2 = axes[1]
    sns.histplot(prices_sorted, kde=True, color="skyblue", ax=ax2)
    ax2.set_title(f"{commodity} Price Distribution (Histogram & KDE)")
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
    sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", ax=ax4, 
                xticklabels=["Date", f"{commodity} Price"], yticklabels=["Date", f"{commodity} Price"])
    ax4.set_title(f"{commodity} Price Correlation Heatmap")

    plt.tight_layout()

    return fig

# Get minimum and maximum date from the dataframe
min_date = copperDf['Date'].min()
max_date = copperDf['Date'].max()

# Create a slider to select date range
date_range = st.slider('Select a date range', min_value=min_date, max_value=max_date, value=(min_date, max_date))

# Visualize the Copper Prices
fig_copper = visualize(copperDf, 'Copper', date_range)
st.pyplot(fig_copper)

# Visualize the Natural Gas Prices
fig_natural_gas = visualize(naturalGasDf, 'Natural Gas', date_range)
st.pyplot(fig_natural_gas)