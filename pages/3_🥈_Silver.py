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

# Silver Table
silverDf = commoditiesDf[['Date','Silver']]

# Set up Streamlit App
st.title('\tSilver Prices Visualization')

# Function visualize Silver Table w/ LoBF
def visualize(silverDf, date_range):
    # Filter data based on selected date range
    silverDf_filtered = silverDf[(silverDf['Date'] >= date_range[0]) & (silverDf['Date'] <= date_range[1])]

    # Convert date strings to datetime objects
    dates = pd.to_datetime(silverDf_filtered['Date'])
    silver_prices = silverDf_filtered['Silver'].values

    # Ensure dates are sorted in ascending order
    dates_sorted, silver_prices_sorted = zip(*sorted(zip(dates, silver_prices)))

    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(12, 20))

    # Plot the scatter points for silver prices and line of best fit
    ax1 = axes[0]
    ax1.plot(dates_sorted, silver_prices_sorted, label='Silver Prices')
    coefficients = np.polyfit(mdates.date2num(dates_sorted), silver_prices_sorted, 1)
    line_of_best_fit = np.poly1d(coefficients)
    ax1.plot(dates_sorted, line_of_best_fit(mdates.date2num(dates_sorted)), color='red', label='Line of Best Fit', linestyle='--')
    ax1.set_title("Silver Prices Over Time")
    ax1.set_ylabel('Price')
    ax1.set_xlabel('Date')
    ax1.legend()
    ax1.grid(True)

    # Calculate average price
    average_price = np.mean(silver_prices_sorted)
    st.markdown(f"<p style='font-size:18px;font-weight:bold;'>Average Silver Price: ${average_price:.2f}</p>", unsafe_allow_html=True)

    # Histogram and KDE
    ax2 = axes[1]
    sns.histplot(silver_prices_sorted, kde=True, color="skyblue", ax=ax2)
    ax2.set_title("Silver Price Distribution (Histogram & KDE)")
    ax2.set_xlabel("Silver Price")
    ax2.set_ylabel("Frequency")

    # Box Plot
    ax3 = axes[2]
    sns.boxplot(x=silver_prices_sorted, ax=ax3, orient='h', color='lightblue')
    ax3.set_title("Silver Price Distribution (Box Plot)")
    ax3.set_xlabel("Silver Price")

    # Heatmap
    ax4 = axes[3]
    numeric_dates = mdates.date2num(dates_sorted)
    data = np.vstack((numeric_dates, silver_prices_sorted)).T
    correlation_matrix = np.corrcoef(data, rowvar=False)
    sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", ax=ax4, 
                xticklabels=["Date", "Silver Price"], yticklabels=["Date", "Silver Price"])
    ax4.set_title("Correlation Heatmap")

    plt.tight_layout()

    return fig

# Get minimum and maximum date from the dataframe
min_date = silverDf['Date'].min()
max_date = silverDf['Date'].max()

# Create a slider to select date range
date_range = st.slider('Select a date range', min_value=min_date, max_value=max_date, value=(min_date, max_date))

# Visualize the Silver Prices
fig = visualize(silverDf, date_range)
st.pyplot(fig)
