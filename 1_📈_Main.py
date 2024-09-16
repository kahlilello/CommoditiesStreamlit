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

# Create a sidebar
st.sidebar.success('Select a Page')

# Multiselect to select commodities
selected_commodities = st.multiselect("Select Commodities", commoditiesDf.columns[1:], default=commoditiesDf.columns[1:].tolist())

# Filter dataframe based on selected commodities
selected_commodities_df = commoditiesDf[['Date'] + selected_commodities]

# Date range slider
start_date = selected_commodities_df['Date'].min()
end_date = selected_commodities_df['Date'].max()
selected_start_date, selected_end_date = st.slider("Select Date Range", start_date, end_date, (start_date, end_date))

# Filter dataframe based on selected date range
selected_commodities_df = selected_commodities_df[(selected_commodities_df['Date'] >= selected_start_date) & (selected_commodities_df['Date'] <= selected_end_date)]

dates = pd.to_datetime(selected_commodities_df['Date'])

# Calculate average price for each selected commodity
average_prices = selected_commodities_df[selected_commodities].mean()
max_prices = selected_commodities_df[selected_commodities].max()
min_prices = selected_commodities_df[selected_commodities].min()


# Display average prices above the figure
if len(selected_commodities) > 1:
    # Display metrics in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.write("<h3>Summary</h3>", unsafe_allow_html=True)
        for commodity in selected_commodities[:len(selected_commodities)//2]:
            st.write(f"<b>{commodity}</b> Avg: {average_prices[commodity]:.2f} Max: {max_prices[commodity]:.2f} Min: {min_prices[commodity]:.2f}", unsafe_allow_html=True)
    with col2:
        st.write("<h3><br> </h3>", unsafe_allow_html=True)
        for commodity in selected_commodities[len(selected_commodities)//2:]:
            st.write(f"<b>{commodity}</b> Avg: {average_prices[commodity]:.2f} Max: {max_prices[commodity]:.2f} Min: {min_prices[commodity]:.2f}", unsafe_allow_html=True)
else:
    # Display metrics in one column
    st.write("<h3>Summary</h3>", unsafe_allow_html=True)
    for commodity in selected_commodities:
        st.write(f"<b>{commodity}</b>: Avg: {average_prices[commodity]:.2f} Max: {max_prices[commodity]:.2f}", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(12,6))

for commodity in selected_commodities:
    commodity_prices = selected_commodities_df[commodity].values

    # Ensure dates are sorted in ascending order
    dates_sorted, commodity_prices_sorted = zip(*sorted(zip(dates, commodity_prices)))

    # Plot the scatter points for selected commodity prices
    ax.plot(dates_sorted, commodity_prices_sorted, label=commodity)

    # Calculate the coefficients of the line of best fit (1st-degree polynomial)
    coefficients = np.polyfit(mdates.date2num(dates_sorted), commodity_prices_sorted, 1)

    # Create the line of best fit equation
    line_of_best_fit = np.poly1d(coefficients)

    # Plot the line of best fit within the selected date range
    ax.plot(dates_sorted, line_of_best_fit(mdates.date2num(dates_sorted)), linestyle='--')

# Set x-axis limits
ax.set_xlim(selected_start_date, selected_end_date)

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