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
    