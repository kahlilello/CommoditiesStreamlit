import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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


max_date = 
