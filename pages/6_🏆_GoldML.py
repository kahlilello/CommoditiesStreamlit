import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Function to train SVR model and plot predictions
def train_and_plot_svr(dates, gold_prices, C, gamma, epsilon):
    plt.figure(figsize=(12,6))
    plt.plot(dates, gold_prices, label='Gold Prices')
    
    dates_num = mdates.date2num(dates).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(dates_num, gold_prices, test_size=0.2, random_state=42)
    
    model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
    model.fit(X_train, y_train)

    predicted_prices_train = model.predict(X_train)
    predicted_prices_test = model.predict(X_test)
    
    sort_indices_train = np.argsort(X_train.flatten())
    sort_indices_test = np.argsort(X_test.flatten())

    plt.plot(X_train[sort_indices_train], predicted_prices_train[sort_indices_train], color='green', label='Predicted Line (Training)')
    plt.plot(X_test[sort_indices_test], predicted_prices_test[sort_indices_test], color='blue', label='Predicted Line (Testing)')

    plt.title("Gold Prices Over Time (SVR)")
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.tight_layout()
    st.pyplot(plt)

    mse_train = mean_squared_error(y_train, predicted_prices_train)
    mse_test = mean_squared_error(y_test, predicted_prices_test)
    st.write(f"Mean Squared Error (Training): {mse_train}")
    st.write(f"Mean Squared Error (Testing): {mse_test}")

# Streamlit app
def main():
    st.title('Gold Price Prediction with SVR')
    st.sidebar.title('SVR Hyperparameters')
    
    # Load data
    # Assuming you have already imported your data and created `dates` and `gold_prices`
    
    # Slider for C
    C = st.sidebar.slider('C', 0.1, 10.0, 1.0)
    
    # Slider for gamma
    gamma = st.sidebar.slider('Gamma', 0.001, 1.0, 0.1)
    
    # Slider for epsilon
    epsilon = st.sidebar.slider('Epsilon', 0.1, 5.0, 1.0)
    
    # Train and plot SVR model
    train_and_plot_svr(dates, gold_prices, C, gamma, epsilon)

if __name__ == "__main__":
    main()
