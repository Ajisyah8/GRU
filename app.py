import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
import streamlit as st

# Fungsi untuk membuat plot
def plot_predictions(y_train_rescaled, train_pred_rescaled, y_test_rescaled, y_pred_rescaled):
    plt.figure(figsize=(25, 6))

    # Plot untuk data training
    plt.plot(y_train_rescaled, label='Actual Temperature (Train)', color='blue')
    plt.plot(train_pred_rescaled, label='Predicted Temperature (Train)', color='orange')

    # Plot untuk data testing
    plt.plot(np.arange(len(y_train_rescaled), len(y_train_rescaled) + len(y_test_rescaled)), y_test_rescaled, label='Actual Temperature (Test)', color='red')
    plt.plot(np.arange(len(train_pred_rescaled), len(train_pred_rescaled) + len(y_pred_rescaled)), y_pred_rescaled, label='Predicted Temperature (Test)', color='green')

    plt.legend()
    plt.title("Actual vs Predicted Temperature (Train and Test)")
    st.pyplot(plt)

# Judul aplikasi
st.title("Temperature Prediction Using GRU")

# Tab untuk dashboard
tabs = st.tabs(["📊 Data Overview", "⚙️ Configure & Train", "📈 Evaluation", "🔮 Future Prediction"])

with tabs[0]:
    st.header("📊 Data Overview")

    # Input data dari pengguna
    uploaded_file = st.file_uploader("Upload a CSV file with 'Tanggal' and 'Tavg' columns", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.info("Using default dataset")
        url = "https://raw.githubusercontent.com/Ajisyah8/Dataset/refs/heads/master/cleaned_temperature_data.csv"
        data = pd.read_csv(url)

    # Menampilkan dataset
    st.write("### Dataset")
    st.dataframe(data.head())

    # Preprocessing data
    data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d-%m-%Y')
    data.set_index('Tanggal', inplace=True)

    scaler = MinMaxScaler()
    data['Tavg'] = scaler.fit_transform(data[['Tavg']])

with tabs[1]:
    st.header("⚙️ Configure & Train")
    
    def create_sequences(data, time_steps=5):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps)])
            y.append(data[i + time_steps])
        return np.array(X), np.array(y)

    # Parameter dari pengguna
    time_steps = st.slider("Select time steps for sequence generation", 1, 30, 5)
    neurons = st.number_input("Number of neurons in GRU layer", min_value=1, max_value=256, value=32)
    batch_size = st.number_input("Batch size", min_value=1, max_value=256, value=64)
    epochs = st.number_input("Number of epochs", min_value=1, max_value=500, value=50)

    apply_training = st.button("Apply & Train Model")
    if apply_training:
        # Pembagian data
        train_start = "2022-01-01"
        train_end = "2023-05-27"
        test_start = "2023-05-28"
        test_end = "2023-12-31"

        train_data = data.loc[train_start:train_end]
        test_data = data.loc[test_start:test_end]

        X_train, y_train = create_sequences(train_data['Tavg'].values, time_steps)
        X_test, y_test = create_sequences(test_data['Tavg'].values, time_steps)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Model GRU
        model = Sequential([
            GRU(neurons, activation='tanh', return_sequences=False, input_shape=(time_steps, 1)),
            Dense(1)
        ])

        learning_rate = 0.001
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

        # Train model
        st.write("Training the model...")
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
        st.success("Training complete!")

with tabs[2]:
    st.header("📈 Evaluation")
    
    if apply_training:
        # Predictions
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        accuracy = (1 - mape) * 100

        # Denormalisasi hasil
        train_pred = model.predict(X_train)
        y_pred_rescaled = scaler.inverse_transform(y_pred)
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        train_pred_rescaled = scaler.inverse_transform(train_pred)
        y_train_rescaled = scaler.inverse_transform(y_train.reshape(-1, 1))

        # Menampilkan hasil evaluasi
        st.write("### Model Evaluation")
        st.write(f"MSE: {mse:.4f}")
        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"MAPE: {mape:.4f}")
        st.write(f"Accuracy: {accuracy:.2f}%")

        # Plot hasil
        st.write("### Plot: Actual vs Predicted")
        plot_predictions(y_train_rescaled, train_pred_rescaled, y_test_rescaled, y_pred_rescaled)
    else:
        st.warning("Please apply and train the model first!")

with tabs[3]:
    st.header("🔮 Future Prediction")
    
    if apply_training:
        # Prediksi masa depan
        last_input = X_test[-1]
        n_future = st.number_input("Number of future days to predict", min_value=1, max_value=30, value=7)
        future_predictions = []

        for _ in range(n_future):
            next_pred = model.predict(last_input.reshape(1, -1, last_input.shape[-1]))
            next_pred_rescaled = scaler.inverse_transform(next_pred)
            future_predictions.append(next_pred_rescaled.flatten()[0])
            last_input = np.append(last_input[1:], next_pred, axis=0)

        future_df = pd.DataFrame({
            'Day': [f'Day {i+1}' for i in range(n_future)],
            'Predicted Temperature': future_predictions
        })

        st.write("### Future Predictions")
        st.dataframe(future_df)
    else:
        st.warning("Please apply and train the model first!")
