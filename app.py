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

# Title of the App
st.title("SITEMP")

# Function to plot Actual vs Predicted
def plot_predictions(y_train_rescaled, train_pred_rescaled, y_test_rescaled, y_pred_rescaled):
    plt.figure(figsize=(25, 6))

    # Plot for training data
    plt.plot(y_train_rescaled, label='Actual Temperature (Train)', color='blue')
    plt.plot(train_pred_rescaled, label='Predicted Temperature (Train)', color='orange')

    # Plot for testing data
    plt.plot(
        np.arange(len(y_train_rescaled), len(y_train_rescaled) + len(y_test_rescaled)),
        y_test_rescaled,
        label='Actual Temperature (Test)',
        color='red',
    )
    plt.plot(
        np.arange(len(train_pred_rescaled), len(train_pred_rescaled) + len(y_pred_rescaled)),
        y_pred_rescaled,
        label='Predicted Temperature (Test)',
        color='green',
    )

    plt.legend()
    plt.title("Actual vs Predicted Temperature (Train and Test)")
    st.pyplot(plt)

# User authentication function
users = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"},
}

def authenticate(username, password):
    if username in users and users[username]["password"] == password:
        return users[username]["role"]
    return None

# Login form
if "role" not in st.session_state:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        role = authenticate(username, password)
        if role:
            st.session_state["role"] = role
            st.success(f"Logged in as {role}")
        else:
            st.error("Invalid username or password")
else:
    role = st.session_state["role"]

    # Sidebar for logout
    st.sidebar.write(f"Logged in as: {role}")
    if st.sidebar.button("Logout"):
        keys_to_delete = ["role", "model", "scaler", "X_test", "y_test"]
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
        st.success("You have successfully logged out!")

    if "model_config" not in st.session_state:
        st.session_state["model_config"] = {
            "epochs": 50,
            "batch_size": 64,
            "neurons": 32,
            "time_steps": 5,
            "n_future": 7,
        }

    # Tabs for functionality based on role
    if role == "admin":
        tabs = st.tabs(["⚙️ Configure Parameters"])
    else:
        tabs = st.tabs(["📊 Data Overview", "📈 Evaluation", "🔮 Future Prediction"])

    # Admin - Configure Parameters Tab
    if role == "admin":
        with tabs[0]:
            st.header("⚙️ Configure Parameters")
            epochs = st.slider("Epochs", 1, 200, st.session_state["model_config"]["epochs"])
            batch_size = st.slider("Batch Size", 16, 128, st.session_state["model_config"]["batch_size"])
            neurons = st.slider("Neurons", 16, 128, st.session_state["model_config"]["neurons"])
            time_steps = st.slider("Time Steps", 1, 20, st.session_state["model_config"]["time_steps"])

            st.session_state["model_config"] = {
                "epochs": epochs,
                "batch_size": batch_size,
                "neurons": neurons,
                "time_steps": time_steps,
                "n_future": st.session_state["model_config"]["n_future"],
            }

    else:
        # Data Overview Tab
        with tabs[0]:
            st.header("📊 Data Overview")
            uploaded_file = st.file_uploader("Upload a CSV file with 'Tanggal' and 'Tavg' columns", type=["csv"])

            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    st.warning("Error decoding file. Trying with alternative encoding...")
                    uploaded_file.seek(0)
                    data = pd.read_csv(uploaded_file, encoding='latin1')

        # Validasi kolom
        expected_columns = ['Tanggal', 'Tavg']
        if all(column in data.columns for column in expected_columns):
            st.success("Dataset is valid!")
        else:
            st.error(f"Dataset harus memiliki kolom berikut: {expected_columns}")
            st.stop()
    else:
        st.info("Using default dataset")
        url = "https://raw.githubusercontent.com/Ajisyah8/Dataset/refs/heads/master/cleaned_temperature_data.csv"
        data = pd.read_csv(url)

            # Display the dataset
            st.write("### Dataset")
            st.dataframe(data.head(730), width=1200)

            # Preprocess data
            data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d-%m-%Y')
            data.set_index('Tanggal', inplace=True)
            data['Tavg'] = data['Tavg'].astype(str).replace(',', '.', regex=True).astype(float)

            scaler = MinMaxScaler()
            data['Tavg'] = scaler.fit_transform(data[['Tavg']])

            # Future prediction settings
            n_future = st.number_input("Number of future days to predict:", 1, 30, 7)
            st.session_state["model_config"]["n_future"] = n_future

            # Train model
            if st.button("Train Model"):
                epochs = st.session_state["model_config"]["epochs"]
                batch_size = st.session_state["model_config"]["batch_size"]
                neurons = st.session_state["model_config"]["neurons"]
                time_steps = st.session_state["model_config"]["time_steps"]

                # Split dataset into training and testing
                train_data = data[:int(len(data) * 0.7)]
                test_data = data[int(len(data) * 0.7):]

                # Create sequences
                def create_sequences(data, time_steps):
                    X, y = [], []
                    for i in range(len(data) - time_steps):
                        X.append(data[i:i + time_steps])
                        y.append(data[i + time_steps])
                    return np.array(X), np.array(y)

                X_train, y_train = create_sequences(train_data['Tavg'].values, time_steps)
                X_test, y_test = create_sequences(test_data['Tavg'].values, time_steps)

                # Reshape for GRU input
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

                st.session_state["X_train"] = X_train
                st.session_state["y_train"] = y_train
                st.session_state["X_test"] = X_test
                st.session_state["y_test"] = y_test

                # Build GRU model
                model = Sequential([
                    GRU(neurons, activation='tanh', return_sequences=False, input_shape=(time_steps, 1)),
                    Dense(1)
                ])

                model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

                # Train the model
                st.write("Training the model...")
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
                st.success("Training complete!")

                st.session_state["model"] = model
                st.session_state["scaler"] = scaler

        # Evaluation Tab
        with tabs[1]:
            st.header("📈 Evaluation")

            if "model" in st.session_state and "scaler" in st.session_state:
                model = st.session_state["model"]
                scaler = st.session_state["scaler"]
                X_train = st.session_state["X_train"]
                y_train = st.session_state["y_train"]
                X_test = st.session_state["X_test"]
                y_test = st.session_state["y_test"]

                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                accuracy = (1 - mape) * 100

                y_pred_rescaled = scaler.inverse_transform(y_pred)
                y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

                train_pred = model.predict(X_train)
                train_pred_rescaled = scaler.inverse_transform(train_pred)

                y_train_rescaled = scaler.inverse_transform(y_train.reshape(-1, 1))

                st.write("### Metrics:")
                st.write(f"- MSE: {mse:.4f}")
                st.write(f"- RMSE: {rmse:.4f}")
                st.write(f"- Accuracy: {accuracy:.2f}%")

                st.write("### Plot: Actual vs Predicted")
                plot_predictions(y_train_rescaled, train_pred_rescaled, y_test_rescaled, y_pred_rescaled)
            else:
                st.warning("Model is not trained yet. Please train the model in the Data Overview tab.")

        # Future Prediction Tab
        with tabs[2]:
            st.header("🔮 Future Prediction")

            if "model" in st.session_state and "scaler" in st.session_state:
                model = st.session_state["model"]
                scaler = st.session_state["scaler"]
                X_test = st.session_state["X_test"]

                last_date = data.index[-1]
                future_dates = pd.date_range(
                    last_date + pd.Timedelta(days=1),
                    periods=st.session_state["model_config"]["n_future"]
                ).date

                last_input = X_test[-1]
                future_predictions = []

                for _ in range(st.session_state["model_config"]["n_future"]):
                    next_pred = model.predict(last_input.reshape(1, -1, last_input.shape[-1]))
                    next_pred_rescaled = scaler.inverse_transform(next_pred)
                    future_predictions.append(next_pred_rescaled.flatten()[0])
                    last_input = np.append(last_input[1:], next_pred, axis=0)

                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Temperature': future_predictions
                })

                st.write("### Future Predictions")
                st.dataframe(future_df, width=1200)
            else:
                st.warning("Model is not trained yet. Please train the model in the Data Overview tab.")
