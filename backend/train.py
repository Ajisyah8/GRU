import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from model import TemperatureGRU

# Generate sample data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
feature1 = np.random.rand(len(dates))
feature2 = np.random.rand(len(dates))
temperature = 20 + 15 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.randn(len(dates)) * 3

data = pd.DataFrame({
    'date': dates,
    'feature1': feature1,
    'feature2': feature2,
    'temperature': temperature
})

data.to_csv('data/temperature_data.csv', index=False)

# Load and preprocess data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['feature1', 'feature2', 'temperature']])

# Prepare sequences
sequence_length = 7
X, y = [], []
for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i+sequence_length, :2])
    y.append(scaled_data[i+sequence_length, 2])

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# Initialize model
input_size = 2
hidden_size = 64
num_layers = 2
output_size = 1

model = TemperatureGRU(input_size, hidden_size, num_layers, output_size)

# Train model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# Save model
torch.save(model.state_dict(), 'model/temperature_gru_model.pth')

print("GRU model trained and saved successfully!")