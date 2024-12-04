from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
from model import TemperatureGRU
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

# Load model
input_size = 2
hidden_size = 64
num_layers = 2
output_size = 1

model = TemperatureGRU(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load('model/temperature_gru_model.pth'))
model.eval()

# Load data and scaler
data = pd.read_csv('data/temperature_data.csv')
scaler = MinMaxScaler()
scaler.fit(data[['feature1', 'feature2', 'temperature']])

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json['input']
    sequence_length = 7
    
    # Prepare input sequence
    input_sequence = np.array([input_data for _ in range(sequence_length)])
    scaled_input = scaler.transform(input_sequence)[:, :2]
    
    with torch.no_grad():
        input_tensor = torch.FloatTensor(scaled_input).unsqueeze(0)
        prediction = model(input_tensor)
    
    temperature = scaler.inverse_transform(
        np.hstack([scaled_input[-1], prediction.numpy()])
    )[0, 2]
    
    return jsonify({'temperature': float(temperature)})

@app.route('/model_info', methods=['GET'])
def model_info():
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return jsonify({
        'total_params': total_params,
        'trainable_params': trainable_params,
        'num_layers': model.num_layers,
        'hidden_size': model.hidden_size,
        'input_size': model.gru.input_size,
        'output_size': model.fc.out_features
    })

@app.route('/dataset', methods=['GET'])
def get_dataset():
    return jsonify(data.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)