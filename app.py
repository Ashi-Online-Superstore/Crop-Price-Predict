# app.py
# this one

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model, encoders, and scaler
model_path = 'crop_price_model.pkl'
crop_encoder_path = 'crop_encoder.pkl'
month_encoder_path = 'month_encoder.pkl'
city_encoder_path = 'city_encoder.pkl'
state_encoder_path = 'state_encoder.pkl'
scaler_path = 'scaler.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)
with open(crop_encoder_path, 'rb') as file:
    crop_encoder = pickle.load(file)
with open(month_encoder_path, 'rb') as file:
    month_encoder = pickle.load(file)
with open(city_encoder_path, 'rb') as file:
    city_encoder = pickle.load(file)
with open(state_encoder_path, 'rb') as file:
    state_encoder = pickle.load(file)
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received Data:", data)

        # Extract input values
        year = data.get('year')
        crop = data.get('crop')
        month = data.get('month')
        city = data.get('city')
        state = data.get('state')
        

        print("Extracted Data:", year,crop, month, city, state)

        # Use a default value or handle unseen labels
        year_encoded = int(year)

        if crop not in crop_encoder.classes_:
            crop_encoded = -1 
        else:
            crop_encoded = crop_encoder.transform([crop])[0]

        if month not in month_encoder.classes_:
            month_encoded = -1
        else:
            month_encoded = month_encoder.transform([month])[0]

        if city not in city_encoder.classes_:
            city_encoded = -1
        else:
            city_encoded = city_encoder.transform([city])[0]

        if state not in state_encoder.classes_:
            state_encoded = -1
        else:
            state_encoded = state_encoder.transform([state])[0]


        print("Encoded Values:", year_encoded,crop_encoded, month_encoded, city_encoded, state_encoded)

        # Prepare input for prediction
        input_data = np.array([[year_encoded,crop_encoded, month_encoded, city_encoded, state_encoded]])

        # Apply scaling
        input_data_scaled = scaler.transform(input_data)

        # Make prediction using the scaled input
        predicted_price = model.predict(input_data_scaled)[0]
        print("Predicted Price:", predicted_price) 

        # Return the predicted price
        return jsonify({'price': float(predicted_price)})

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)  

