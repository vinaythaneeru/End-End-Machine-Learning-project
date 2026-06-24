import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# 1. Correctly load the models using Pickle (Instructor's Method)
# Ensure these filenames match what you saved in your notebook
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Capture the JSON data
    data = request.json['data']
    print("Received Data:", data)
    
    # 2. Prepare data: Convert dictionary values to a list, then a 2D array
    # Retail Sales expects: [Sales, Quantity, Discount]
    input_values = np.array(list(data.values())).reshape(1, -1)
    
    # 3. Transform data using the scaler (Standardization is crucial)
    new_data = scalar.transform(input_values)
    
    # 4. Predict using the regression model
    output = regmodel.predict(new_data)
    
    print("Prediction:", output)
    return jsonify(float(output))

# 5. Fix the main entry point
if __name__ == "__main__":
    app.run(debug=True)