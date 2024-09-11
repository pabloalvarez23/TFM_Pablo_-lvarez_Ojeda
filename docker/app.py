import datetime

from flask import request
import pandas as pd

from ms import app
from ms.functions import get_model_response

model_name = "Mental Disorder Classifier"
model_file = "model_binary.dat.gz"
version = "v1.0.0"

# Return model information, version, call
@app.route('/info', methods=['GET'])
def info():
    result = {}

    result["name"] = model_name
    result["version"] = version

    return result

# Return server status
@app.route('/health', methods=['GET'])
def health():
    return 'OK'

# Prediction request
@app.route('/predict', methods=['POST'])
def predict():
    feature_dict = request.get_json()
    if not feature_dict:
        return {
            'error': 'Body is empty.'
        }, 500
    try:
        response = get_model_response(feature_dict)
    except ValueError as e:
        return {'error': str(e).split('\n')[-1].strip()}, 500
    
    return response, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)