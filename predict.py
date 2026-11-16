import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, request


model_file = 'xgb_model_depth_6.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('stellar-class')

@app.route('/predict', methods=['POST'])
def predict():
    stellar_object = request.get_json()

    feature_names = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
    X = pd.DataFrame([stellar_object])[feature_names]
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    y_pred = model.predict(dmatrix)[0]
    y_pred_class = int(np.argmax(y_pred))

    classes = {0: 'GALAXY', 1: 'STAR', 2: 'QSO'}

    result = {
        'predicted_class': classes[y_pred_class],
        'probabilities': {
            'GALAXY': float(y_pred[0]),
            'STAR': float(y_pred[1]),
            'QSO': float(y_pred[2])
        }
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
