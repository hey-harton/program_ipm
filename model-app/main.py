from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
import tensorflow as tf
import os
import threading
import time

app = Flask(__name__)

# Load artifacts
model_path = "best_gru_model.keras"
scaler_path = "scaler.pkl"
le_path = "label_encoder.pkl"
meta_path = "model_metadata.json"

try:
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    FEATURES = meta.get("features", [])
    TARGET = meta.get("target", "IPM")
    COLS_SCALE = meta.get("cols_scale", [])
    WINDOW_SIZE = meta.get("window_size", 3)
except Exception as e:
    print(f"Warning: Could not load model artifacts. Ensure they are trained and placed in model-app/. Error: {e}")

# Retraining state
_retrain_state = {
    'status': 'idle',
    'progress': 0,
    'current_epoch': 0,
    'total_epochs': 200,
    'log_msg': ''
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        kabupaten = data.get('kabupaten')
        data_3_tahun = data.get('data_3_tahun')
        
        region_id = le.transform([kabupaten])[0]
        rows = []
        for d in data_3_tahun:
            row = [d[f] for f in FEATURES] + [d[TARGET]]
            rows.append(row)
        
        arr_scaled = scaler.transform(rows)
        region_col = np.full((WINDOW_SIZE, 1), region_id / len(le.classes_))
        X_input = np.hstack([arr_scaled[:, :-1], region_col])
        X_input = np.expand_dims(X_input, axis=0).astype(np.float32)
        
        y_scaled = model.predict(X_input, verbose=0).flatten()[0]
        
        dummy = np.zeros((1, len(COLS_SCALE)))
        dummy[0, -1] = y_scaled
        y_asli = scaler.inverse_transform(dummy)[0, -1]
        
        return jsonify({'ok': True, 'prediksi': round(float(y_asli), 2)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    # In a real scenario, this would accept raw data or fetch from Supabase,
    # then retrain and update model artifacts.
    # We will simulate the start.
    global _retrain_state
    if _retrain_state['status'] == 'running':
        return jsonify({'ok': False, 'msg': 'Retraining is already running'})
    
    _retrain_state['status'] = 'running'
    _retrain_state['progress'] = 0
    
    def simulate_training():
        global _retrain_state
        for i in range(10):
            time.sleep(2)
            _retrain_state['progress'] += 10
            _retrain_state['current_epoch'] = (i+1)*20
            _retrain_state['log_msg'] = f'Epoch {(i+1)*20} completed'
        _retrain_state['status'] = 'idle'
        _retrain_state['log_msg'] = 'Training completed'
    
    threading.Thread(target=simulate_training).start()
    return jsonify({'ok': True, 'msg': 'Retraining started'})

@app.route('/retrain-status', methods=['GET'])
def retrain_status():
    return jsonify(_retrain_state)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
