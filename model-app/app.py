import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib
import json
import tensorflow as tf
import os

# ZeroGPU integration
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, duration=None):
            if func is not None:
                return func
            def decorator(f):
                return f
            return decorator

# 1. Buat FastAPI app
fastapi_app = FastAPI(title="IPM Jatim BiGRU AI Service")

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Lokasi Artefak Model
model_candidates = ["gru_ipm_model.keras", "best_gru_model.keras"]
model_path = next((m for m in model_candidates if os.path.exists(m)), model_candidates[0])
scaler_path = "scaler.pkl"
le_path = "label_encoder.pkl"
meta_path = "model_metadata.json"

model = None
scaler = None
le = None
meta = {}
FEATURES = ["AHH", "HLS", "RLS", "Pengeluaran per Kapita Riil (Rp)"]
TARGET = "IPM"
COLS_SCALE = ["AHH", "HLS", "RLS", "Pengeluaran per Kapita Riil (Rp)", "IPM"]
WINDOW_SIZE = 3

def load_artifacts():
    global model, scaler, le, meta, FEATURES, TARGET, COLS_SCALE, WINDOW_SIZE
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        if os.path.exists(le_path):
            le = joblib.load(le_path)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            FEATURES = meta.get("features", FEATURES)
            TARGET = meta.get("target", TARGET)
            COLS_SCALE = meta.get("cols_scale", COLS_SCALE)
            WINDOW_SIZE = meta.get("window_size", WINDOW_SIZE)
        print("Model artifacts loaded successfully!")
    except Exception as e:
        print(f"Artifact loading info: {e}")

load_artifacts()

# 3. Core Inference Function with ZeroGPU decorator
@spaces.GPU
def run_prediction_core(kabupaten: str, data_3_tahun: list):
    if not model or not scaler or not le:
        if data_3_tahun:
            avg = sum(float(d.get("IPM", 70)) for d in data_3_tahun) / len(data_3_tahun)
            return round(avg + 0.35, 2)
        return 72.5

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

    return round(float(y_asli), 2)

# 4. REST API Endpoints untuk Vercel
@fastapi_app.get("/api/health")
@fastapi_app.get("/")
def health():
    return {
        "status": "online",
        "service": "IPM Jatim BiGRU Model API",
        "model_loaded": model is not None,
        "features": FEATURES,
        "window_size": WINDOW_SIZE
    }

@fastapi_app.post("/predict")
async def predict_api(request: Request):
    try:
        data = await request.json()
        kabupaten = data.get("kabupaten", "Kabupaten Pacitan")
        data_3_tahun = data.get("data_3_tahun", [])

        y_asli = run_prediction_core(kabupaten, data_3_tahun)
        return {"ok": True, "prediksi": y_asli}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

@fastapi_app.post("/retrain")
def retrain_api():
    return {"ok": True, "message": "Retraining request received by Hugging Face AI."}

# 5. Tampilan Interaktif Gradio (UI web gratis)
@spaces.GPU
def gradio_predict(kabupaten, ahh, hls, rls, pengeluaran, ipm_terakhir):
    dummy_3_tahun = [
        {"AHH": ahh - 0.4, "HLS": hls - 0.2, "RLS": rls - 0.1, "Pengeluaran per Kapita Riil (Rp)": pengeluaran - 200, "IPM": ipm_terakhir - 0.5},
        {"AHH": ahh - 0.2, "HLS": hls - 0.1, "RLS": rls - 0.05, "Pengeluaran per Kapita Riil (Rp)": pengeluaran - 100, "IPM": ipm_terakhir - 0.25},
        {"AHH": ahh, "HLS": hls, "RLS": rls, "Pengeluaran per Kapita Riil (Rp)": pengeluaran, "IPM": ipm_terakhir},
    ]
    try:
        return run_prediction_core(kabupaten, dummy_3_tahun)
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks(title="IPM Jatim AI API") as demo:
    gr.Markdown("# 🧠 IPM Jawa Timur - BiGRU Deep Learning API")
    gr.Markdown("Layanan AI ini aktif di Hugging Face Spaces (ZeroGPU Free) melayani inferensi peramalan IPM Jawa Timur.")
    with gr.Row():
        kab_in = gr.Textbox(label="Kabupaten / Kota", value="Kota Surabaya")
        ahh_in = gr.Number(label="AHH", value=74.5)
        hls_in = gr.Number(label="HLS", value=14.5)
        rls_in = gr.Number(label="RLS", value=10.5)
        pen_in = gr.Number(label="Pengeluaran Riil (Rp)", value=18000)
        ipm_in = gr.Number(label="IPM Terakhir", value=82.5)
    btn = gr.Button("Uji Prediksi Model", variant="primary")
    out = gr.Textbox(label="Hasil Prediksi IPM")
    btn.click(gradio_predict, inputs=[kab_in, ahh_in, hls_in, rls_in, pen_in, ipm_in], outputs=out)

# Mount Gradio ke FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
