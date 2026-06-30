import numpy as np
import pandas as pd
import joblib
import json
import tensorflow as tf

# ── Load artefak (jalankan sekali saat server start) ──────────────────────────
model    = tf.keras.models.load_model("gru_ipm_model.keras")
scaler   = joblib.load("scaler.pkl")
le       = joblib.load("label_encoder.pkl")

with open("model_metadata.json", "r") as f:
    meta = json.load(f)

FEATURES     = meta["features"]
TARGET       = meta["target"]
COLS_SCALE   = meta["cols_scale"]
WINDOW_SIZE  = meta["window_size"]


def predict_ipm(kabupaten: str, data_3_tahun: list[dict]) -> float:
    """
    Prediksi IPM tahun berikutnya untuk satu kabupaten.

    Parameters
    ----------
    kabupaten    : nama kabupaten/kota (harus ada di label_encoder)
    data_3_tahun : list 3 dict, masing-masing berisi:
                   {"AHH": float, "HLS": float, "RLS": float,
                    "Pengeluaran per Kapita Riil (Rp)": float, "IPM": float}

    Returns
    -------
    float : prediksi nilai IPM (skala asli)
    """
    assert len(data_3_tahun) == WINDOW_SIZE, f"Butuh tepat {WINDOW_SIZE} baris data"

    region_id = le.transform([kabupaten])[0]

    # Susun dataframe dan scaling
    rows = []
    for d in data_3_tahun:
        row = [d[f] for f in FEATURES] + [d[TARGET]]
        rows.append(row)

    arr_scaled = scaler.transform(rows)  # shape (3, 5)

    # Tambahkan Region_ID sebagai fitur terakhir (tidak discale)
    region_col = np.full((WINDOW_SIZE, 1), region_id / len(le.classes_))
    X_input = np.hstack([arr_scaled[:, :-1], region_col])  # buang kolom IPM

    X_input = np.expand_dims(X_input, axis=0).astype(np.float32)  # (1, 3, 5)

    # Prediksi
    y_scaled = model.predict(X_input, verbose=0).flatten()[0]

    # Inverse transform ke skala asli
    dummy = np.zeros((1, len(COLS_SCALE)))
    dummy[0, -1] = y_scaled
    y_asli = scaler.inverse_transform(dummy)[0, -1]

    return round(float(y_asli), 2)


# ── Contoh penggunaan ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    contoh_data = [
        {"AHH": 73.5, "HLS": 13.2, "RLS": 8.1, "Pengeluaran per Kapita Riil (Rp)": 11500, "IPM": 72.0},
        {"AHH": 73.8, "HLS": 13.4, "RLS": 8.3, "Pengeluaran per Kapita Riil (Rp)": 11800, "IPM": 72.5},
        {"AHH": 74.1, "HLS": 13.6, "RLS": 8.5, "Pengeluaran per Kapita Riil (Rp)": 12100, "IPM": 73.0},
    ]
    hasil = predict_ipm("Kota Surabaya", contoh_data)
    print(f"Prediksi IPM: {hasil}")