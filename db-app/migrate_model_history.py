import psycopg2
import re
import json
import os

SUPABASE_URL = "postgres://postgres.vmddpyryjnkewvswlhuy:rIrX6zq0bwXpQTkY@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres?sslmode=require"

def main():
    conn = psycopg2.connect(SUPABASE_URL)
    cur = conn.cursor()

    sql_path = os.path.join("upgrade", "program_ipm_jatim_2 (1).sql")
    with open(sql_path, "r", encoding="utf-8") as f:
        sql_text = f.read()

    # 1. Map id_wilayah di Supabase
    cur.execute("SELECT id_wilayah, nama_wilayah FROM wilayah")
    supabase_wilayah = {name.strip().lower(): id_w for id_w, name in cur.fetchall()}

    # Map old_id -> nama_wilayah dari SQL dump
    old_wilayah_map = {}
    m_wil = re.findall(r"\((\d+),\s*'([^']+)'", sql_text)
    for old_id, name in m_wil:
        old_wilayah_map[int(old_id)] = name.strip().lower()

    # 2. Insert riwayat_model
    cur.execute("SELECT id_admin FROM admin LIMIT 1")
    admin_row = cur.fetchone()
    id_admin = admin_row[0] if admin_row else None

    m_loss = re.search(r"'(\{\"loss\":\s*\[.*?\],\s*\"val_loss\":\s*\[.*?\]\})'", sql_text)
    loss_curve_str = m_loss.group(1) if m_loss else None

    cur.execute("""
        INSERT INTO riwayat_model
            (id_model, id_admin, tgl_latih, skor_mape, mape_train, skor_mae, mae_train, skor_rmse, rmse_train, file_model, loss_curve, best_epoch)
        VALUES (1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id_model) DO UPDATE SET
            skor_mape = EXCLUDED.skor_mape,
            loss_curve = EXCLUDED.loss_curve;
    """, (
        id_admin,
        "2026-06-15 12:18:52",
        0.5585, 0.4889, 0.438178, 0.336482, 0.552656, 0.461943,
        "best_gru_model.keras",
        loss_curve_str,
        169
    ))
    print("riwayat_model berhasil dimasukkan/diperbarui.")

    # 3. Insert hasil_prediksi_model
    cur.execute("DELETE FROM hasil_prediksi_model WHERE id_model = 1")
    pattern = re.compile(r"\((\d+),\s*(\d+),\s*(\d+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*'([^']+)'\)")
    matches = pattern.findall(sql_text)

    inserted = 0
    for m in matches:
        if m[1] == '2':
            old_id_wil = int(m[2])
            wil_name = old_wilayah_map.get(old_id_wil)
            new_id_wil = supabase_wilayah.get(wil_name)
            if not new_id_wil:
                continue

            ipm_act = float(m[3])
            ipm_pred = float(m[4])
            err_pct = float(m[5])
            kat = m[6]
            cur.execute("""
                INSERT INTO hasil_prediksi_model (id_model, id_wilayah, ipm_aktual, ipm_prediksi, error_persen, kategori)
                VALUES (1, %s, %s, %s, %s, %s)
            """, (new_id_wil, ipm_act, ipm_pred, err_pct, kat))
            inserted += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"Berhasil memasukkan {inserted} baris hasil prediksi model ke Supabase!")

if __name__ == "__main__":
    main()
