"""
seed_to_supabase.py
Script untuk memasukkan seluruh data awal ke Supabase:
  1. Akun Admin default
  2. 38 Wilayah Kabupaten/Kota Jawa Timur (dengan path logo & landmark)
  3. Data Historis IPM 2010–2024 dari CSV
"""

import os
import csv
import psycopg2
from werkzeug.security import generate_password_hash

# Ganti dengan URL Supabase Anda jika belum di-set di environment
SUPABASE_URL = os.environ.get(
    'SUPABASE_URL',
    input("Masukkan connection string Supabase (misal: postgresql://postgres:...): ").strip()
)

CSV_FILE = os.path.join(os.path.dirname(__file__), 'IPM Kabupaten_Kota_Prov_Jawa_Timur.csv')

def run_seed():
    print("Menghubungkan ke Supabase...")
    conn = psycopg2.connect(SUPABASE_URL)
    cur = conn.cursor()

    # 1. Seed Admin Default
    print("1. Membuat akun admin default...")
    admin_user = "admin"
    admin_pass = "admin123"  # Silakan ganti sesuai keinginan
    admin_hash = generate_password_hash(admin_pass)
    cur.execute("""
        INSERT INTO admin (username, password, nama_lengkap)
        VALUES (%s, %s, %s)
        ON CONFLICT (username) DO NOTHING;
    """, (admin_user, admin_hash, "Administrator IPM"))

    # 2. Ambil daftar unik wilayah dari CSV
    print("2. Membaca data CSV dan mendaftarkan wilayah...")
    wilayah_set = set()
    rows = []
    with open(CSV_FILE, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        for r in reader:
            w_nama = r['Kabupaten/Kota'].strip()
            wilayah_set.add(w_nama)
            rows.append(r)

    # Insert wilayah ke tabel wilayah
    wilayah_map = {}
    for w in sorted(list(wilayah_set)):
        # Format logo & landmark filename
        clean_name = w.replace("Kabupaten ", "Kab_").replace("Kota ", "Kota_").replace(" ", "_")
        logo_url = f"/static/logo_wilayah/{clean_name}.png"
        landmark_url = f"/static/landmark_wilayah/{clean_name}.png"
        deskripsi = f"Profil data indikator IPM untuk wilayah {w}, Provinsi Jawa Timur."

        cur.execute("""
            INSERT INTO wilayah (nama_wilayah, deskripsi, url_logo, url_landmark, is_deleted)
            VALUES (%s, %s, %s, %s, FALSE)
            ON CONFLICT DO NOTHING;
        """, (w, deskripsi, logo_url, landmark_url))

    # Ambil map id_wilayah
    cur.execute("SELECT id_wilayah, nama_wilayah FROM wilayah")
    for row in cur.fetchall():
        wilayah_map[row[1].strip().lower()] = row[0]

    print(f"   -> {len(wilayah_map)} wilayah berhasil disiapkan.")

    # 3. Insert Data Historis IPM
    print("3. Memasukkan seluruh baris data indikator historis...")
    count = 0
    for r in rows:
        w_nama = r['Kabupaten/Kota'].strip()
        id_wil = wilayah_map.get(w_nama.lower())
        if not id_wil:
            continue

        tahun = int(r['Tahun'])
        ahh = float(r['AHH'].replace(',', '.'))
        hls = float(r['HLS'].replace(',', '.'))
        rls = float(r['RLS'].replace(',', '.'))
        
        pen_raw = r['Pengeluaran per Kapita Riil (Rp)'].strip()
        if ',' in pen_raw:
            pen_raw = pen_raw.replace('.', '').replace(',', '.')
        else:
            pen_raw = pen_raw.replace(',', '')
        pengeluaran = float(pen_raw)
        
        ipm = float(r['IPM'].replace(',', '.'))

        cur.execute("""
            INSERT INTO indikator_historis (id_wilayah, tahun, ahh, hls, rls, pengeluaran, ipm_aktual)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id_wilayah, tahun) DO UPDATE SET
                ahh = EXCLUDED.ahh,
                hls = EXCLUDED.hls,
                rls = EXCLUDED.rls,
                pengeluaran = EXCLUDED.pengeluaran,
                ipm_aktual = EXCLUDED.ipm_aktual;
        """, (id_wil, tahun, ahh, hls, rls, pengeluaran, ipm))
        count += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"SUKSES! Berhasil memasukkan {count} baris data historis ke Supabase.")
    print(f"Akun login admin: username = '{admin_user}', password = '{admin_pass}'")

if __name__ == '__main__':
    run_seed()
