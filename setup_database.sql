# ══════════════════════════════════════════════════════════
#  SETUP DATABASE — db_ipm_jatim
#  Jalankan SQL ini di phpMyAdmin (XAMPP) atau MySQL CLI
# ══════════════════════════════════════════════════════════

-- Buat database
CREATE DATABASE IF NOT EXISTS db_ipm_jatim
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE db_ipm_jatim;

-- 1. Tabel Admin
CREATE TABLE admin (
    id_admin     INT(11) AUTO_INCREMENT PRIMARY KEY,
    username     VARCHAR(50)  NOT NULL UNIQUE,
    password     VARCHAR(255) NOT NULL,          -- disimpan dalam bentuk hash (bcrypt)
    nama_lengkap VARCHAR(100) NOT NULL
);

-- 2. Tabel Wilayah
CREATE TABLE wilayah (
    id_wilayah   INT(11) AUTO_INCREMENT PRIMARY KEY,
    nama_wilayah VARCHAR(100) NOT NULL
);

-- 3. Tabel Indikator Historis IPM
CREATE TABLE indikator_historis (
    id_indikator INT(11) AUTO_INCREMENT PRIMARY KEY,
    id_wilayah   INT(11),
    tahun        YEAR,
    ahh          FLOAT,
    hls          FLOAT,
    rls          FLOAT,
    pengeluaran  DECIMAL(15,2),
    ipm_aktual   FLOAT,
    FOREIGN KEY (id_wilayah) REFERENCES wilayah(id_wilayah)
);

-- 4. Tabel Riwayat Model
CREATE TABLE riwayat_model (
    id_model   INT(11) AUTO_INCREMENT PRIMARY KEY,
    id_admin   INT(11),
    tgl_latih  DATETIME,
    skor_mape  FLOAT,
    skor_mae   FLOAT,
    skor_rmse  FLOAT,
    file_model VARCHAR(100),
    FOREIGN KEY (id_admin) REFERENCES admin(id_admin)
);

-- 5. Tabel Hasil Prediksi
CREATE TABLE hasil_prediksi (
    id_prediksi    INT(11) AUTO_INCREMENT PRIMARY KEY,
    id_model       INT(11),
    id_wilayah     INT(11),
    tahun_prediksi YEAR,
    nilai_prediksi FLOAT,
    kategori_ipm   VARCHAR(20),
    tgl_simulasi   DATETIME,
    FOREIGN KEY (id_model)    REFERENCES riwayat_model(id_model),
    FOREIGN KEY (id_wilayah)  REFERENCES wilayah(id_wilayah)
);


# ══════════════════════════════════════════════════════════
#  MEMBUAT AKUN ADMIN PERTAMA (jalankan SEKALI saja)
#  Buka terminal, aktifkan virtual env, lalu:
# ══════════════════════════════════════════════════════════

#  python create_admin.py
#
#  Isi file create_admin.py:
# ----------------------------------------------------------
#  from app import get_db_connection
#  from werkzeug.security import generate_password_hash
#
#  conn = get_db_connection()
#  cursor = conn.cursor()
#  hashed = generate_password_hash('admin123')   # <-- ganti password sesuai keinginan
#  cursor.execute(
#      "INSERT INTO admin (username, password, nama_lengkap) VALUES (%s, %s, %s)",
#      ('admin', hashed, 'Administrator')
#  )
#  conn.commit()
#  cursor.close()
#  conn.close()
#  print("✅ Akun admin berhasil dibuat!")
# ----------------------------------------------------------
