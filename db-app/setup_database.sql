# ══════════════════════════════════════════════════════════
#  SETUP DATABASE — db_ipm_jatim
#  Sistem Prediksi IPM Jawa Timur berbasis BiGRU
#  Jalankan SQL ini di phpMyAdmin (XAMPP) atau MySQL/MariaDB CLI
# ══════════════════════════════════════════════════════════

-- Buat database
CREATE DATABASE IF NOT EXISTS db_ipm_jatim
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_general_ci;

USE db_ipm_jatim;

-- ──────────────────────────────────────────────
-- 1. Tabel Admin
-- ──────────────────────────────────────────────
CREATE TABLE admin (
    id_admin     INT(11) AUTO_INCREMENT PRIMARY KEY,
    username     VARCHAR(50)  NOT NULL UNIQUE,
    password     VARCHAR(255) NOT NULL,          -- disimpan dalam bentuk hash (werkzeug scrypt)
    nama_lengkap VARCHAR(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- ──────────────────────────────────────────────
-- 2. Tabel Wilayah (38 kabupaten/kota Jawa Timur)
-- ──────────────────────────────────────────────
CREATE TABLE wilayah (
    id_wilayah   INT(11) AUTO_INCREMENT PRIMARY KEY,
    nama_wilayah VARCHAR(100) NOT NULL,
    deskripsi    TEXT         DEFAULT NULL,       -- profil singkat wilayah (ditampilkan di halaman publik)
    url_logo     VARCHAR(255) DEFAULT NULL,       -- path logo wilayah, mis. /static/logo_wilayah/...
    url_landmark VARCHAR(255) DEFAULT NULL,       -- path foto landmark wilayah
    is_deleted   TINYINT(1)   DEFAULT 0,          -- soft delete flag
    KEY idx_wilayah_active (is_deleted)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- ──────────────────────────────────────────────
-- 3. Tabel Indikator Historis IPM
-- ──────────────────────────────────────────────
CREATE TABLE indikator_historis (
    id_indikator INT(11) AUTO_INCREMENT PRIMARY KEY,
    id_wilayah   INT(11),
    tahun        YEAR(4),
    ahh          FLOAT,
    hls          FLOAT,
    rls          FLOAT,
    pengeluaran  DECIMAL(15,2),
    ipm_aktual   FLOAT,
    UNIQUE KEY uq_wilayah_tahun (id_wilayah, tahun),   -- cegah duplikasi data tahun yg sama utk 1 wilayah
    FOREIGN KEY (id_wilayah) REFERENCES wilayah(id_wilayah)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- ──────────────────────────────────────────────
-- 4. Tabel Riwayat Model (log setiap sesi pelatihan BiGRU)
-- ──────────────────────────────────────────────
CREATE TABLE riwayat_model (
    id_model   INT(11) AUTO_INCREMENT PRIMARY KEY,
    id_admin   INT(11),
    tgl_latih  DATETIME,
    skor_mape  FLOAT,                 -- MAPE data test
    mape_train FLOAT,                 -- MAPE data train
    skor_mae   FLOAT,
    mae_train  FLOAT,
    skor_rmse  FLOAT,
    rmse_train FLOAT,
    file_model VARCHAR(100),
    loss_curve LONGTEXT,              -- JSON {"loss": [...], "val_loss": [...]} per epoch, utk chart dinamis
    best_epoch INT(11),               -- epoch terbaik berdasarkan val_loss minimum (EarlyStopping)
    FOREIGN KEY (id_admin) REFERENCES admin(id_admin)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- ──────────────────────────────────────────────
-- 5. Tabel Hasil Prediksi Model (evaluasi model pada seluruh wilayah)
-- ──────────────────────────────────────────────
CREATE TABLE hasil_prediksi_model (
    id_uji       INT(11) AUTO_INCREMENT PRIMARY KEY,
    id_model     INT(11) NOT NULL,
    id_wilayah   INT(11) NOT NULL,
    ipm_aktual   FLOAT,
    ipm_prediksi FLOAT,
    error_persen FLOAT,
    kategori     ENUM('Sangat Tinggi','Tinggi','Sedang','Rendah'),
    CONSTRAINT fk_uji_model   FOREIGN KEY (id_model)   REFERENCES riwayat_model(id_model),
    CONSTRAINT fk_uji_wilayah FOREIGN KEY (id_wilayah) REFERENCES wilayah(id_wilayah)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- ──────────────────────────────────────────────
-- 6. Tabel Hasil Uji Simulasi (simulasi input manual per wilayah)
-- ──────────────────────────────────────────────
CREATE TABLE hasil_uji_simulasi (
    id_prediksi    INT(11) AUTO_INCREMENT PRIMARY KEY,
    id_model       INT(11),
    id_wilayah     INT(11),
    data_sequence  LONGTEXT NOT NULL COMMENT 'Menyimpan 12 indikator (4 variabel x 3 tahun) dalam format JSON',
    tahun_prediksi YEAR(4),
    nilai_prediksi FLOAT,
    kategori_ipm   VARCHAR(20),
    tgl_simulasi   DATETIME,
    FOREIGN KEY (id_model)   REFERENCES riwayat_model(id_model),
    FOREIGN KEY (id_wilayah) REFERENCES wilayah(id_wilayah)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- ──────────────────────────────────────────────
-- 7. Tabel Parameter Klasifikasi (ambang batas kategori IPM)
-- ──────────────────────────────────────────────
CREATE TABLE parameter_klasifikasi (
    id_parameter  INT(11) AUTO_INCREMENT PRIMARY KEY,
    kategori      VARCHAR(50)  NOT NULL,
    ambang_bawah  DECIMAL(5,2) NOT NULL,
    ambang_atas   DECIMAL(5,2) DEFAULT NULL,
    warna_label   VARCHAR(20)  DEFAULT '#000000',
    tgl_update    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- Seed data kategori klasifikasi IPM standar BPS
INSERT INTO parameter_klasifikasi (kategori, ambang_bawah, ambang_atas, warna_label) VALUES
('Sangat Tinggi', 80.00, 100.00, '#006400'),
('Tinggi',        70.00, 79.99,  '#228B22'),
('Sedang',        60.00, 69.99,  '#FFD700'),
('Rendah',         0.00, 59.99,  '#FF0000');


# ══════════════════════════════════════════════════════════
#  MEMBUAT AKUN ADMIN (bisa dijalankan kapan pun butuh admin baru)
#  Buka terminal, aktifkan virtual env, lalu:
# ══════════════════════════════════════════════════════════

#  python create_admin.py
#
#  Script akan bertanya secara interaktif di terminal:
#    - Username admin
#    - Nama lengkap
#    - Password (diketik via getpass, tidak tampil di layar, tidak
#      di-hardcode di file mana pun)
#
#  Password langsung di-hash pakai werkzeug generate_password_hash()
#  (scrypt) sebelum disimpan ke tabel admin. Lihat create_admin.py
#  untuk detail lengkapnya.
# ----------------------------------------------------------

# ══════════════════════════════════════════════════════════
#  CATATAN
# ══════════════════════════════════════════════════════════
#  - Data 38 wilayah (kabupaten/kota Jawa Timur), data historis
#    indikator per tahun, dan riwayat model TIDAK disertakan di
#    script ini (hanya struktur/skema). Import data dari file
#    dump lengkap (mis. program_ipm_jatim_2.sql) jika diperlukan.
#  - werkzeug generate_password_hash() versi terbaru default-nya
#    menghasilkan hash scrypt (bukan bcrypt), jadi tidak perlu
#    instal library tambahan seperti flask-bcrypt.
