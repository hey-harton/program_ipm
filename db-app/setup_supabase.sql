-- ══════════════════════════════════════════════════════════
--  SETUP DATABASE — Supabase (PostgreSQL)
--  Sistem Prediksi IPM Jawa Timur berbasis BiGRU
-- ══════════════════════════════════════════════════════════

-- ──────────────────────────────────────────────
-- 1. Tabel Admin
-- ──────────────────────────────────────────────
CREATE TABLE admin (
    id_admin     SERIAL PRIMARY KEY,
    username     VARCHAR(50)  NOT NULL UNIQUE,
    password     VARCHAR(255) NOT NULL,          -- disimpan dalam bentuk hash (werkzeug scrypt)
    nama_lengkap VARCHAR(100) NOT NULL
);

-- ──────────────────────────────────────────────
-- 2. Tabel Wilayah (38 kabupaten/kota Jawa Timur)
-- ──────────────────────────────────────────────
CREATE TABLE wilayah (
    id_wilayah   SERIAL PRIMARY KEY,
    nama_wilayah VARCHAR(100) NOT NULL,
    deskripsi    TEXT         DEFAULT NULL,
    url_logo     VARCHAR(255) DEFAULT NULL,
    url_landmark VARCHAR(255) DEFAULT NULL,
    is_deleted   BOOLEAN      DEFAULT FALSE
);

CREATE INDEX idx_wilayah_active ON wilayah (is_deleted);

-- ──────────────────────────────────────────────
-- 3. Tabel Indikator Historis IPM
-- ──────────────────────────────────────────────
CREATE TABLE indikator_historis (
    id_indikator SERIAL PRIMARY KEY,
    id_wilayah   INTEGER REFERENCES wilayah(id_wilayah),
    tahun        INTEGER,
    ahh          DOUBLE PRECISION,
    hls          DOUBLE PRECISION,
    rls          DOUBLE PRECISION,
    pengeluaran  DECIMAL(15,2),
    ipm_aktual   DOUBLE PRECISION,
    UNIQUE (id_wilayah, tahun)
);

-- ──────────────────────────────────────────────
-- 4. Tabel Riwayat Model (log setiap sesi pelatihan BiGRU)
-- ──────────────────────────────────────────────
CREATE TABLE riwayat_model (
    id_model   SERIAL PRIMARY KEY,
    id_admin   INTEGER REFERENCES admin(id_admin),
    tgl_latih  TIMESTAMP,
    skor_mape  DOUBLE PRECISION,                 
    mape_train DOUBLE PRECISION,                 
    skor_mae   DOUBLE PRECISION,
    mae_train  DOUBLE PRECISION,
    skor_rmse  DOUBLE PRECISION,
    rmse_train DOUBLE PRECISION,
    file_model VARCHAR(100),
    loss_curve TEXT,              -- JSON
    best_epoch INTEGER
);

-- ──────────────────────────────────────────────
-- 5. Tabel Hasil Prediksi Model (evaluasi model pada seluruh wilayah)
-- ──────────────────────────────────────────────
CREATE TABLE hasil_prediksi_model (
    id_uji       SERIAL PRIMARY KEY,
    id_model     INTEGER NOT NULL REFERENCES riwayat_model(id_model),
    id_wilayah   INTEGER NOT NULL REFERENCES wilayah(id_wilayah),
    ipm_aktual   DOUBLE PRECISION,
    ipm_prediksi DOUBLE PRECISION,
    error_persen DOUBLE PRECISION,
    kategori     VARCHAR(50) CHECK (kategori IN ('Sangat Tinggi','Tinggi','Sedang','Rendah'))
);

-- ──────────────────────────────────────────────
-- 6. Tabel Hasil Uji Simulasi (simulasi input manual per wilayah)
-- ──────────────────────────────────────────────
CREATE TABLE hasil_uji_simulasi (
    id_prediksi    SERIAL PRIMARY KEY,
    id_model       INTEGER REFERENCES riwayat_model(id_model),
    id_wilayah     INTEGER REFERENCES wilayah(id_wilayah),
    data_sequence  TEXT NOT NULL,
    tahun_prediksi INTEGER,
    nilai_prediksi DOUBLE PRECISION,
    kategori_ipm   VARCHAR(20),
    tgl_simulasi   TIMESTAMP
);

-- ──────────────────────────────────────────────
-- 7. Tabel Parameter Klasifikasi (ambang batas kategori IPM)
-- ──────────────────────────────────────────────
CREATE TABLE parameter_klasifikasi (
    id_parameter  SERIAL PRIMARY KEY,
    kategori      VARCHAR(50)  NOT NULL,
    ambang_bawah  DECIMAL(5,2) NOT NULL,
    ambang_atas   DECIMAL(5,2) DEFAULT NULL,
    warna_label   VARCHAR(20)  DEFAULT '#000000',
    tgl_update    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Seed data kategori klasifikasi IPM standar BPS
INSERT INTO parameter_klasifikasi (kategori, ambang_bawah, ambang_atas, warna_label) VALUES
('Sangat Tinggi', 80.00, 100.00, '#006400'),
('Tinggi',        70.00, 79.99,  '#228B22'),
('Sedang',        60.00, 69.99,  '#FFD700'),
('Rendah',         0.00, 59.99,  '#FF0000');
