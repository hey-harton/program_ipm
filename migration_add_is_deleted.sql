-- ══════════════════════════════════════════════════════════════
--  MIGRATION: Tambah kolom is_deleted ke tabel wilayah
--  Jalankan di phpMyAdmin (XAMPP) → tab SQL
-- ══════════════════════════════════════════════════════════════

USE db_ipm_jatim;

-- Tambah kolom is_deleted (jika belum ada)
ALTER TABLE wilayah
    ADD COLUMN IF NOT EXISTS is_deleted TINYINT(1) NOT NULL DEFAULT 0;

-- ── Isi 38 wilayah Jawa Timur (abjad) ─────────────────────────
INSERT INTO wilayah (nama_wilayah, is_deleted) VALUES
  ('Kabupaten Bangkalan',    0),
  ('Kabupaten Banyuwangi',   0),
  ('Kabupaten Blitar',       0),
  ('Kabupaten Bojonegoro',   0),
  ('Kabupaten Bondowoso',    0),
  ('Kabupaten Gresik',       0),
  ('Kabupaten Jember',       0),
  ('Kabupaten Jombang',      0),
  ('Kabupaten Kediri',       0),
  ('Kabupaten Lamongan',     0),
  ('Kabupaten Lumajang',     0),
  ('Kabupaten Madiun',       0),
  ('Kabupaten Magetan',      0),
  ('Kabupaten Malang',       0),
  ('Kabupaten Mojokerto',    0),
  ('Kabupaten Nganjuk',      0),
  ('Kabupaten Ngawi',        0),
  ('Kabupaten Pacitan',      0),
  ('Kabupaten Pamekasan',    0),
  ('Kabupaten Pasuruan',     0),
  ('Kabupaten Ponorogo',     0),
  ('Kabupaten Probolinggo',  0),
  ('Kabupaten Sampang',      0),
  ('Kabupaten Sidoarjo',     0),
  ('Kabupaten Situbondo',    0),
  ('Kabupaten Sumenep',      0),
  ('Kabupaten Trenggalek',   0),
  ('Kabupaten Tuban',        0),
  ('Kabupaten Tulungagung',  0),
  ('Kota Batu',              0),
  ('Kota Blitar',            0),
  ('Kota Kediri',            0),
  ('Kota Madiun',            0),
  ('Kota Malang',            0),
  ('Kota Mojokerto',         0),
  ('Kota Pasuruan',          0),
  ('Kota Probolinggo',       0),
  ('Kota Surabaya',          0);
