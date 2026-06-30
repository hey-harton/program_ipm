"""
create_admin.py
───────────────
Jalankan file ini SATU KALI untuk membuat akun admin pertama.

Cara pakai:
  1. Pastikan database db_ipm_jatim sudah dibuat dan tabel admin sudah ada.
  2. Aktifkan virtual environment (jika pakai):
       Windows : venv\Scripts\activate
       Mac/Linux: source venv/bin/activate
  3. Jalankan:
       python create_admin.py
"""

from app import get_db_connection
from werkzeug.security import generate_password_hash

# ── Isi sesuai kebutuhan ──────────────────────────────────
USERNAME     = 'admin'
PASSWORD     = 'admin123'        # Ganti dengan password yang kuat!
NAMA_LENGKAP = 'Administrator'
# ─────────────────────────────────────────────────────────

def create_admin():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Cek apakah username sudah ada
    cursor.execute("SELECT id_admin FROM admin WHERE username = %s", (USERNAME,))
    existing = cursor.fetchone()

    if existing:
        print(f"⚠️  Username '{USERNAME}' sudah ada di database. Tidak ada yang diubah.")
    else:
        hashed_password = generate_password_hash(PASSWORD)
        cursor.execute(
            "INSERT INTO admin (username, password, nama_lengkap) VALUES (%s, %s, %s)",
            (USERNAME, hashed_password, NAMA_LENGKAP)
        )
        conn.commit()
        print(f"   Akun admin berhasil dibuat!")
        print(f"   Username     : {USERNAME}")
        print(f"   Nama Lengkap : {NAMA_LENGKAP}")
        print(f"   Password     : {PASSWORD}  ← simpan baik-baik!")

    cursor.close()
    conn.close()

if __name__ == '__main__':
    create_admin()
