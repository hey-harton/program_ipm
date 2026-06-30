"""
app.py — IPM Predictor Jawa Timur
Flask backend lengkap untuk semua halaman:
  • Auth (login/logout)
  • Dashboard Admin
  • Manajemen Data Historis (dengan soft-delete & pagination)
  • Riwayat Prediksi (arsip semua prediksi)
  • Detail Prediksi (chart + metrik per wilayah)
  • Retraining Model (async training + long-polling progress)
  • Konfigurasi Parameter (klasifikasi IPM — terintegrasi ke semua kategori)
  • API endpoints pendukung
"""

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify, abort
)
from werkzeug.security import check_password_hash
import mysql.connector
from functools import wraps
import os, json, io, csv, threading, time, traceback, logging

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ipm-jatim-secret-key-2025')

# ─── Database Config ───────────────────────────────────────────────────────────
DB_CONFIG = {
    'host':     os.environ.get('MYSQLHOST', 'localhost'),
    'user':     os.environ.get('MYSQLUSER', 'root'),
    'password': os.environ.get('MYSQLPASSWORD', ''),
    'database': os.environ.get('MYSQL_DATABASE', 'db_ipm_jatim'),
}
METADATA_PATH = os.path.join('static', 'model_metadata.json')
MODEL_PATH    = os.path.join('static', 'best_gru_model.keras')
CSV_DATASET   = 'IPM Kabupaten_Kota_Prov_Jawa_Timur.csv'

def get_db():
    return mysql.connector.connect(**DB_CONFIG)

# ─── Global retraining state ───────────────────────────────────────────────────
_retrain_state = {
    'status':        'idle',   # idle | running | done | error
    'progress':      0,
    'current_epoch': 0,
    'total_epochs':  200,
    'train_loss':    None,
    'val_loss':      None,
    'mape':          None,
    'mae':           None,
    'rmse':          None,
    'history':       [],
    'log_msg':       '',
    'log_type':      'info',
    'error_msg':     '',
}
_retrain_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════════════════════
# DECORATORS & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'admin_id' not in session:
            flash('Silakan login terlebih dahulu.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def get_klasifikasi_params():
    """
    Ambil parameter klasifikasi dari DB.
    Dipakai oleh SEMUA fungsi yang perlu menentukan kategori IPM.
    Fallback ke nilai BPS standar jika tabel kosong.
    """
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT id_parameter, ambang_bawah, ambang_atas, kategori, warna_label
            FROM parameter_klasifikasi
            ORDER BY ambang_bawah ASC
        """)
        params = cur.fetchall(); cur.close(); conn.close()
        if params:
            return params
    except Exception:
        pass
    # Fallback BPS standar
    return [
        {'id_parameter': None, 'ambang_bawah': 0,  'ambang_atas': 60,  'kategori': 'Rendah',       'warna_label': '#FF0000'},
        {'id_parameter': None, 'ambang_bawah': 60, 'ambang_atas': 70,  'kategori': 'Sedang',       'warna_label': '#FFD700'},
        {'id_parameter': None, 'ambang_bawah': 70, 'ambang_atas': 80,  'kategori': 'Tinggi',       'warna_label': '#228B22'},
        {'id_parameter': None, 'ambang_bawah': 80, 'ambang_atas': 100, 'kategori': 'Sangat Tinggi','warna_label': '#006400'},
    ]


def get_kategori_ipm(nilai: float, params: list = None) -> str:
    """
    Tentukan kategori IPM berdasarkan parameter_klasifikasi dari DB.
    Selalu ambil params terbaru jika tidak disuplai.
    """
    if params is None:
        params = get_klasifikasi_params()
    for p in sorted(params, key=lambda x: float(x['ambang_bawah'])):
        if float(p['ambang_bawah']) <= nilai < float(p['ambang_atas']):
            return p['kategori']
    # Jika melebihi ambang_atas tertinggi, kembalikan kategori terakhir
    if params:
        return sorted(params, key=lambda x: float(x['ambang_bawah']))[-1]['kategori']
    return 'Sangat Tinggi'


def validate_params_no_overlap(params_list: list, exclude_id: int = None) -> tuple:
    """
    Validasi bahwa parameter tidak saling overlap dan tidak ada gap.
    Returns: (is_valid: bool, error_msg: str)
    """
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)
        query = "SELECT id_parameter, ambang_bawah, ambang_atas, kategori FROM parameter_klasifikasi"
        if exclude_id:
            query += f" WHERE id_parameter != {int(exclude_id)}"
        query += " ORDER BY ambang_bawah"
        cur.execute(query)
        existing = cur.fetchall(); cur.close(); conn.close()

        # Merge with incoming params_list for full picture
        all_params = list(existing) + params_list
        all_params.sort(key=lambda x: float(x['ambang_bawah']))

        for i in range(len(all_params) - 1):
            a = all_params[i]
            b = all_params[i + 1]
            if float(a['ambang_atas']) > float(b['ambang_bawah']):
                return False, f"Rentang '{a['kategori']}' ({a['ambang_bawah']}–{a['ambang_atas']}) overlap dengan '{b['kategori']}' ({b['ambang_bawah']}–{b['ambang_atas']})."
        return True, ''
    except Exception as e:
        return False, str(e)


def get_model_stats():
    """Ambil statistik model aktif dari DB atau metadata JSON."""
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT skor_mape, skor_mae, skor_rmse, tgl_latih
            FROM riwayat_model ORDER BY tgl_latih DESC LIMIT 1
        """)
        row = cur.fetchone(); cur.close(); conn.close()
        if row:
            return {
                'mape':      round(row['skor_mape'], 4),
                'mae':       round(row['skor_mae'],  5),
                'rmse':      round(row['skor_rmse'], 5),
                'tgl_latih': row['tgl_latih'].strftime('%d %b %Y %H:%M'),
                'sumber':    'database',
            }
    except Exception:
        pass

    try:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH) as f:
                meta = json.load(f)
            return {
                'mape':      round(meta.get('test_mape_pct', 1.75), 4),
                'mae':       round(meta.get('test_mae_riil', 0.00112), 5),
                'rmse':      round(meta.get('test_rmse_riil', 0.0168), 5),
                'tgl_latih': meta.get('tanggal_latih', '-'),
                'sumber':    'metadata_json',
            }
    except Exception:
        pass

    return {
        'mape': 1.75, 'mae': 0.00112, 'rmse': 0.0168,
        'tgl_latih': 'Model awal (notebook)', 'sumber': 'fallback',
    }


def get_kategori_summary():
    """
    Hitung distribusi kategori IPM dari hasil_prediksi_model (model terbaru).
    Re-klasifikasi berdasarkan parameter terkini agar selalu sinkron.
    """
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)
        params = get_klasifikasi_params()
        cur.execute("""
            SELECT ipm_prediksi FROM hasil_prediksi_model
            WHERE id_model = (SELECT MAX(id_model) FROM hasil_prediksi_model)
        """)
        rows = cur.fetchall(); cur.close(); conn.close()
        summary = {}
        for r in rows:
            kat = get_kategori_ipm(float(r['ipm_prediksi'] or 0), params)
            summary[kat] = summary.get(kat, 0) + 1
        return summary
    except Exception:
        return {}


def get_riwayat_prediksi_dashboard(limit=10):
    """Riwayat uji simulasi terbaru untuk widget dashboard."""
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT hus.id_prediksi, w.nama_wilayah,
                   hus.nilai_prediksi,
                   hus.kategori_ipm,
                   hus.tahun_prediksi,
                   hus.tgl_simulasi
            FROM hasil_uji_simulasi hus
            JOIN wilayah w ON hus.id_wilayah = w.id_wilayah
            ORDER BY hus.tgl_simulasi DESC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall(); cur.close(); conn.close()
        return rows
    except Exception:
        return []


def klasifikasi_pill_class(kategori: str) -> str:
    k = (kategori or '').lower()
    if 'sangat' in k: return 'pill-sangat-tinggi'
    if 'tinggi' in k: return 'pill-tinggi'
    if 'sedang' in k: return 'pill-sedang'
    return 'pill-rendah'

app.jinja_env.filters['pill_class'] = klasifikasi_pill_class


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return redirect(url_for('publik_home'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'admin_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('Username dan password tidak boleh kosong.', 'error')
            return render_template('login.html')

        try:
            conn = get_db(); cur = conn.cursor(dictionary=True)
            cur.execute(
                'SELECT id_admin, username, password, nama_lengkap '
                'FROM admin WHERE username = %s',
                (username,)
            )
            admin = cur.fetchone(); cur.close(); conn.close()

            if admin and check_password_hash(admin['password'], password):
                session.clear()
                session['admin_id']    = admin['id_admin']
                session['username']    = admin['username']
                session['nama_lengkap'] = admin['nama_lengkap']
                flash(f'Selamat datang, {admin["nama_lengkap"]}!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Username atau password salah.', 'error')

        except mysql.connector.Error as e:
            flash(f'Kesalahan koneksi database: {str(e)}', 'error')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('Berhasil logout.', 'info')
    return redirect(url_for('login'))


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template(
        'dashboard.html',
        stats       = get_model_stats(),
        riwayat     = get_riwayat_prediksi_dashboard(10),
        kategori    = get_kategori_summary(),
        nama_admin  = session.get('nama_lengkap', 'Admin'),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MANAJEMEN DATA HISTORIS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/data-indikator')
@login_required
def data_indikator():
    conn = get_db(); cur = conn.cursor(dictionary=True)

    # Daftar wilayah aktif
    cur.execute("""
        SELECT id_wilayah, nama_wilayah
        FROM wilayah
        WHERE is_deleted = 0
        ORDER BY nama_wilayah
    """)
    wilayah_list = cur.fetchall()

    # Daftar semua wilayah (termasuk soft deleted, untuk manajemen wilayah)
    cur.execute("""
        SELECT id_wilayah, nama_wilayah, is_deleted
        FROM wilayah
        ORDER BY nama_wilayah
    """)
    wilayah_all = cur.fetchall()

    filter_wilayah = request.args.get('filter_wilayah', '').strip()
    filter_tahun   = request.args.get('filter_tahun', '').strip()
    page           = max(1, int(request.args.get('page', 1)))
    per_page       = int(request.args.get('per_page', 10))
    per_page       = per_page if per_page in (5, 10, 20, 50, 100) else 10

    where_clauses = ['w.is_deleted = 0']
    params: list = []

    if filter_wilayah:
        where_clauses.append('w.id_wilayah = %s')
        params.append(filter_wilayah)
    if filter_tahun:
        where_clauses.append('ih.tahun = %s')
        params.append(filter_tahun)

    where_sql = ' AND '.join(where_clauses)

    cur.execute(f"""
        SELECT COUNT(*) AS total
        FROM indikator_historis ih
        JOIN wilayah w ON ih.id_wilayah = w.id_wilayah
        WHERE {where_sql}
    """, params)
    total_rows  = cur.fetchone()['total']
    total_pages = max(1, (total_rows + per_page - 1) // per_page)
    page        = min(page, total_pages)
    offset      = (page - 1) * per_page

    cur.execute(f"""
        SELECT ih.id_indikator, w.id_wilayah, w.nama_wilayah,
               ih.tahun, ih.ahh, ih.hls, ih.rls, ih.pengeluaran, ih.ipm_aktual
        FROM indikator_historis ih
        JOIN wilayah w ON ih.id_wilayah = w.id_wilayah
        WHERE {where_sql}
        ORDER BY w.nama_wilayah, ih.tahun DESC
        LIMIT %s OFFSET %s
    """, params + [per_page, offset])
    historis_list = cur.fetchall()

    cur.execute("SELECT DISTINCT tahun FROM indikator_historis ORDER BY tahun DESC")
    tahun_list = [int(str(r['tahun'])[:4]) for r in cur.fetchall()]

    cur.close(); conn.close()

    return render_template(
        'manajemen_data.html',
        wilayah_list   = wilayah_list,
        wilayah_all    = wilayah_all,
        historis_list  = historis_list,
        tahun_list     = tahun_list,
        filter_wilayah = filter_wilayah,
        filter_tahun   = filter_tahun,
        page           = page,
        total_pages    = total_pages,
        total_rows     = total_rows,
        per_page       = per_page,
        nama_admin     = session.get('nama_lengkap', 'Admin'),
    )


# ─── API: Export Excel Data Historis (semua data, ikut filter) ────────────────
@app.route('/api/indikator/export-excel')
@login_required
def api_export_excel():
    """
    Export SEMUA data historis ke JSON untuk diproses SheetJS di frontend.
    Parameter query: filter_wilayah, filter_tahun
    """
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)

        filter_wilayah = request.args.get('filter_wilayah', '').strip()
        filter_tahun   = request.args.get('filter_tahun', '').strip()

        where = ['w.is_deleted = 0']
        params = []
        if filter_wilayah:
            where.append('w.id_wilayah = %s'); params.append(filter_wilayah)
        if filter_tahun:
            where.append('ih.tahun = %s'); params.append(filter_tahun)
        where_sql = ' AND '.join(where)

        cur.execute(f"""
            SELECT w.nama_wilayah, ih.tahun AS tahun,
                   ih.ahh, ih.hls, ih.rls, ih.pengeluaran, ih.ipm_aktual
            FROM indikator_historis ih
            JOIN wilayah w ON ih.id_wilayah = w.id_wilayah
            WHERE {where_sql}
            ORDER BY w.nama_wilayah, ih.tahun
        """, params)
        rows = cur.fetchall(); cur.close(); conn.close()

        # Konversi decimal/date ke tipe Python biasa
        data = []
        for r in rows:
            data.append({
                'Kabupaten/Kota':            r['nama_wilayah'],
                'Tahun':                     int(str(r['tahun'])[:4]),
                'AHH':                       float(r['ahh'])         if r['ahh']         else None,
                'HLS':                       float(r['hls'])         if r['hls']         else None,
                'RLS':                       float(r['rls'])         if r['rls']         else None,
                'Pengeluaran per Kapita (Rp)': float(r['pengeluaran']) if r['pengeluaran'] else None,
                'IPM Aktual':                float(r['ipm_aktual'])  if r['ipm_aktual']  else None,
            })

        return jsonify({'ok': True, 'data': data, 'total': len(data)})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ─── API: Tambah Wilayah ───────────────────────────────────────────────────────
@app.route('/api/wilayah/tambah', methods=['POST'])
@login_required
def api_tambah_wilayah():
    data = request.get_json(force=True)
    nama = (data.get('nama_wilayah') or '').strip()
    if not nama:
        return jsonify({'ok': False, 'msg': 'Nama wilayah tidak boleh kosong.'}), 400
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute(
            "INSERT INTO wilayah (nama_wilayah, is_deleted) VALUES (%s, 0)",
            (nama,)
        )
        new_id = cur.lastrowid
        conn.commit(); cur.close(); conn.close()
        return jsonify({'ok': True, 'id_wilayah': new_id, 'nama_wilayah': nama})
    except mysql.connector.IntegrityError:
        return jsonify({'ok': False, 'msg': 'Nama wilayah sudah ada.'}), 409
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ─── API: Edit Nama Wilayah ────────────────────────────────────────────────────
@app.route('/api/wilayah/edit/<int:id_wilayah>', methods=['PUT'])
@login_required
def api_edit_wilayah(id_wilayah):
    data = request.get_json(force=True)
    nama = (data.get('nama_wilayah') or '').strip()
    if not nama:
        return jsonify({'ok': False, 'msg': 'Nama wilayah tidak boleh kosong.'}), 400
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute(
            "UPDATE wilayah SET nama_wilayah = %s WHERE id_wilayah = %s",
            (nama, id_wilayah)
        )
        conn.commit(); cur.close(); conn.close()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ─── API: Soft Delete Wilayah ──────────────────────────────────────────────────
@app.route('/api/wilayah/hapus/<int:id_wilayah>', methods=['DELETE'])
@login_required
def api_hapus_wilayah(id_wilayah):
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute(
            "UPDATE wilayah SET is_deleted = 1 WHERE id_wilayah = %s",
            (id_wilayah,)
        )
        conn.commit(); cur.close(); conn.close()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ─── API: Restore Wilayah (undo soft delete) ───────────────────────────────────
@app.route('/api/wilayah/restore/<int:id_wilayah>', methods=['PUT'])
@login_required
def api_restore_wilayah(id_wilayah):
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute(
            "UPDATE wilayah SET is_deleted = 0 WHERE id_wilayah = %s",
            (id_wilayah,)
        )
        conn.commit(); cur.close(); conn.close()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ─── API: Edit Data Indikator ──────────────────────────────────────────────────
@app.route('/api/indikator/edit/<int:id_indikator>', methods=['PUT'])
@login_required
def api_edit_indikator(id_indikator):
    data = request.get_json(force=True)
    try:
        ahh         = float(data.get('ahh'))
        hls         = float(data.get('hls'))
        rls         = float(data.get('rls'))
        pengeluaran = float(data.get('pengeluaran'))
        ipm_aktual  = float(data.get('ipm_aktual'))

        # Validasi range sederhana
        if not (0 < ahh < 120):
            return jsonify({'ok': False, 'msg': 'AHH tidak valid (0–120).'}), 400
        if not (0 < hls < 30):
            return jsonify({'ok': False, 'msg': 'HLS tidak valid (0–30).'}), 400
        if not (0 < rls < 20):
            return jsonify({'ok': False, 'msg': 'RLS tidak valid (0–20).'}), 400
        if pengeluaran <= 0:
            return jsonify({'ok': False, 'msg': 'Pengeluaran harus > 0.'}), 400
        if not (0 < ipm_aktual < 100):
            return jsonify({'ok': False, 'msg': 'IPM Aktual tidak valid (0–100).'}), 400

        conn = get_db(); cur = conn.cursor()
        cur.execute("""
            UPDATE indikator_historis
            SET ahh = %s, hls = %s, rls = %s,
                pengeluaran = %s, ipm_aktual = %s
            WHERE id_indikator = %s
        """, (ahh, hls, rls, pengeluaran, ipm_aktual, id_indikator))
        conn.commit(); cur.close(); conn.close()
        return jsonify({'ok': True})
    except (TypeError, ValueError) as e:
        return jsonify({'ok': False, 'msg': f'Data tidak valid: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ─── API: Tambah Data Satu Tahun (semua wilayah aktif) ────────────────────────
@app.route('/api/indikator/tambah-tahun', methods=['POST'])
@login_required
def api_tambah_tahun():
    """
    Tambah data satu tahun untuk semua wilayah aktif sekaligus.
    Body: { tahun: int, data: [{id_wilayah, ahh, hls, rls, pengeluaran, ipm_aktual}, ...] }
    Wilayah yang ada di data tapi sudah di-soft-delete akan diabaikan.
    """
    data = request.get_json(force=True)
    tahun = data.get('tahun')
    rows  = data.get('data', [])

    if not tahun or not isinstance(rows, list) or not rows:
        return jsonify({'ok': False, 'msg': 'Data tidak lengkap.'}), 400

    try:
        tahun = int(tahun)
        conn = get_db(); cur = conn.cursor(dictionary=True)

        # Ambil semua wilayah aktif
        cur.execute("SELECT id_wilayah FROM wilayah WHERE is_deleted = 0")
        aktif_ids = {r['id_wilayah'] for r in cur.fetchall()}

        inserted = 0; updated = 0; errors = []
        for r in rows:
            id_wil = int(r.get('id_wilayah', 0))
            if id_wil not in aktif_ids:
                continue  # skip wilayah non-aktif

            try:
                ahh         = float(r['ahh'])
                hls         = float(r['hls'])
                rls         = float(r['rls'])
                pengeluaran = float(r['pengeluaran'])
                ipm_aktual  = float(r['ipm_aktual'])
            except (TypeError, ValueError) as ve:
                errors.append(f"id_wilayah={id_wil}: {ve}")
                continue

            cur.execute("""
                INSERT INTO indikator_historis
                    (id_wilayah, tahun, ahh, hls, rls, pengeluaran, ipm_aktual)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    ahh=%s, hls=%s, rls=%s, pengeluaran=%s, ipm_aktual=%s
            """, (
                id_wil, tahun, ahh, hls, rls, pengeluaran, ipm_aktual,
                ahh, hls, rls, pengeluaran, ipm_aktual,
            ))
            if cur.rowcount == 1:
                inserted += 1
            else:
                updated += 1

        conn.commit(); cur.close(); conn.close()

        msg = f'{inserted} baris ditambah, {updated} baris diperbarui.'
        if errors:
            msg += f' {len(errors)} baris gagal: ' + '; '.join(errors[:3])

        return jsonify({'ok': True, 'msg': msg, 'inserted': inserted, 'updated': updated})

    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ─── API: Import CSV & Trigger Retraining ─────────────────────────────────────
@app.route('/api/indikator/import-csv', methods=['POST'])
@login_required
def api_import_csv():
    """
    Format CSV (separator titik koma):
    Kabupaten/Kota;Tahun;AHH;HLS;RLS;Pengeluaran per Kapita Riil (Rp);IPM

    INSERT ... ON DUPLICATE KEY UPDATE — tidak ada data yang hilang.
    Setelah import selesai, redirect ke halaman retraining.
    """
    if 'file' not in request.files:
        return jsonify({'ok': False, 'msg': 'File tidak ditemukan.'}), 400

    f = request.files['file']
    if not f.filename.endswith('.csv'):
        return jsonify({'ok': False, 'msg': 'Hanya file .csv yang diterima.'}), 400

    try:
        content = f.read().decode('utf-8-sig')
        reader  = csv.DictReader(io.StringIO(content), delimiter=';')
        reader.fieldnames = [h.strip() for h in (reader.fieldnames or [])]

        conn = get_db(); cur = conn.cursor(dictionary=True)

        # Cache nama wilayah → id_wilayah (hanya yang aktif)
        cur.execute("SELECT id_wilayah, nama_wilayah FROM wilayah WHERE is_deleted = 0")
        wilayah_map = {
            r['nama_wilayah'].strip().lower(): r['id_wilayah']
            for r in cur.fetchall()
        }

        inserted = updated = skipped = 0

        for row in reader:
            nama  = (row.get('Kabupaten/Kota') or '').strip()
            tahun = (row.get('Tahun') or '').strip()

            id_wil = wilayah_map.get(nama.lower())
            if not id_wil or not tahun:
                skipped += 1
                continue

            try:
                ahh         = float((row.get('AHH') or '').replace(',', '.'))
                hls         = float((row.get('HLS') or '').replace(',', '.'))
                rls         = float((row.get('RLS') or '').replace(',', '.'))
                # Pengeluaran bisa dalam format "9.841,00" atau "9841.00" atau "9841"
                pen_raw     = (row.get('Pengeluaran per Kapita Riil (Rp)') or '').strip()
                # Hapus titik ribuan, ganti koma desimal jadi titik
                if ',' in pen_raw:
                    pen_raw = pen_raw.replace('.', '').replace(',', '.')
                else:
                    pen_raw = pen_raw.replace(',', '')
                pengeluaran = float(pen_raw)
                ipm         = float((row.get('IPM') or '').replace(',', '.'))
            except (ValueError, TypeError):
                skipped += 1
                continue

            cur.execute("""
                INSERT INTO indikator_historis
                    (id_wilayah, tahun, ahh, hls, rls, pengeluaran, ipm_aktual)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    ahh=%s, hls=%s, rls=%s, pengeluaran=%s, ipm_aktual=%s
            """, (
                id_wil, tahun, ahh, hls, rls, pengeluaran, ipm,
                ahh, hls, rls, pengeluaran, ipm,
            ))
            if cur.rowcount == 1:
                inserted += 1
            else:
                updated += 1

        conn.commit(); cur.close(); conn.close()

        return jsonify({
            'ok':       True,
            'inserted': inserted,
            'updated':  updated,
            'skipped':  skipped,
            'msg':      f'{inserted} baris ditambah, {updated} diperbarui, {skipped} dilewati.',
            'redirect': url_for('retraining', auto_start=1),
        })

    except Exception as e:
        return jsonify({'ok': False, 'msg': f'Gagal memproses CSV: {str(e)}'}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RIWAYAT PREDIKSI
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/riwayat-prediksi')
@login_required
def riwayat_prediksi():
    conn = get_db(); cur = conn.cursor(dictionary=True)

    filter_wilayah  = request.args.get('filter_wilayah', '').strip()
    filter_tahun    = request.args.get('filter_tahun', '').strip()
    filter_kategori = request.args.get('filter_kategori', '').strip()
    search          = request.args.get('search', '').strip()
    page            = max(1, int(request.args.get('page', 1)))
    per_page        = int(request.args.get('per_page', 10))
    per_page        = per_page if per_page in (5, 10, 20, 50, 100) else 10

    where_clauses = ['w.is_deleted = 0']
    params: list  = []

    if filter_wilayah:
        where_clauses.append('hus.id_wilayah = %s')
        params.append(filter_wilayah)
    if filter_tahun:
        where_clauses.append('hus.tahun_prediksi = %s')
        params.append(filter_tahun)
    if filter_kategori:
        where_clauses.append('hus.kategori_ipm LIKE %s')
        params.append(f'%{filter_kategori}%')
    if search:
        where_clauses.append('w.nama_wilayah LIKE %s')
        params.append(f'%{search}%')

    where_sql = ' AND '.join(where_clauses)

    cur.execute(f"""
        SELECT COUNT(*) AS total
        FROM hasil_uji_simulasi hus
        JOIN wilayah w ON hus.id_wilayah = w.id_wilayah
        WHERE {where_sql}
    """, params)
    total_rows  = cur.fetchone()['total']
    total_pages = max(1, (total_rows + per_page - 1) // per_page)
    page        = min(page, total_pages)
    offset      = (page - 1) * per_page

    cur.execute(f"""
        SELECT hus.id_prediksi, w.id_wilayah, w.nama_wilayah,
               hus.nilai_prediksi,
               hus.kategori_ipm,
               hus.tahun_prediksi,
               hus.tgl_simulasi,
               hus.data_sequence
        FROM hasil_uji_simulasi hus
        JOIN wilayah w ON hus.id_wilayah = w.id_wilayah
        WHERE {where_sql}
        ORDER BY hus.tgl_simulasi DESC, hus.tahun_prediksi DESC
        LIMIT %s OFFSET %s
    """, params + [per_page, offset])
    riwayat = cur.fetchall()

    cur.execute("SELECT id_wilayah, nama_wilayah FROM wilayah WHERE is_deleted = 0 ORDER BY nama_wilayah")
    wilayah_list = cur.fetchall()

    cur.execute("SELECT DISTINCT tahun_prediksi FROM hasil_uji_simulasi ORDER BY tahun_prediksi DESC")
    tahun_list = [r['tahun_prediksi'] for r in cur.fetchall()]

    cur.execute("SELECT DISTINCT kategori_ipm FROM hasil_uji_simulasi WHERE kategori_ipm IS NOT NULL ORDER BY kategori_ipm")
    kategori_list = [r['kategori_ipm'] for r in cur.fetchall()]

    cur.execute("""
        SELECT kategori_ipm, COUNT(*) AS jumlah
        FROM hasil_uji_simulasi
        WHERE kategori_ipm IS NOT NULL
        GROUP BY kategori_ipm
    """)
    kategori = {r['kategori_ipm']: r['jumlah'] for r in cur.fetchall()}

    cur.close(); conn.close()

    return render_template(
        'riwayat_prediksi.html',
        riwayat         = riwayat,
        wilayah_list    = wilayah_list,
        tahun_list      = tahun_list,
        kategori_list   = kategori_list,
        kategori        = kategori,
        filter_wilayah  = filter_wilayah,
        filter_tahun    = filter_tahun,
        filter_kategori = filter_kategori,
        search          = search,
        page            = page,
        total_pages     = total_pages,
        total_rows      = total_rows,
        per_page        = per_page,
        nama_admin      = session.get('nama_lengkap', 'Admin'),
    )


# ─── API: Hapus Riwayat Prediksi ──────────────────────────────────────────────
@app.route('/api/riwayat-prediksi/<int:id_prediksi>', methods=['DELETE'])
@login_required
def api_hapus_riwayat_prediksi(id_prediksi):
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("DELETE FROM hasil_uji_simulasi WHERE id_prediksi = %s", (id_prediksi,))
        if cur.rowcount == 0:
            cur.close(); conn.close()
            return jsonify({'ok': False, 'msg': 'Data tidak ditemukan.'}), 404
        conn.commit(); cur.close(); conn.close()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DETAIL PREDIKSI
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/detail-prediksi/<int:id_prediksi>')
def detail_prediksi(id_prediksi):
    """
    Halaman detail prediksi — bisa diakses publik maupun admin.
    Query param ?from=publik → navbar publik, kembali ke /history publik
    Query param ?from=admin  → navbar admin, kembali ke /riwayat-prediksi (butuh login)
    Default: auto-detect dari session
    """
    from_ctx = request.args.get('from', '').strip()

    # Jika context admin tapi belum login → redirect login
    if from_ctx == 'admin' and 'admin_id' not in session:
        flash('Silakan login terlebih dahulu.', 'warning')
        return redirect(url_for('login'))

    # Auto-detect: jika ada session admin dan tidak ada from param → anggap admin
    if not from_ctx:
        from_ctx = 'admin' if 'admin_id' in session else 'publik'

    is_publik = (from_ctx == 'publik')

    # Back URL sesuai context
    if is_publik:
        back_url   = url_for('publik_history')
        back_label = 'Riwayat Simulasi'
    else:
        back_url   = url_for('riwayat_prediksi')
        back_label = 'Riwayat Prediksi'

    conn = get_db(); cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT hus.id_prediksi, hus.tahun_prediksi, hus.nilai_prediksi,
               hus.kategori_ipm, hus.tgl_simulasi, hus.id_model,
               hus.data_sequence,
               w.id_wilayah, w.nama_wilayah, w.deskripsi,
               w.url_logo, w.url_landmark
        FROM hasil_uji_simulasi hus
        JOIN wilayah w ON hus.id_wilayah = w.id_wilayah
        WHERE hus.id_prediksi = %s
    """, (id_prediksi,))
    prediksi = cur.fetchone()

    if not prediksi:
        cur.close(); conn.close()
        abort(404)

    data_sequence = []
    try:
        data_sequence = json.loads(prediksi['data_sequence']) if prediksi['data_sequence'] else []
    except (json.JSONDecodeError, TypeError):
        data_sequence = []

    cur.execute("""
        SELECT tahun, ipm_aktual, ahh, hls, rls, pengeluaran
        FROM indikator_historis
        WHERE id_wilayah = %s
        ORDER BY tahun ASC
    """, (prediksi['id_wilayah'],))
    historis_all = cur.fetchall()

    cur.execute("""
        SELECT tahun_prediksi, nilai_prediksi
        FROM hasil_uji_simulasi
        WHERE id_wilayah = %s
        ORDER BY tahun_prediksi ASC
    """, (prediksi['id_wilayah'],))
    prediksi_all = cur.fetchall()

    # Ambil skor model — selalu dari kolom TEST (skor_mape/mae/rmse)
    # Prioritas: id_model FK di record simulasi, fallback ke MAX(id_model)
    stats = {}
    try:
        id_model_target = prediksi.get('id_model')
        if not id_model_target:
            cur.execute("SELECT MAX(id_model) AS id_model_max FROM riwayat_model")
            row_max = cur.fetchone()
            id_model_target = row_max['id_model_max'] if row_max else None

        if id_model_target:
            cur.execute("""
                SELECT id_model, skor_mape, skor_mae, skor_rmse, tgl_latih
                FROM riwayat_model
                WHERE id_model = %s
            """, (id_model_target,))
            model_row = cur.fetchone()
            if model_row:
                stats = {
                    'id_model':  model_row['id_model'],
                    'mape':      round(float(model_row['skor_mape']), 4) if model_row['skor_mape'] is not None else None,
                    'mae':       round(float(model_row['skor_mae']),  4) if model_row['skor_mae']  is not None else None,
                    'rmse':      round(float(model_row['skor_rmse']), 4) if model_row['skor_rmse'] is not None else None,
                    'tgl_latih': model_row['tgl_latih'].strftime('%d %b %Y %H:%M') if model_row['tgl_latih'] else '—',
                    'sumber':    'database',
                }
    except Exception as e:
        logger.warning(f'detail_prediksi stats error: {e}')

    if not stats:
        stats = get_model_stats()

    cur.close(); conn.close()

    chart_labels   = [str(r['tahun']) for r in historis_all]
    chart_aktual   = [float(r['ipm_aktual']) if r['ipm_aktual'] else None for r in historis_all]
    pred_map       = {r['tahun_prediksi']: float(r['nilai_prediksi']) for r in prediksi_all}
    chart_prediksi = [pred_map.get(r['tahun']) for r in historis_all]

    if prediksi['tahun_prediksi'] not in [r['tahun'] for r in historis_all]:
        chart_labels.append(str(prediksi['tahun_prediksi']))
        chart_aktual.append(None)
        chart_prediksi.append(float(prediksi['nilai_prediksi']))

    return render_template(
        'detail_prediksi.html',
        prediksi        = prediksi,
        data_sequence   = data_sequence,
        historis        = historis_all,
        stats           = stats,
        chart_labels    = json.dumps(chart_labels),
        chart_actuals   = json.dumps(chart_aktual),
        chart_prediksi  = json.dumps(chart_prediksi),
        nama_admin      = session.get('nama_lengkap', 'Admin'),
        is_publik       = is_publik,
        back_url        = back_url,
        back_label      = back_label,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RETRAINING MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/retraining')
@login_required
def retraining():
    auto_start = request.args.get('auto_start', 0, type=int)
    stats = get_model_stats()

    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)

        # ── Hasil prediksi model (id_model terbaru) ───────────────────────
        cur.execute("""
            SELECT w.nama_wilayah,
                   hp.ipm_aktual,
                   hp.ipm_prediksi,
                   hp.error_persen,
                   hp.kategori,
                   hp.id_model,
                   rm.tgl_latih AS tgl_model
            FROM hasil_prediksi_model hp
            JOIN wilayah w  ON hp.id_wilayah = w.id_wilayah
            JOIN riwayat_model rm ON hp.id_model = rm.id_model
            WHERE hp.id_model = (
                SELECT MAX(id_model) FROM hasil_prediksi_model
            )
            ORDER BY hp.ipm_prediksi DESC
        """)
        hasil_prediksi_list = cur.fetchall()

        # ── Riwayat training — dari riwayat_model (5 terbaru) ─────────────
        cur.execute("""
            SELECT id_model, tgl_latih,
                   skor_mape, mape_train,
                   skor_mae,  mae_train,
                   skor_rmse, rmse_train,
                   loss_curve
            FROM riwayat_model
            ORDER BY tgl_latih DESC LIMIT 5
        """)
        riwayat_model_list = cur.fetchall()

        # Ambil id_model terbaru untuk info cetak
        latest_model_id = hasil_prediksi_list[0]['id_model'] if hasil_prediksi_list else None
        latest_tgl      = hasil_prediksi_list[0]['tgl_model'] if hasil_prediksi_list else None

        # Tahun prediksi = max tahun indikator historis + 1
        cur2 = conn.cursor(dictionary=True)
        cur2.execute("SELECT MAX(tahun) AS max_thn FROM indikator_historis")
        row_thn = cur2.fetchone()
        tahun_prediksi = (int(str(row_thn['max_thn'])[:4]) + 1) if row_thn and row_thn['max_thn'] else None
        cur2.close()

        cur.close(); conn.close()
    except Exception as e:
        logger.warning(f'retraining query error: {e}')
        hasil_prediksi_list = []
        riwayat_model_list  = []
        latest_model_id     = None
        latest_tgl          = None
        tahun_prediksi      = None

    return render_template(
        'retraining.html',
        auto_start          = auto_start,
        stats               = stats,
        hasil_prediksi_list = hasil_prediksi_list,
        riwayat_model_list  = riwayat_model_list,
        latest_model_id     = latest_model_id,
        latest_tgl          = latest_tgl,
        tahun_prediksi      = tahun_prediksi,
        nama_admin          = session.get('nama_lengkap', 'Admin'),
    )


# ─── API: Latest Pred Table ────────────────────────────────────────────────────
@app.route('/api/retraining/latest-pred')
@login_required
def api_retraining_latest_pred():
    """Refresh tabel prediksi di halaman retraining setelah training selesai."""
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)
        params = get_klasifikasi_params()
        cur.execute("""
            SELECT w.nama_wilayah,
                   hp.ipm_aktual, hp.ipm_prediksi, hp.error_persen,
                   hp.kategori, hp.id_model,
                   rm.tgl_latih AS tgl_model
            FROM hasil_prediksi_model hp
            JOIN wilayah w ON hp.id_wilayah = w.id_wilayah
            JOIN riwayat_model rm ON hp.id_model = rm.id_model
            WHERE hp.id_model = (
                SELECT MAX(id_model) FROM hasil_prediksi_model
            )
            ORDER BY hp.ipm_prediksi DESC
        """)
        rows = cur.fetchall()
        # Re-apply kategori dari params terkini
        for r in rows:
            if r.get('ipm_prediksi'):
                r['kategori'] = get_kategori_ipm(float(r['ipm_prediksi']), params)
            # Convert datetime to string for JSON
            if r.get('tgl_model') and hasattr(r['tgl_model'], 'strftime'):
                r['tgl_model'] = r['tgl_model'].strftime('%d %b %Y %H:%M')
        cur.close(); conn.close()
        # Hitung tahun prediksi dari indikator historis
        conn2 = get_db(); cur2 = conn2.cursor(dictionary=True)
        cur2.execute("SELECT MAX(tahun) AS max_thn FROM indikator_historis")
        r = cur2.fetchone()
        tahun_pred = (int(str(r['max_thn'])[:4]) + 1) if r and r['max_thn'] else None
        cur2.close(); conn2.close()
        return jsonify({'ok': True, 'data': rows, 'tahun_prediksi': tahun_pred})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ─── API: Cetak PDF Hasil Prediksi Model ──────────────────────────────────────
@app.route('/api/retraining/cetak-pdf')
@login_required
def api_cetak_pdf():
    """
    Mengembalikan data JSON semua hasil prediksi model terbaru
    untuk di-render menjadi PDF di frontend (via jsPDF / print).
    """
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)
        params = get_klasifikasi_params()

        cur.execute("""
            SELECT w.nama_wilayah,
                   hp.ipm_aktual, hp.ipm_prediksi, hp.error_persen,
                   hp.kategori, hp.id_model,
                   rm.tgl_latih AS tgl_model
            FROM hasil_prediksi_model hp
            JOIN wilayah w ON hp.id_wilayah = w.id_wilayah
            JOIN riwayat_model rm ON hp.id_model = rm.id_model
            WHERE hp.id_model = (
                SELECT MAX(id_model) FROM hasil_prediksi_model
            )
            ORDER BY hp.ipm_prediksi DESC
        """)
        rows = cur.fetchall()
        for r in rows:
            if r.get('ipm_prediksi'):
                r['kategori'] = get_kategori_ipm(float(r['ipm_prediksi']), params)
            if r.get('tgl_model') and hasattr(r['tgl_model'], 'strftime'):
                r['tgl_model'] = r['tgl_model'].strftime('%d %B %Y %H:%M')

        # Ringkasan distribusi kategori
        distribusi = {}
        for r in rows:
            kat = r.get('kategori', '—')
            distribusi[kat] = distribusi.get(kat, 0) + 1

        # Info model
        model_info = {}
        if rows:
            model_info = {
                'id_model': rows[0]['id_model'],
                'tgl_latih': rows[0]['tgl_model'],
                'jumlah_wilayah': len(rows),
            }

        cur.close(); conn.close()
        return jsonify({
            'ok': True,
            'data': rows,
            'distribusi': distribusi,
            'model_info': model_info,
        })
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ─── API: Progress Retraining (Long Polling) ──────────────────────────────────
@app.route('/api/retraining/progress')
@login_required
def api_retraining_progress():
    with _retrain_lock:
        state = dict(_retrain_state)
    state['ok'] = True
    return jsonify(state)


# ─── API: Reset State Retraining ──────────────────────────────────────────────
@app.route('/api/retraining/reset', methods=['POST'])
@login_required
def api_retraining_reset():
    global _retrain_state
    with _retrain_lock:
        if _retrain_state['status'] == 'running':
            return jsonify({'ok': False, 'msg': 'Tidak dapat reset saat training berjalan.'}), 409
        _retrain_state.update({
            'status': 'idle', 'progress': 0, 'current_epoch': 0,
            'train_loss': None, 'val_loss': None, 'mape': None, 'mae': None, 'rmse': None,
            'history': [], 'log_msg': '', 'log_type': 'info', 'error_msg': '',
        })
    return jsonify({'ok': True})


# ─── API: Mulai Retraining (async via thread) ─────────────────────────────────
@app.route('/api/retraining/start', methods=['POST'])
@login_required
def api_retraining_start():
    global _retrain_state

    with _retrain_lock:
        if _retrain_state['status'] == 'running':
            return jsonify({'ok': False, 'msg': 'Training sedang berjalan.'}), 409
        _retrain_state.update({
            'status': 'running', 'progress': 0, 'current_epoch': 0,
            'total_epochs': 200, 'train_loss': None, 'val_loss': None,
            'mape': None, 'mae': None, 'rmse': None,
            'history': [], 'log_msg': 'Mempersiapkan data...', 'log_type': 'info', 'error_msg': '',
        })

    t = threading.Thread(target=_run_retraining, daemon=True)
    t.start()
    return jsonify({'ok': True, 'total_epochs': 200})


def _update(**kwargs):
    with _retrain_lock:
        _retrain_state.update(kwargs)


def _run_retraining():
    """Thread utama proses training."""
    try:
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler, LabelEncoder
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, Input
        from tensorflow.keras.callbacks import (
            EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
        )

        SEQUENCE_LEN = 3
        features = ['ahh', 'hls', 'rls', 'pengeluaran', 'wilayah_enc']

        _update(log_msg='Mengambil data dari database...', log_type='info', progress=5)

        conn = get_db(); cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT w.id_wilayah, w.nama_wilayah,
                   ih.tahun, ih.ahh, ih.hls, ih.rls,
                   ih.pengeluaran, ih.ipm_aktual
            FROM indikator_historis ih
            JOIN wilayah w ON ih.id_wilayah = w.id_wilayah
            WHERE w.is_deleted = 0 AND ih.ipm_aktual IS NOT NULL
            ORDER BY w.nama_wilayah, ih.tahun
        """)
        rows = cur.fetchall()
        cur.close(); conn.close()

        df = pd.DataFrame(rows)
        _update(log_msg=f'Data loaded: {len(df)} baris, {df["id_wilayah"].nunique()} wilayah.', log_type='info', progress=10)

        le = LabelEncoder()
        df['wilayah_enc'] = le.fit_transform(df['nama_wilayah'])

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(df[features])
        y_scaled = scaler_y.fit_transform(df[['ipm_aktual']])
        _update(log_msg='Normalisasi selesai.', log_type='info', progress=15)

        def build_sequences(X, y, seq_len):
            Xs, ys = [], []
            unique_wilayah = df['wilayah_enc'].unique()
            for wil in unique_wilayah:
                mask = df['wilayah_enc'] == wil
                X_w  = X[mask]
                y_w  = y[mask]
                for i in range(len(X_w) - seq_len):
                    Xs.append(X_w[i:i+seq_len])
                    ys.append(y_w[i+seq_len])
            return np.array(Xs), np.array(ys)

        X_seq, y_seq = build_sequences(X_scaled, y_scaled, SEQUENCE_LEN)
        _update(log_msg=f'Sequences dibuat: {X_seq.shape[0]} sampel.', log_type='info', progress=18)

        if len(X_seq) == 0:
            _update(status='error', error_msg='Data tidak cukup untuk membuat sequence.', log_msg='❌ Data terlalu sedikit.', log_type='error')
            return

        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        _update(log_msg=f'Split data: train={len(X_train)}, test={len(X_test)}.', log_type='info', progress=22)

        TOTAL_EPOCHS = 200
        _update(total_epochs=TOTAL_EPOCHS)

        model = Sequential([
            Input(shape=(SEQUENCE_LEN, len(features))),
            Bidirectional(GRU(64, return_sequences=True)),
            Dropout(0.2),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1),
        ])
        model.compile(optimizer='adam', loss='mse')
        _update(log_msg='Arsitektur model GRU Bidirectional berhasil dibangun.', log_type='success', progress=26)

        class ProgressCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                progress = 26 + int(((epoch + 1) / TOTAL_EPOCHS) * 68)
                history_entry = {
                    'train_loss': round(float(logs.get('loss', 0)), 6),
                    'val_loss':   round(float(logs.get('val_loss', 0)), 6),
                }
                with _retrain_lock:
                    _retrain_state['history'].append(history_entry)
                    _retrain_state.update({
                        'progress':      progress,
                        'current_epoch': epoch + 1,
                        'train_loss':    history_entry['train_loss'],
                        'val_loss':      history_entry['val_loss'],
                        'log_msg':       f'Epoch {epoch+1}/{TOTAL_EPOCHS} — loss: {history_entry["train_loss"]:.5f}, val_loss: {history_entry["val_loss"]:.5f}',
                        'log_type':      'info',
                    })

        callbacks = [
            ProgressCallback(),
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
            ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0),
        ]

        _update(log_msg='Memulai training model...', log_type='warn', progress=27)

        model.fit(
            X_train, y_train,
            validation_data = (X_test, y_test),
            epochs          = TOTAL_EPOCHS,
            batch_size      = 16,
            callbacks       = callbacks,
            verbose         = 0,
        )

        # ── Evaluasi Testing ─────────────────────────────────────────────────
        _update(log_msg='Training selesai. Mengevaluasi model (testing)...', log_type='success', progress=93)
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred_test = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_true_test = scaler_y.inverse_transform(y_test).flatten()
        mae_test  = float(mean_absolute_error(y_true_test, y_pred_test))
        rmse_test = float(np.sqrt(mean_squared_error(y_true_test, y_pred_test)))
        mape_test = float(np.mean(np.abs((y_true_test - y_pred_test) / (y_true_test + 1e-8))) * 100)

        # ── Evaluasi Training ─────────────────────────────────────────────────
        _update(log_msg='Mengevaluasi model (training)...', log_type='info', progress=95)
        y_pred_train_scaled = model.predict(X_train, verbose=0)
        y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled).flatten()
        y_true_train = scaler_y.inverse_transform(y_train).flatten()
        mae_train  = float(mean_absolute_error(y_true_train, y_pred_train))
        rmse_train = float(np.sqrt(mean_squared_error(y_true_train, y_pred_train)))
        mape_train = float(np.mean(np.abs((y_true_train - y_pred_train) / (y_true_train + 1e-8))) * 100)

        _update(
            mape     = round(mape_test, 4),
            mae      = round(mae_test,  6),
            rmse     = round(rmse_test, 6),
            log_msg  = f'Testing — MAPE: {mape_test:.4f}%, MAE: {mae_test:.6f}, RMSE: {rmse_test:.6f}',
            log_type = 'success',
            progress = 97,
        )

        # ── Simpan metadata ───────────────────────────────────────────────────
        import datetime
        now_dt  = datetime.datetime.now()
        now_str = now_dt.strftime('%Y-%m-%d %H:%M:%S')

        metadata = {
            'test_mape_pct':   round(mape_test, 4),
            'test_mae_riil':   round(mae_test, 6),
            'test_rmse_riil':  round(rmse_test, 6),
            'train_mape_pct':  round(mape_train, 4),
            'train_mae_riil':  round(mae_train, 6),
            'train_rmse_riil': round(rmse_train, 6),
            'tanggal_latih':   now_str,
            'total_epochs':    TOTAL_EPOCHS,
        }
        with open(METADATA_PATH, 'w') as mf:
            json.dump(metadata, mf, indent=2)

        # Simpan ke tabel riwayat_model
        new_model_id = None
        try:
            conn = get_db(); cur = conn.cursor()
            # Simpan loss history sebagai JSON untuk loss_curve
            with _retrain_lock:
                history_data = list(_retrain_state.get('history', []))
            loss_curve_json = json.dumps({
                'loss':     [h['train_loss'] for h in history_data],
                'val_loss': [h['val_loss']   for h in history_data],
            })
            cur.execute("""
                INSERT INTO riwayat_model
                    (tgl_latih, skor_mape, skor_mae, skor_rmse,
                     mape_train, mae_train, rmse_train,
                     file_model, loss_curve)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                now_dt,
                round(mape_test,  4), round(mae_test,  6), round(rmse_test,  6),
                round(mape_train, 4), round(mae_train, 6), round(rmse_train, 6),
                MODEL_PATH, loss_curve_json,
            ))
            new_model_id = cur.lastrowid
            conn.commit(); cur.close(); conn.close()
        except Exception as e:
            logger.warning(f'Gagal simpan riwayat_model: {e}')

        # ── Generate prediksi untuk semua wilayah ────────────────────────────
        _update(log_msg='Menghasilkan prediksi baru untuk semua wilayah...', log_type='info', progress=98)
        try:
            _generate_predictions_all(
                model, scaler_X, scaler_y, le,
                features, SEQUENCE_LEN, new_model_id
            )
        except Exception as e:
            logger.warning(f'Gagal generate prediksi: {e}')

        _update(
            status   = 'done',
            progress = 100,
            log_msg  = '✅ Model berhasil disimpan ke best_gru_model.keras',
            log_type = 'success',
        )

    except ImportError as e:
        _update(status='error', error_msg=f'Library tidak tersedia: {str(e)}',
                log_msg=f'❌ Import error: {str(e)}', log_type='error')
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f'Retraining error:\n{tb}')
        _update(status='error', error_msg=str(e),
                log_msg=f'❌ Error: {str(e)}', log_type='error')


def _generate_predictions_all(model, scaler_X, scaler_y, le,
                               features, seq_len, id_model):
    """
    Hasilkan prediksi tahun depan untuk semua wilayah aktif.
    Kategori ditentukan dari parameter_klasifikasi terkini di DB.
    """
    import numpy as np
    import datetime
    import pandas as pd

    conn = get_db(); cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT w.id_wilayah, w.nama_wilayah,
               ih.tahun, ih.ahh, ih.hls, ih.rls,
               ih.pengeluaran, ih.ipm_aktual
        FROM indikator_historis ih
        JOIN wilayah w ON ih.id_wilayah = w.id_wilayah
        WHERE w.is_deleted = 0 AND ih.ipm_aktual IS NOT NULL
        ORDER BY w.nama_wilayah, ih.tahun
    """)
    rows = cur.fetchall()
    df = pd.DataFrame(rows)

    # Ambil parameter klasifikasi dari DB
    params = get_klasifikasi_params()
    tahun_prediksi = int(df['tahun'].max()) + 1
    now = datetime.datetime.now()

    for id_wil in df['id_wilayah'].unique():
        df_wil = df[df['id_wilayah'] == id_wil].sort_values('tahun')
        if len(df_wil) < seq_len:
            continue

        nama_wil = df_wil.iloc[0]['nama_wilayah']
        try:
            nama_enc = le.transform([nama_wil])[0]
        except Exception:
            continue

        last_seq = df_wil.tail(seq_len).copy()
        last_seq['wilayah_enc'] = nama_enc

        X_input = scaler_X.transform(last_seq[features].values)
        X_input = X_input.reshape(1, seq_len, len(features))

        y_pred_scaled = model.predict(X_input, verbose=0)
        y_pred = float(scaler_y.inverse_transform(y_pred_scaled)[0][0])
        # Gunakan parameter klasifikasi terkini dari DB
        kategori = get_kategori_ipm(y_pred, params)

        # Ambil ipm_aktual terakhir untuk error_persen
        ipm_aktual_last = float(df_wil.iloc[-1]['ipm_aktual']) if df_wil.iloc[-1]['ipm_aktual'] else None
        error_persen = None
        if ipm_aktual_last and ipm_aktual_last > 0:
            error_persen = round(abs(y_pred - ipm_aktual_last) / ipm_aktual_last * 100, 2)

        # INSERT baru setiap retraining (bukan upsert) agar riwayat terjaga
        cur.execute("""
            INSERT INTO hasil_prediksi_model
                (id_wilayah, id_model, ipm_aktual, ipm_prediksi, error_persen, kategori)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            int(id_wil), id_model,
            round(ipm_aktual_last, 4) if ipm_aktual_last else None,
            round(y_pred, 4), error_persen, kategori,
        ))

    conn.commit(); cur.close(); conn.close()
    logger.info(f'Prediksi tahun {tahun_prediksi} berhasil disimpan untuk semua wilayah.')


# ═══════════════════════════════════════════════════════════════════════════════
# 5. KONFIGURASI PARAMETER KLASIFIKASI
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/konfigurasi')
@login_required
def konfigurasi():
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT id_parameter, ambang_bawah, ambang_atas, kategori, warna_label
            FROM parameter_klasifikasi ORDER BY ambang_bawah
        """)
        params = cur.fetchall(); cur.close(); conn.close()
    except Exception:
        params = []

    return render_template(
        'konfigurasi.html',
        params     = params,
        nama_admin = session.get('nama_lengkap', 'Admin'),
    )


# ─── API: Tambah Parameter Klasifikasi ────────────────────────────────────────
@app.route('/api/parameter/tambah', methods=['POST'])
@login_required
def api_tambah_parameter():
    data = request.get_json(force=True)
    try:
        bb  = float(data.get('ambang_bawah', 0))
        ba  = float(data.get('ambang_atas',  0))
        kat = str(data.get('kategori', '')).strip()

        if bb >= ba:
            return jsonify({'ok': False, 'msg': 'Batas bawah harus lebih kecil dari batas atas.'}), 400
        if not kat:
            return jsonify({'ok': False, 'msg': 'Nama kategori tidak boleh kosong.'}), 400
        if bb < 0 or ba > 200:
            return jsonify({'ok': False, 'msg': 'Nilai rentang tidak realistis.'}), 400

        # Cek overlap dengan parameter yang sudah ada
        valid, err = validate_params_no_overlap([{'ambang_bawah': bb, 'ambang_atas': ba, 'kategori': kat}])
        if not valid:
            return jsonify({'ok': False, 'msg': f'Terjadi overlap: {err}'}), 400

        conn = get_db(); cur = conn.cursor()
        cur.execute("""
            INSERT INTO parameter_klasifikasi (ambang_bawah, ambang_atas, kategori)
            VALUES (%s, %s, %s)
        """, (bb, ba, kat))
        new_id = cur.lastrowid
        conn.commit(); cur.close(); conn.close()
        return jsonify({'ok': True, 'id_parameter': new_id})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ─── API: Edit Parameter Klasifikasi ──────────────────────────────────────────
@app.route('/api/parameter/edit/<int:id_parameter>', methods=['PUT'])
@login_required
def api_edit_parameter(id_parameter):
    data = request.get_json(force=True)
    try:
        bb  = float(data.get('ambang_bawah', 0))
        ba  = float(data.get('ambang_atas',  0))
        kat = str(data.get('kategori', '')).strip()

        if bb >= ba:
            return jsonify({'ok': False, 'msg': 'Batas bawah harus lebih kecil dari batas atas.'}), 400
        if not kat:
            return jsonify({'ok': False, 'msg': 'Nama kategori tidak boleh kosong.'}), 400
        if bb < 0 or ba > 200:
            return jsonify({'ok': False, 'msg': 'Nilai rentang tidak realistis.'}), 400

        # Cek overlap dengan parameter lain (exclude diri sendiri)
        valid, err = validate_params_no_overlap(
            [{'ambang_bawah': bb, 'ambang_atas': ba, 'kategori': kat}],
            exclude_id=id_parameter
        )
        if not valid:
            return jsonify({'ok': False, 'msg': f'Terjadi overlap: {err}'}), 400

        conn = get_db(); cur = conn.cursor()
        cur.execute("""
            UPDATE parameter_klasifikasi
            SET ambang_bawah = %s, ambang_atas = %s, kategori = %s
            WHERE id_parameter = %s
        """, (bb, ba, kat, id_parameter))
        conn.commit(); cur.close(); conn.close()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ─── API: Hapus Parameter Klasifikasi ─────────────────────────────────────────
@app.route('/api/parameter/hapus/<int:id_parameter>', methods=['DELETE'])
@login_required
def api_hapus_parameter(id_parameter):
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)
        # Cek minimal 1 parameter tersisa
        cur.execute("SELECT COUNT(*) AS total FROM parameter_klasifikasi")
        total = cur.fetchone()['total']
        if total <= 1:
            cur.close(); conn.close()
            return jsonify({'ok': False, 'msg': 'Minimal harus ada 1 parameter klasifikasi.'}), 400

        cur2 = conn.cursor()
        cur2.execute("DELETE FROM parameter_klasifikasi WHERE id_parameter = %s", (id_parameter,))
        conn.commit(); cur.close(); cur2.close(); conn.close()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ─── API: Re-apply kategori ke semua data yang ada ────────────────────────────
@app.route('/api/parameter/reapply', methods=['POST'])
@login_required
def api_parameter_reapply():
    """
    Setelah perubahan parameter, re-apply kategori ke semua tabel
    yang menyimpan kategori_ipm / klasifikasi.
    """
    try:
        params = get_klasifikasi_params()
        conn = get_db(); cur = conn.cursor(dictionary=True)

        # 1. Update hasil_uji_simulasi
        cur.execute("SELECT id_prediksi, nilai_prediksi FROM hasil_uji_simulasi")
        simulasi_rows = cur.fetchall()
        for r in simulasi_rows:
            kat = get_kategori_ipm(float(r['nilai_prediksi'] or 0), params)
            cur.execute(
                "UPDATE hasil_uji_simulasi SET kategori_ipm = %s WHERE id_prediksi = %s",
                (kat, r['id_prediksi'])
            )

        # 2. Update hasil_prediksi_model
        cur.execute("SELECT id_uji, ipm_prediksi FROM hasil_prediksi_model")
        model_rows = cur.fetchall()
        for r in model_rows:
            kat = get_kategori_ipm(float(r['ipm_prediksi'] or 0), params)
            cur.execute(
                "UPDATE hasil_prediksi_model SET kategori = %s WHERE id_uji = %s",
                (kat, r['id_uji'])
            )

        conn.commit(); cur.close(); conn.close()
        return jsonify({
            'ok': True,
            'updated_simulasi': len(simulasi_rows),
            'updated_model':    len(model_rows),
        })
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# API UMUM
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/stats')
@login_required
def api_stats():
    return jsonify(get_model_stats())


@app.route('/api/wilayah/list')
@login_required
def api_wilayah_list():
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)
        cur.execute("SELECT id_wilayah, nama_wilayah FROM wilayah WHERE is_deleted = 0 ORDER BY nama_wilayah")
        rows = cur.fetchall(); cur.close(); conn.close()
        return jsonify({'ok': True, 'data': rows})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


@app.route('/api/indikator/<int:id_wilayah>/latest')
@login_required
def api_indikator_latest(id_wilayah):
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT tahun, ahh, hls, rls, pengeluaran, ipm_aktual
            FROM indikator_historis
            WHERE id_wilayah = %s
            ORDER BY tahun DESC LIMIT 1
        """, (id_wilayah,))
        row = cur.fetchone(); cur.close(); conn.close()
        if not row:
            return jsonify({'ok': False, 'msg': 'Data tidak ditemukan.'}), 404
        return jsonify({'ok': True, 'data': row})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ─── API: Ambil semua wilayah aktif + indikator terkini (untuk form tambah tahun) ──
@app.route('/api/indikator/wilayah-aktif-latest')
@login_required
def api_wilayah_aktif_latest():
    """
    Mengembalikan daftar semua wilayah aktif beserta data indikator
    tahun terbaru mereka (untuk form tambah data satu tahun penuh).
    """
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT w.id_wilayah, w.nama_wilayah,
                   ih.tahun AS tahun_terakhir,
                   ih.ahh, ih.hls, ih.rls, ih.pengeluaran, ih.ipm_aktual
            FROM wilayah w
            LEFT JOIN indikator_historis ih ON ih.id_wilayah = w.id_wilayah
                AND ih.tahun = (
                    SELECT MAX(tahun) FROM indikator_historis
                    WHERE id_wilayah = w.id_wilayah
                )
            WHERE w.is_deleted = 0
            ORDER BY w.nama_wilayah
        """)
        rows = cur.fetchall(); cur.close(); conn.close()
        return jsonify({'ok': True, 'data': rows})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# HALAMAN PUBLIK
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/home')
def publik_home():
    """
    Halaman publik utama:
    - Statistik model (MAPE, RMSE)
    - Daftar riwayat IPM dari indikator_historis (dengan filter tahun & search)
    """
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)

        # Daftar tahun tersedia — kolom year(4) dikembalikan sebagai date object oleh mysql-connector
        cur.execute("SELECT DISTINCT tahun FROM indikator_historis ORDER BY tahun DESC")
        tahun_list = [int(str(r['tahun'])[:4]) for r in cur.fetchall()]

        filter_tahun = request.args.get('tahun', tahun_list[0] if tahun_list else None, type=int)
        search       = request.args.get('search', '').strip()
        page         = max(1, request.args.get('page', 1, type=int))
        per_page     = request.args.get('per_page', 10, type=int)
        per_page     = per_page if per_page in (10, 25, 50) else 10

        where = ['w.is_deleted = 0']
        params = []
        if filter_tahun:
            where.append('ih.tahun = %s'); params.append(filter_tahun)
        if search:
            where.append('w.nama_wilayah LIKE %s'); params.append(f'%{search}%')
        where_sql = ' AND '.join(where)

        cur.execute(f"""
            SELECT COUNT(*) AS total
            FROM indikator_historis ih
            JOIN wilayah w ON ih.id_wilayah = w.id_wilayah
            WHERE {where_sql}
        """, params)
        total_rows  = cur.fetchone()['total']
        total_pages = max(1, (total_rows + per_page - 1) // per_page)
        page        = min(page, total_pages)
        offset      = (page - 1) * per_page

        cur.execute(f"""
            SELECT w.nama_wilayah, ih.tahun,
                   ih.ahh, ih.hls, ih.rls, ih.pengeluaran, ih.ipm_aktual
            FROM indikator_historis ih
            JOIN wilayah w ON ih.id_wilayah = w.id_wilayah
            WHERE {where_sql}
            ORDER BY w.nama_wilayah
            LIMIT %s OFFSET %s
        """, params + [per_page, offset])
        historis_list = cur.fetchall()
        cur.close(); conn.close()

    except Exception as e:
        logger.error(f'publik_home error: {e}\n{traceback.format_exc()}')
        tahun_list = []; historis_list = []; filter_tahun = None
        total_rows = 0; total_pages = 1; search = ''; per_page = 10; page = 1

    stats = get_model_stats()

    return render_template(
        'publik_home.html',
        stats        = stats,
        tahun_list   = tahun_list,
        filter_tahun = filter_tahun,
        search       = search,
        historis_list= historis_list,
        total_rows   = total_rows,
        total_pages  = total_pages,
        page         = page,
        per_page     = per_page,
    )


@app.route('/prediksi')
def publik_prediksi():
    """
    Halaman publik prediksi IPM:
    - Kiri: tabel hasil_prediksi_model terbaru semua wilayah
    - Kanan: form uji coba simulasi prediksi manual (3 tahun input)
    """
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)

        # Hasil prediksi model terbaru
        params = get_klasifikasi_params()
        cur.execute("""
            SELECT w.nama_wilayah, hp.ipm_prediksi, hp.ipm_aktual,
                   hp.error_persen, hp.kategori,
                   rm.tgl_latih, hp.id_model
            FROM hasil_prediksi_model hp
            JOIN wilayah w  ON hp.id_wilayah = w.id_wilayah
            JOIN riwayat_model rm ON hp.id_model = rm.id_model
            WHERE hp.id_model = (SELECT MAX(id_model) FROM hasil_prediksi_model)
            ORDER BY hp.ipm_prediksi DESC
        """)
        prediksi_list = cur.fetchall()
        for p in prediksi_list:
            if p.get('ipm_prediksi'):
                p['kategori'] = get_kategori_ipm(float(p['ipm_prediksi']), params)

        # Tahun prediksi (max tahun historis + 1) — year(4) returns date-like, pakai str()
        cur.execute("SELECT MAX(tahun) AS max_tahun FROM indikator_historis")
        row = cur.fetchone()
        tahun_prediksi = (int(str(row['max_tahun'])[:4]) + 1) if row and row['max_tahun'] else 2025

        # Daftar wilayah aktif untuk dropdown simulasi
        cur.execute("SELECT id_wilayah, nama_wilayah FROM wilayah WHERE is_deleted=0 ORDER BY nama_wilayah")
        wilayah_list = cur.fetchall()

        tgl_model = prediksi_list[0]['tgl_latih'] if prediksi_list else None
        cur.close(); conn.close()

    except Exception as e:
        logger.error(f'publik_prediksi error: {e}')
        prediksi_list = []; wilayah_list = []; tahun_prediksi = 2025; tgl_model = None

    return render_template(
        'publik_prediksi.html',
        prediksi_list  = prediksi_list,
        wilayah_list   = wilayah_list,
        tahun_prediksi = tahun_prediksi,
        tgl_model      = tgl_model,
    )


@app.route('/history')
def publik_history():
    """Halaman publik riwayat uji simulasi (hasil_uji_simulasi)."""
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)
        filter_wilayah  = request.args.get('filter_wilayah', '').strip()
        filter_kategori = request.args.get('filter_kategori', '').strip()
        search          = request.args.get('search', '').strip()
        page            = max(1, request.args.get('page', 1, type=int))
        per_page        = 10

        where = ['w.is_deleted = 0']
        params_q = []
        if filter_wilayah:
            where.append('hus.id_wilayah = %s'); params_q.append(filter_wilayah)
        if filter_kategori:
            where.append('hus.kategori_ipm = %s'); params_q.append(filter_kategori)
        if search:
            where.append('w.nama_wilayah LIKE %s'); params_q.append(f'%{search}%')
        where_sql = ' AND '.join(where)

        cur.execute(f"""
            SELECT COUNT(*) AS total FROM hasil_uji_simulasi hus
            JOIN wilayah w ON hus.id_wilayah = w.id_wilayah
            WHERE {where_sql}
        """, params_q)
        total_rows  = cur.fetchone()['total']
        total_pages = max(1, (total_rows + per_page - 1) // per_page)
        page = min(page, total_pages)
        offset = (page - 1) * per_page

        cur.execute(f"""
            SELECT hus.id_prediksi, w.nama_wilayah,
                   hus.nilai_prediksi, hus.kategori_ipm,
                   hus.tahun_prediksi, hus.tgl_simulasi
            FROM hasil_uji_simulasi hus
            JOIN wilayah w ON hus.id_wilayah = w.id_wilayah
            WHERE {where_sql}
            ORDER BY hus.tgl_simulasi DESC
            LIMIT %s OFFSET %s
        """, params_q + [per_page, offset])
        riwayat = cur.fetchall()

        cur.execute("SELECT id_wilayah, nama_wilayah FROM wilayah WHERE is_deleted=0 ORDER BY nama_wilayah")
        wilayah_list = cur.fetchall()
        cur.execute("SELECT DISTINCT kategori_ipm FROM hasil_uji_simulasi WHERE kategori_ipm IS NOT NULL")
        kategori_list = [r['kategori_ipm'] for r in cur.fetchall()]
        cur.close(); conn.close()

    except Exception as e:
        logger.error(f'publik_history error: {e}')
        riwayat = []; wilayah_list = []; kategori_list = []
        total_rows = 0; total_pages = 1; page = 1
        filter_wilayah = ''; filter_kategori = ''; search = ''

    return render_template(
        'publik_history.html',
        riwayat         = riwayat,
        wilayah_list    = wilayah_list,
        kategori_list   = kategori_list,
        filter_wilayah  = filter_wilayah,
        filter_kategori = filter_kategori,
        search          = search,
        total_rows      = total_rows,
        total_pages     = total_pages,
        page            = page,
    )


# ─── API: Indikator historis per wilayah (untuk chart di form simulasi) ────────
@app.route('/api/publik/indikator/<int:id_wilayah>')
def api_publik_indikator(id_wilayah):
    """Semua data historis satu wilayah untuk chart tren di form simulasi."""
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT tahun, ahh, hls, rls, pengeluaran, ipm_aktual
            FROM indikator_historis
            WHERE id_wilayah = %s
            ORDER BY tahun ASC
        """, (id_wilayah,))
        rows = cur.fetchall(); cur.close(); conn.close()
        return jsonify({'ok': True, 'data': rows})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


# ─── API: Simulasi prediksi publik ────────────────────────────────────────────
@app.route('/api/publik/simulasi', methods=['POST'])
def api_publik_simulasi():
    """
    Terima input 3 tahun × 4 indikator dari user publik,
    jalankan model GRU, simpan ke hasil_uji_simulasi, return hasil.
    """
    try:
        data       = request.get_json(force=True)
        id_wilayah = int(data.get('id_wilayah', 0))
        sequence   = data.get('sequence', [])   # list of 3 dicts: {tahun, ahh, hls, rls, pengeluaran}
        tahun_pred = int(data.get('tahun_prediksi', 0))

        if not id_wilayah or len(sequence) != 3 or not tahun_pred:
            return jsonify({'ok': False, 'msg': 'Data tidak lengkap. Diperlukan 3 tahun data.'}), 400

        import numpy as np
        from sklearn.preprocessing import MinMaxScaler, LabelEncoder
        from tensorflow.keras.models import load_model as keras_load

        if not os.path.exists(MODEL_PATH):
            return jsonify({'ok': False, 'msg': 'Model belum tersedia. Lakukan retraining terlebih dahulu.'}), 503

        # Load model & scaler dari DB historis (fit ulang scaler agar konsisten)
        conn = get_db(); cur = conn.cursor(dictionary=True)

        cur.execute("""
            SELECT w.nama_wilayah, ih.ahh, ih.hls, ih.rls, ih.pengeluaran, ih.ipm_aktual
            FROM indikator_historis ih
            JOIN wilayah w ON ih.id_wilayah = w.id_wilayah
            WHERE w.is_deleted = 0 AND ih.ipm_aktual IS NOT NULL
            ORDER BY w.nama_wilayah, ih.tahun
        """)
        all_rows = cur.fetchall()

        import pandas as pd
        df_all = pd.DataFrame(all_rows)
        features = ['ahh', 'hls', 'rls', 'pengeluaran', 'wilayah_enc']

        le = LabelEncoder()
        df_all['wilayah_enc'] = le.fit_transform(df_all['nama_wilayah'])
        scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
        scaler_X.fit(df_all[features])
        scaler_y.fit(df_all[['ipm_aktual']])

        # Nama wilayah untuk encode
        cur.execute("SELECT nama_wilayah FROM wilayah WHERE id_wilayah = %s", (id_wilayah,))
        wil_row = cur.fetchone()
        if not wil_row:
            cur.close(); conn.close()
            return jsonify({'ok': False, 'msg': 'Wilayah tidak ditemukan.'}), 404
        nama_wil = wil_row['nama_wilayah']

        try:
            nama_enc = le.transform([nama_wil])[0]
        except Exception:
            cur.close(); conn.close()
            return jsonify({'ok': False, 'msg': f'Wilayah "{nama_wil}" tidak ada dalam data training.'}), 400

        # Build input sequence
        seq_data = []
        for s in sequence:
            seq_data.append([
                float(s.get('ahh', 0)),
                float(s.get('hls', 0)),
                float(s.get('rls', 0)),
                float(s.get('pengeluaran', 0)),
                float(nama_enc),
            ])
        X_input = scaler_X.transform(seq_data)
        X_input = X_input.reshape(1, 3, len(features))

        # Load model & predict
        model   = keras_load(MODEL_PATH)
        y_scaled = model.predict(X_input, verbose=0)
        y_pred   = float(scaler_y.inverse_transform(y_scaled)[0][0])

        # Kategori berdasarkan parameter DB
        params_klas = get_klasifikasi_params()
        kategori    = get_kategori_ipm(y_pred, params_klas)

        # Ambil id_model terbaru dari riwayat_model
        cur.execute("SELECT MAX(id_model) AS id_model_terbaru FROM riwayat_model")
        row_model = cur.fetchone()
        id_model_terbaru = row_model['id_model_terbaru'] if row_model and row_model['id_model_terbaru'] else None

        # Simpan ke hasil_uji_simulasi
        import datetime, json as json_lib
        now = datetime.datetime.now()
        data_seq_json = json_lib.dumps(sequence)

        if id_model_terbaru:
            cur.execute("""
                INSERT INTO hasil_uji_simulasi
                    (id_model, id_wilayah, data_sequence, tahun_prediksi, nilai_prediksi, kategori_ipm, tgl_simulasi)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (id_model_terbaru, id_wilayah, data_seq_json, tahun_pred, round(y_pred, 4), kategori, now))
        else:
            cur.execute("""
                INSERT INTO hasil_uji_simulasi
                    (id_wilayah, data_sequence, tahun_prediksi, nilai_prediksi, kategori_ipm, tgl_simulasi)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (id_wilayah, data_seq_json, tahun_pred, round(y_pred, 4), kategori, now))
        new_id = cur.lastrowid
        conn.commit(); cur.close(); conn.close()

        # Historis untuk chart tren
        conn2 = get_db(); cur2 = conn2.cursor(dictionary=True)
        cur2.execute("""
            SELECT tahun, ipm_aktual FROM indikator_historis
            WHERE id_wilayah = %s ORDER BY tahun ASC
        """, (id_wilayah,))
        historis_chart = cur2.fetchall(); cur2.close(); conn2.close()

        return jsonify({
            'ok':           True,
            'id_prediksi':  new_id,
            'nilai_prediksi': round(y_pred, 4),
            'kategori':     kategori,
            'tahun_prediksi': tahun_pred,
            'nama_wilayah': nama_wil,
            'historis':     historis_chart,
        })

    except Exception as e:
        logger.error(f'simulasi error: {traceback.format_exc()}')
        return jsonify({'ok': False, 'msg': f'Terjadi kesalahan: {str(e)}'}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'ok': False, 'msg': 'Endpoint tidak ditemukan.'}), 404
    return render_template('login.html'), 404


@app.errorhandler(500)
def internal_error(e):
    logger.error(f'Internal error: {e}')
    if request.path.startswith('/api/'):
        return jsonify({'ok': False, 'msg': 'Kesalahan server internal.'}), 500
    return render_template('login.html'), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)