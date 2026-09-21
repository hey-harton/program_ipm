import os
import psycopg2
import re

SUPABASE_URL = "postgres://postgres.vmddpyryjnkewvswlhuy:rIrX6zq0bwXpQTkY@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres?sslmode=require"

def main():
    conn = psycopg2.connect(SUPABASE_URL)
    cur = conn.cursor()

    sql_path = os.path.join(os.path.dirname(__file__), "program_ipm_jatim_2.sql")
    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()

    # Ekstrak baris wilayah
    m = re.findall(r"\(\d+,\s*'([^']+)',\s*(NULL|'[^']*'),\s*(NULL|'[^']*'),\s*(NULL|'[^']*'),\s*(\d)\)", sql)
    updated = 0
    for nama, desk, logo, land, is_del in m:
        if desk != "NULL":
            desk_val = desk.strip("'")
            logo_val = logo.strip("'") if logo != "NULL" else None
            land_val = land.strip("'") if land != "NULL" else None
            cur.execute("""
                UPDATE wilayah 
                SET deskripsi = %s,
                    url_logo = COALESCE(%s, url_logo),
                    url_landmark = COALESCE(%s, url_landmark)
                WHERE LOWER(nama_wilayah) = LOWER(%s)
            """, (desk_val, logo_val, land_val, nama))
            updated += 1
            print(f"Updated description for: {nama}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"Berhasil mengupdate {updated} deskripsi wilayah ke Supabase!")

if __name__ == "__main__":
    main()
