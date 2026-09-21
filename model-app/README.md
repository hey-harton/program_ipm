---
title: IPM Jatim AI Service
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# IPM Jatim - BiGRU Model & AI Microservice

Microservice backend untuk inferensi model Deep Learning BiGRU dan Retraining data IPM 38 Kabupaten/Kota Jawa Timur.
Dihubungkan ke frontend di Vercel dan database di Supabase.

### Endpoints
- `GET /`: Healthcheck & API status
- `POST /predict`: Melakukan inferensi peramalan IPM satu kabupaten
- `POST /retrain`: Menjalankan pelatihan ulang model BiGRU
- `GET /retrain-status`: Cek progress retraining secara real-time
