# Pengujian dan regresi

Test dibuat tetapi belum dijalankan. Sesudah dependency tersedia, jalankan manual `pytest -q`. Suite memeriksa status API, validasi upload, cleanup, respons, error aman, artefak/shape, matematika dan kesamaan fungsi algoritma.

Regresi video memerlukan Flask legacy dan FastAPI pada port berbeda serta sample asli:

```powershell
python scripts/compare_flask_fastapi.py C:\path\sample.mp4 --flask-url http://127.0.0.1:5001 --fastapi-url http://127.0.0.1:5000
```

Harapan: `compatible: true`, label/status/frame/shape identik, score berbeda maksimal 1e-6. Toleransi 1e-4 hanya bila perbedaan runtime numerik terdokumentasi; perubahan preprocessing/label tidak boleh diterima.

