# Setup Docker manual

Codex tidak menjalankan perintah ini. Salin `.env.example` ke `.env`, lalu:

```powershell
docker network inspect deepfake-net
docker build -t deepfake-backend:latest .
docker compose up -d
docker compose ps
docker compose logs -f backend
```

Harapan: Torch CPU dipasang dari index resmi, container `deepfake-backend` menjalankan satu worker port 5000 dan menjadi healthy setelah model termuat. Bila network belum ada: `docker network create deepfake-net`. Error artefak berarti mount/path salah; error pickle mengarah ke kompatibilitas versi; exit 137 biasanya RAM habis.

Frontend satu network memakai `http://deepfake-backend:5000`; frontend Windows host memakai `http://host.docker.internal:5000` atau localhost sesuai arah request.

```powershell
curl.exe http://localhost:5000/health
curl.exe -i http://localhost:5000/ready
curl.exe http://localhost:5000/models/status
curl.exe -X POST -F "video=@C:\path\sample.mp4;type=video/mp4" http://localhost:5000/predict-video
```

Harapan: health 200; ready 200 saat siap atau 503 dengan alasan; status menunjukkan V21/0.5; prediksi mengembalikan payload Flask-compatible.

