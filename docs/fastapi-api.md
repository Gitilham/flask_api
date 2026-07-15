# API FastAPI V21

Swagger `/docs`; ReDoc `/redoc`. Endpoint: `GET /health`, `/ready`, `/models/status`, `/model-info`, `/`; `POST /predict-video` dan alias `/api/v1/predict-video`.

Upload memakai multipart field `video`; ekstensi mp4/avi/mov/mkv, MIME video terkait atau `application/octet-stream` untuk client lama, default maksimum 300 MB.

Respons sukses mempertahankan property Flask, termasuk prediction/label/status/confidence/scores/threshold/margin/model_version/frames. Keputusan akhir adalah `<50% REAL`, `=50% MENCURIGAKAN`, dan `>50% DEEPFAKE`. `NO_FACE` tetap success seperti Flask. Error berisi `success=false,status=error,message,error_code`; kode meliputi VIDEO_REQUIRED, EMPTY_FILE, FILE_TOO_LARGE, INVALID_EXTENSION, INVALID_MIME, CORRUPT_VIDEO, NO_FRAMES, MODEL_NOT_READY, QUEUE_FULL, QUEUE_TIMEOUT, INTERNAL_ERROR. Traceback tidak dikirim. Header `X-Request-ID` selalu diberikan.
