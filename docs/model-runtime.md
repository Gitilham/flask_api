# Runtime model V21

Lifespan membuat satu registry, memvalidasi lima artefak, memuat payload/scaler/YOLO/Xception satu kali, lalu memvalidasi shape 21.250, threshold 0.5, label map dan correction config. Gunakan satu Uvicorn worker agar model tidak digandakan di RAM.

Alur inferensi tetap: validasi video, sampling, YOLO, Xception float32 0–255, agregasi fitur, classifier, local similarity. Tidak ada batching baru, FP16, quantization, ONNX, atau perubahan resolusi.

Default `INFERENCE_CONCURRENCY=1`; request menunggu semaphore di luar event loop. Timeout dan antrean diatur `INFERENCE_QUEUE_TIMEOUT`/`MAX_UPLOAD_QUEUE`. Thread CPU konservatif. File sementara dibersihkan kecuali `PRESERVE_UPLOADS=true`.

Celery/Redis kelak dapat menjadi adapter yang memanggil `prediction_service.predict_video`; kontrak prediksi tidak perlu berubah.

