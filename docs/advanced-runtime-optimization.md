# Optimasi runtime fase kedua

OpenVINO/ONNX dapat diuji khusus CPU setelah tersedia fixture regresi bounding box, embedding, feature vector, probabilitas dan keputusan akhir. FP16, quantization, TensorRT, multiprocessing, banyak Uvicorn worker, pengurangan 32 frame, dan penurunan resolusi tidak boleh menjadi jalur produksi tanpa pembuktian kesetaraan. Celery/Redis hanya relevan untuk antrean lintas proses dan bukan percepatan model langsung.

