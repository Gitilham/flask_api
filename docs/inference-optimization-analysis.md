# Audit optimasi inferensi

Pipeline: temporary video → OpenCV sampling 32 frame → YOLO/crop dalam RAM → satu batch Xception → agregasi embedding/basic/artifact → scaler pada basic per frame → classifier → satu base flip bila payload memintanya → local similarity → keputusan terpusat.

Bottleneck utama adalah YOLO yang saat ini dipanggil sekali per frame (32 call untuk 32 frame). Xception sebelumnya dan sekarang sudah satu pemanggilan untuk seluruh crop, kini dengan batch size internal 8. Tidak ada JPG/crop/embedding intermediate yang ditulis; disk hanya dipakai untuk video upload sementara dan selalu dibersihkan.

Registry lifespan memuat config, classifier, reference bank, scaler, YOLO, dan Xception satu kali. Tidak ada flip pada prediction service atau response frontend. Debug print dalam loop telah dihapus dan timing agregat ditambahkan.

Batching YOLO belum diaktifkan pada jalur utama karena perubahan bentuk input Ultralytics dapat mengubah pemetaan hasil/bounding box dan belum tersedia video fixture untuk regression test. Optimasi ini sengaja ditahan demi kualitas/identitas output. FP16, ONNX, OpenVINO, quantization, pengurangan frame dan resolusi juga tidak diterapkan.

